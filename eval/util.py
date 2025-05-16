import os
import re
import glob
import math
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from ortools.graph.python import min_cost_flow
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

#############################################
# Roles
#############################################

def metric_multi_label_f1(gold_annotations, pred_annotations, field):
    total_prec = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    for g, p in zip(gold_annotations, pred_annotations):
        gold_set = set(g.get(field, []))
        pred_set = set(p.get(field, []))

        if not gold_set and not pred_set:  # Both empty → perfect match
            prec, recall, f1 = 1.0, 1.0, 1.0
        elif not gold_set or not pred_set:  # One is empty → total failure
            prec, recall, f1 = 0.0, 0.0, 0.0
        else:
            intersection = len(gold_set & pred_set)
            prec = intersection / len(pred_set) if pred_set else 0
            recall = intersection / len(gold_set) if gold_set else 0
            f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0

        total_prec += prec
        total_recall += recall
        total_f1 += f1
        count += 1

    return (
        100 * total_prec / count if count > 0 else 0,
        100 * total_recall / count if count > 0 else 0,
        100 * total_f1 / count if count > 0 else 0
    )

def metric_pair_f1(gold_pair_dict, pred_pair_dict):
    total_prec = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    common_line_idxs = sorted(set(gold_pair_dict.keys()).intersection(set(pred_pair_dict.keys())))
    
    if not common_line_idxs:
        return None
    
    aligned_gold = [gold_pair_dict[i] for i in common_line_idxs]
    aligned_pred = [pred_pair_dict[i] for i in common_line_idxs]    

    total_gold = len(aligned_gold)
    total_auto = len(aligned_pred)    
    matched = sum(1 for g, p in zip(aligned_gold, aligned_pred) if g == p)
        
    p = 0.0
    if total_auto > 0:
        p = 100 * matched / total_auto
    r = 0.0
    if total_gold > 0:
        r = 100 * matched / total_gold
    f = 0.0
    if matched > 0:
        f = 2 * p * r / (p + r)
        
    return p, r, f

def metric_precision_at_1(gold_annotations, pred_annotations, field):
    correct = sum(1 for g, p in zip(gold_annotations, pred_annotations) if g.get(field) == p.get(field))
    total = len(pred_annotations)  # Precision: correct / total predictions
    
    return 100 * correct / total if total > 0 else 0

def metric_accuracy(gold_annotations, pred_annotations, field):
    """
    Computes Accuracy for single-label fields (e.g., "speaker").
    """
    return metric_precision_at_1(gold_annotations, pred_annotations, field)

#############################################
# Conversational thread
# -------------------------------------------
# code adapated from
# https://github.com/jkkummerfeld/irc-disentanglement/blob/master/tools/format-conversion/graph-to-cluster.py
#############################################

def build_clusters(utterances):
    def find(x, parents):
        while parents[x] != x:
            parent = parents[x]
            parents[x] = parents[parent]
            x = parent
        return x
    
    def union(x, y, parents, sizes):
        # Get the representative for their sets
        x_root = find(x, parents)
        y_root = find(y, parents)
     
        # If equal, no change needed
        if x_root == y_root:
            return
     
        # Otherwise, merge them
        if sizes[x_root] > sizes[y_root]:
            parents[y_root] = x_root
            sizes[x_root] += sizes[y_root]
        else:
            parents[x_root] = y_root
            sizes[y_root] += sizes[x_root]
    
    def union_find(nodes, edges):
        # Make sets
        parents = {n:n for n in nodes}
        sizes = {n:1 for n in nodes}
    
        for edge in edges:
            union(edge[0], edge[1], parents, sizes)
    
        clusters = {}
        for n in parents:
            clusters.setdefault(find(n, parents), set()).add(n)
        return clusters 

    pair_dicts = {}
    edges = []
    nodes = set()
    for utterance in utterances:
        # print('u -- ', utterance)
        parts = [utterance['line_idx'], utterance['reply_to']]
        pair_dicts.update(
            {utterance['line_idx']: utterance['reply_to']}
        )
        source = max(parts)
        nodes.add(source)
        parts.remove(source)
        for num in parts:
            edges.append((source, num))
            nodes.add(num)

    clusters = union_find(nodes, edges)
    clusters = [cluster for _, cluster in clusters.items()]
    return pair_dicts, clusters
    
def clusters_to_contingency(gold, auto):
    # A table, in the form of:
    # https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    table = {}
    for i, acluster in enumerate(auto):
        aname = f"auto.{i}"
        current = {}
        table[aname] = current
        for j, gcluster in enumerate(gold):
            gname = f"gold.{j}"
            count = len(acluster.intersection(gcluster))
            if count > 0:
                current[gname] = count
    counts_a = {}
    for i, acluster in enumerate(auto):
        aname = f"auto.{i}"
        counts_a[aname] = len(acluster)
    counts_g = {}
    for i, gcluster in enumerate(gold):
        gname = f"gold.{i}"
        counts_g[gname] = len(gcluster)
    return table, counts_a, counts_g

##########################################
# Standard Clustering Metrics
##########################################

def metric_variation_of_information(gold, auto):
    contingency, row_sums, col_sums = clusters_to_contingency(gold, auto)    
    total = 0.0
    for row in row_sums:
        total += row_sums[row]

    if total <= 1:
        return 100.0
    
    H_UV = 0.0
    I_UV = 0.0
    for row in contingency:
        for col in contingency[row]:
            num = contingency[row][col]
            H_UV -= (num / total) * math.log(num / total, 2)
            I_UV += (num / total) * math.log(num * total / (row_sums[row] * col_sums[col]), 2)

    H_U = 0.0
    for row in row_sums:
        num = row_sums[row]
        H_U -= (num / total) * math.log(num / total, 2)
    H_V = 0.0
    for col in col_sums:
        num = col_sums[col]
        H_V -= (num / total) * math.log(num / total, 2)

    # max_score = math.log(total, 2)
    # VI = H_UV - I_UV
    # scaled_VI = VI / max_score
    
    # return 100 - 100 * scaled_VI    
    # # print("{:5.2f}   1 - Scaled VI".format())    

    max_score = math.log(total, 2) if total > 1 else 1  # Avoid log(1) = 0
    VI = H_UV - I_UV
    scaled_VI = VI / max_score if max_score != 0 else 0

    return 100 - 100 * scaled_VI

def metric_exact_match(gold, auto):
    # def exact_match(gold, auto, skip_single=True):
    # P/R/F over complete clusters
    total_gold = 0
    total_matched = 0
    for cluster in gold:
        total_gold += 1
        matched = False
        for ocluster in auto:
            if len(ocluster.symmetric_difference(cluster)) == 0:
                matched = True
                break
        if matched:
            total_matched += 1
    match = []
    subsets = []
    supersets = []
    other = []
    prefix = []
    suffix = []
    gap_free = []
    match_counts = []
    subsets_counts = []
    supersets_counts = []
    other_counts = []
    prefix_counts = []
    suffix_counts = []
    gap_free_counts = []
    total_auto = 0
    for cluster in auto:
        # if skip_single and len(cluster) == 1:
        #     continue
        total_auto += 1
        most_overlap = 0
        fraction = 0
        count = 0
        is_subset = False
        is_superset = False
        is_prefix = False
        is_suffix = False
        is_gap_free = False
        is_match = False
        for ocluster in gold:
            if len(ocluster.symmetric_difference(cluster)) == 0:
                is_match = True
                break

            overlap = len(ocluster.intersection(cluster))
            if overlap > most_overlap:
                most_overlap = overlap
                gaps = False
                for v in ocluster:
                    if min(cluster) <= v <= max(cluster):
                        if v not in cluster:
                            gaps = True
                fraction = 1 - (overlap / len(ocluster.union(cluster)))
                count = len(ocluster.union(cluster)) - overlap

                is_subset = (overlap == len(cluster))
                is_superset = (overlap == len(ocluster))
                if overlap == len(cluster) and (not gaps):
                    is_gap_free = True
                    if min(ocluster) == min(cluster):
                        is_prefix = True
                    if max(ocluster) == max(cluster):
                        is_suffix = True
        if is_match:
            match.append(fraction)
            match_counts.append(count)
        elif is_superset:
            supersets.append(fraction)
            supersets_counts.append(count)
        elif is_subset:
            subsets.append(fraction)
            subsets_counts.append(count)
            if is_prefix:
                prefix.append(fraction)
                prefix_counts.append(count)
            elif is_suffix:
                suffix.append(fraction)
                suffix_counts.append(count)
            elif is_gap_free:
                gap_free.append(fraction)
                gap_free_counts.append(count)
        else:
            other.append(fraction)
            other_counts.append(count)

    p, r, f = 0.0, 0.0, 0.0
    if total_auto > 0:
        p = 100 * total_matched / total_auto
    if total_gold > 0:
        r = 100 * total_matched / total_gold
    if total_matched > 0:
        f = 2 * p * r / (p + r)
    return p, r, f
    

def metric_one_to_one(gold, auto):
    contingency, row_sums, col_sums = clusters_to_contingency(gold, auto)        
    row_to_num = {}
    col_to_num = {}
    num_to_row = []
    num_to_col = []
    for row_num, row in enumerate(row_sums):
        row_to_num[row] = row_num
        num_to_row.append(row)
    for col_num, col in enumerate(col_sums):
        col_to_num[col] = col_num
        num_to_col.append(col)

    min_cost_flow_f = min_cost_flow.SimpleMinCostFlow()
    start_nodes = []
    end_nodes = []
    capacities = []
    costs = []
    source = len(num_to_row) + len(num_to_col)
    sink = len(num_to_row) + len(num_to_col) + 1
    supplies = []
    tasks = min(len(num_to_row), len(num_to_col))
    for row, row_num in row_to_num.items():
        start_nodes.append(source)
        end_nodes.append(row_num)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    for col, col_num in col_to_num.items():
        start_nodes.append(col_num + len(num_to_row))
        end_nodes.append(sink)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    supplies.append(tasks)
    supplies.append(-tasks)
    for row, row_num in row_to_num.items():
        for col, col_num in col_to_num.items():
            cost = 0
            if col in contingency[row]:
                cost = - contingency[row][col]
            start_nodes.append(row_num)
            end_nodes.append(col_num + len(num_to_row))
            capacities.append(1)
            costs.append(cost)

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow_f.add_arc_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])
  
    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow_f.set_node_supply(i, supplies[i])

    # Find the minimum cost flow.
    min_cost_flow_f.solve()

    # Score.
    total_count = sum(v for _, v in row_sums.items())
    overlap = 0
    for arc in range(min_cost_flow_f.num_arcs()):
        # Can ignore arcs leading out of source or into sink.
        if min_cost_flow_f.tail(arc)!=source and min_cost_flow_f.head(arc)!=sink:
            # Arcs in the solution have a flow value of 1. Their start and end nodes
            # give an assignment of worker to task.
            if min_cost_flow_f.flow(arc) > 0:
                row_num = min_cost_flow_f.tail(arc)
                col_num = min_cost_flow_f.head(arc)
                col = num_to_col[col_num - len(num_to_row)]
                row = num_to_row[row_num]
                if col in contingency[row]:
                    overlap += contingency[row][col]
    return overlap * 100 / total_count

def parse_interlocutor(s):
    INTERLOCUTOR_MAP = {}
    def extract_full_inner(raw: str) -> str:
        start = raw.find('(')
        if start == -1:
            return raw.strip()
    
        depth = 0
        # iterate from first "(" forward
        for idx, ch in enumerate(raw[start:], start):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                # when we close the outermost "("
                if depth == 0:
                    # slice out everything between them
                    return raw[start+1 : idx].strip()
    
        # no matching ")" found
        return raw.strip()

    s = s.strip()
    if not s:
        return None
    if '_OS' in s or '(OS)' in s:  # fair comparison - ignore off screen characters
        return None
        
    if s in INTERLOCUTOR_MAP:
        return INTERLOCUTOR_MAP[s]
    raw = s

    if " (" in raw: # from annotations
        interloc_name = extract_full_inner(raw)
    else:
        interloc_name = raw.strip()    

    interloc_name = interloc_name.lower()
    INTERLOCUTOR_MAP[s] = interloc_name.lower()
    return interloc_name

def parse_multiple_interlocutors(s, mapping=None):
    if not s.strip():
        return []
    return [parse_interlocutor(x, mapping=mapping) for x in s.split(",") if x.strip()]

def process_pred_entries(data, prefix=None, file_path=None):
    utterances = []
    roles = data.get("clip_roles", [])
    for i, role in enumerate(roles):

        if not role:
            continue
            
        if not isinstance(role, dict):
            role = role.strip()
            if not role:
                continue

            role = json.loads(role)

        if "line_index" in role:
            line_index = int(role['line_index']) + 1
        elif "line_idx" in role:
            line_index = int(role['line_idx']) + 1   
            # print('woo')
        else:
            line_index = i + 1

        # Normalize and process speaker

        if not isinstance(role, dict):
            continue        
        raw_speaker = role.get("speaker", "").strip()
        if raw_speaker.lower() == "unknown":
            raw_speaker = "unk"
        if raw_speaker.lower() == "none":
            raw_speaker = ""
        speaker = parse_interlocutor(raw_speaker) if raw_speaker else None

        # Normalize and process addressees
        
        raw_addressees = role.get("addressees", [])
        if not raw_addressees:
            raw_addressees = role.get("addressee", [])
            
        if not isinstance(raw_addressees, list):
            raw_addressees = [raw_addressees]
        processed_addressees = []
        for a in raw_addressees:
            a = a.strip()
            if a.lower() == "none":
                continue
            if a.lower() == "unknown":
                a = "unk"
            processed_addressees.append(parse_interlocutor(a))
        
        # Normalize and process side participants
        raw_side = role.get("side_participants", [])
        if not raw_side:
            raw_side = role.get("side_participant", [])
            
        if not isinstance(raw_side, list):
            raw_side = [raw_side]
        processed_side = []
        for a in raw_side:
            a = a.strip()
            if a.lower() == "none":
                continue
            if a.lower() == "unknown":
                a = "unk"
            processed_side.append(parse_interlocutor(a))
        
        # Use reply_to from the JSON directly (expected to be an integer)
        # reply_to = role.get("reply_to")
        if str(role["reply_to"]) in ["None", "none"]:
            reply_to = int(line_index)
        else:
            reply_to = int(role.get("reply_to", line_index))
        
        utterance = {
            "line_idx": line_index,
            "speaker": speaker,
            "addressee": processed_addressees,
            "side_participant": processed_side,
            "reply_to": reply_to + 1
        }
        utterances.append(utterance)
    return utterances
    
def process_pred_file(file_path, file_prefix=None):
    """
    Reads a JSON prediction file.
    Expected format:
      {
        "clip_roles": [
          {
            "line_index": 0,
            "speaker": "Leslie Winkle",
            "addressees": ["Leonard Hofstadter"],
            "side_participants": ["Penny", "Eric"],
            "reply_to": 0   # Provided as an integer (always ≤ line_index)
          },
          ...
        ]
      }
    Converts this to a list of utterance dictionaries.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)

        except Exception as e:
            print(e)
            print(file_path)
            # print(f.read())
            return 
        
    return process_pred_entries(data, file_prefix, file_path)

def derive_clip_and_show_pred(file_path):
    parts = file_path.split(os.sep)
    if len(parts) < 2:
        return None, None, None
    
    model = parts[-2]
    filename = parts[-1]

    # Split at the last '.' to isolate model ID and clip ID
    clip_part, _ = os.path.splitext(filename)
    if '.' in clip_part:
        clip_id, model_id = clip_part.split('.', 1)
    else:
        clip_id, model_id = clip_part, ""

    if re.match(r"^s\d{2}", clip_id):
        show_id = "bbt"
    else:
        show_id = clip_id.split("_")[0]

    return model_id, clip_id, show_id

def normalize_utterance(utt):
    """
    Normalizes an utterance dictionary for comparison:
      - Uses only the name part from interlocutor tuples for speaker, addressee, and side_participant.
      - Drops fields like start, end, and text.
    """
    norm = {}
    norm["line_idx"] = utt.get("line_idx")
    
    # For speaker: use the second element if present and non-empty; otherwise, the first element.
    sp = utt.get("speaker")
    if isinstance(sp, tuple):
        norm["speaker"] = sp[1].strip() if sp[1].strip() else sp[0].strip()
    else:
        norm["speaker"] = sp

    # For addressee: convert each tuple to its name part.
    norm["addressee"] = []
    for a in utt.get("addressee", []):
        if isinstance(a, tuple):
            norm["addressee"].append(a[1].strip() if a[1].strip() else a[0].strip())
        else:
            norm["addressee"].append(a)
    
    # For side_participant: same as addressee.
    norm["side_participant"] = []
    for a in utt.get("side_participant", []):
        if isinstance(a, tuple):
            norm["side_participant"].append(a[1].strip() if a[1].strip() else a[0].strip())
        else:
            norm["side_participant"].append(a)
    
    # For reply_to, leave it as is.
    norm["reply_to"] = utt["reply_to"]
    return norm

def compute_pred_vs_gold_agreement(clip_data):
    """
    Given clip_data with keys "gold" and "pred", computes agreement metrics comparing
    gold annotations with predictions.
    
    This function:
      1. Merges gold and prediction utterances.
      2. Normalizes each utterance (using only the name for speaker, addressee, and side_participant).
      3. Aligns them by their line_idx.
    
    Returns a dict with metrics for each field.
    """     
    # Normalize utterances by retaining only names and line_idx.
    norm_gold = {utt["line_idx"]: normalize_utterance(utt) for utt in clip_data["gold"]}
    norm_pred = {utt["line_idx"]: normalize_utterance(utt) for utt in clip_data["pred"]}
    
    # Get the common line indices.
    common_line_idxs = sorted(set(norm_gold.keys()).intersection(set(norm_pred.keys())))
    
    if not common_line_idxs:
        return None
    
    aligned_gold = [norm_gold[i] for i in common_line_idxs]
    aligned_pred = [norm_pred[i] for i in common_line_idxs]
    
    gold_pred_dict = {"gold": aligned_gold, "pred": aligned_pred}
    
    results = {}
    for field in ["speaker", "addressee", "side_participant", "reply_to"]:
        pair_scores = compute_performance(gold_pred_dict, field)
        results[field] = pair_scores.get("gold vs pred", None)
    return results

def compute_performance(gold_pred_dict, field):
    scores = {}

    gold_dict = gold_pred_dict["gold"]
    pred_dict = gold_pred_dict["pred"]

    pair_key = "gold vs pred" # just for readability ...

    if field == "speaker":
        accuracy = metric_accuracy(gold_dict, pred_dict, field)

        scores[pair_key] = {
            "Accuracy": accuracy,
        }

    elif field in ["addressee", "side_participant"]:
        precision, recall, f1 = metric_multi_label_f1(gold_dict, pred_dict, field)
        scores[pair_key] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    elif field == "reply_to":    
        gold_pairs, gold_clusters = build_clusters(gold_dict)
        pred_pairs, pred_clusters = build_clusters(pred_dict)

        nvi = metric_variation_of_information(gold_clusters, pred_clusters)  # Normalized VI -- bigger stronger better 
        pairwise_precision, pairwise_recall, pairwise_f1 = metric_pair_f1(gold_pairs, pred_pairs)
        one2one = metric_one_to_one(gold_clusters, pred_clusters)
        exact_precision, exact_recall, exact_f1 = metric_exact_match(gold_clusters, pred_clusters)

        scores[pair_key] = {
            "NVI": nvi,
            "Pairwise F1 F1": pairwise_f1,
            "One-to-One": one2one,
            "Exact Match F1": exact_f1
        }

    return scores

#############################################
# Aggregating Agreement Metrics Across Clips and by Show
#############################################
def aggregate_clip_metrics(clip_metrics, field):
    all_vals = defaultdict(list)
    for clip_id, data in clip_metrics.items():
        field_metrics = data["metrics"].get(field)
        if field_metrics is None:
            continue
        for metric, value in field_metrics.items():
            if value is not None:
                all_vals[metric].append(value)
    if not all_vals:
        return None
    return {metric: sum(vals) / len(vals) for metric, vals in all_vals.items()}

def bootstrap_show_field(metrics_list, field, num_bootstrap=10000, ci=95):
    boot_results = defaultdict(list)
    n = len(metrics_list)
    for i in range(num_bootstrap):
        # Sample with replacement from metrics_list
        sample = [random.choice(metrics_list) for _ in range(n)]
        agg = aggregate_by_show(sample, field)
        if agg is not None:
            for metric, value in agg.items():
                boot_results[metric].append(value)
    
    ci_results = {}
    # Compute lower and upper percentiles for each sub-metric
    for metric, values in boot_results.items():
        lower = np.percentile(values, (100 - ci) / 2)
        upper = np.percentile(values, 100 - (100 - ci) / 2)
        ci_results[metric] = (lower, upper)
    return ci_results

def aggregate_by_show(metrics_list, field):
    all_vals = defaultdict(list)
    for m in metrics_list:
        field_metrics = m.get(field)
        if field_metrics is None:
            continue
        for metric, value in field_metrics.items():
            if value is not None:
                all_vals[metric].append(value)
    if not all_vals:
        return None
    return {metric: sum(vals) / len(vals) for metric, vals in all_vals.items()}

#############################################
# Print Aggregated Metrics
#############################################
def print_aggregated_metrics(overall, by_show):
    print("=== Overall Agreement Metrics (Averaged Across All Clips) ===")
    for field, values in overall.items():
        print(f"{field}:")
        if isinstance(values, dict):
            for sub_metric, val in values.items():
                print(f"  {sub_metric}: {val:.3f}")
        else:
            print(f"  {values:.3f}")
        print()
    
    print("=== Agreement Metrics Aggregated by Show ===")
    for show, metrics in by_show.items():
        print(f"Show {show}:")
        for field, values in metrics.items():
            print(f"  {field}:")
            if isinstance(values, dict):
                for sub_metric, val in values.items():
                    print(f"    {sub_metric}: {val:.3f}")
            else:
                print(f"    {values:.3f}")
        print()

# Print Overall Aggregated Metrics and Bootstrapped Confidence Intervals by Show:
def print_aggregated_metrics(overall, by_show, bootstrap_by_show):
    print("=== Overall Agreement Metrics (Averaged Across All Clips) ===")
    for field, values in overall.items():
        print(f"{field}:")
        if isinstance(values, dict):
            for sub_metric, val in values.items():
                print(f"  {sub_metric}: {val:.3f}")
        else:
            print(f"  {values:.3f}")
        print()
    
    print("=== Agreement Metrics Aggregated by Show ===")
    for show, metrics in by_show.items():
        print(f"Show {show}:")
        for field, values in metrics.items():
            print(f"  {field}:")
            if isinstance(values, dict):
                for sub_metric, val in values.items():
                    ci = bootstrap_by_show[show].get(field, {}).get(sub_metric, (None, None))
                    print(f"    {sub_metric}: {val:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")
            else:
                print(f"    {values:.3f}")
        print()

def bootstrap_overall(by_show, bootstrap_by_show):
    overall_bootstrap = {}
    fields = ["speaker", "addressee", "side_participant", "reply_to"]
    for field in fields:
        # Initialize a dictionary to collect values per sub-metric across shows.
        sub_metric_values = {}
        for show_id in by_show:
            # Ensure the current show has data for the field.
            if field in by_show[show_id] and by_show[show_id][field] is not None:
                for sub_metric, mean_value in by_show[show_id][field].items():
                    # Ensure the bootstrapped intervals exist for this sub-metric.
                    if (show_id in bootstrap_by_show and
                        field in bootstrap_by_show[show_id] and
                        sub_metric in bootstrap_by_show[show_id][field]):
                        lower, upper = bootstrap_by_show[show_id][field][sub_metric]
                        if sub_metric not in sub_metric_values:
                            sub_metric_values[sub_metric] = {"means": [], "lowers": [], "uppers": []}
                        sub_metric_values[sub_metric]["means"].append(mean_value)
                        sub_metric_values[sub_metric]["lowers"].append(lower)
                        sub_metric_values[sub_metric]["uppers"].append(upper)
        # Compute the average values for each sub-metric across shows.
        overall_bootstrap[field] = {}
        for sub_metric, values in sub_metric_values.items():
            overall_mean = np.mean(values["means"])
            overall_lower = np.mean(values["lowers"])
            overall_upper = np.mean(values["uppers"])
            overall_bootstrap[field][sub_metric] = (overall_mean, overall_lower, overall_upper)
    return overall_bootstrap

def print_bootstrap_overall(overall_bootstrap):
    print("=== Overall Bootstrapped Agreement Metrics (Averaged Across Shows) ===")
    for field, sub_metrics in overall_bootstrap.items():
        print(f"{field}:")
        for sub_metric, (mean, lower, upper) in sub_metrics.items():
            print(f"  {sub_metric}: {mean:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
        print()        