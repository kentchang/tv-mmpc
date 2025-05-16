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

import util  # you shold have this in your folder

ANNOTATION_ROOT = Path('data') # <- see README.md
PRED_ROOT = Path("inference/output")  # <- output path from inference, example

#############################################
# Load gold annotations
#############################################

gold_paths = list(ANNOTATION_ROOT.glob("*.annotation.json"))
annotations_by_clip = {}
for p in gold_paths:
    clip_id = p.stem.replace(".annotation", "")
    with p.open() as f:
        annotations_by_clip[clip_id] = {
            "show_id": clip_id.split("_")[0] if not clip_id.startswith("s") else "bbt",
            "annotations": json.load(f)
        }

#############################################
# Load predictions
#############################################

pred_paths = PRED_ROOT.glob("*/*.json")
preds_by_clip = {}
for fp in pred_paths:
    model, clip_id, show_id = util.derive_clip_and_show_pred(str(fp))
    if not clip_id:
        continue
    preds_by_clip.setdefault(clip_id, {"show_id": show_id, "predictions": []})
    preds_by_clip[clip_id]["predictions"] = util.process_pred_file(Path(fp))

#############################################
# Further processing
#############################################

final_by_clip = {}
for clip_id, gold in annotations_by_clip.items():
    final_by_clip[clip_id] = {
        "show_id": gold["show_id"],
        "gold": gold["annotations"],
        "pred": preds_by_clip.get(clip_id, {}).get("predictions", [])
    }

clip_metrics = {}
for clip_id, clip_data in final_by_clip.items():
    m = util.compute_pred_vs_gold_agreement(clip_data)
    if m is not None:
        clip_metrics[clip_id] = {"show_id": clip_data["show_id"], "metrics": m}

overall = {f: util.aggregate_clip_metrics(clip_metrics, f)
           for f in ["speaker", "addressee", "side_participant", "reply_to"]}

shows = defaultdict(list)
for clip_id, cd in clip_metrics.items():
    shows[cd["show_id"],].append(cd["metrics"])

by_show = {sid: {f: util.aggregate_by_show(m, f)
                 for f in ["speaker", "addressee", "side_participant", "reply_to"]}
           for sid, m in shows.items()}

bootstrap_by_show = {sid: {f: util.bootstrap_show_field(m, f, num_bootstrap=10000)
                           for f in ["speaker", "addressee", "side_participant", "reply_to"]}
                     for sid, m in shows.items()}

overall_boot = util.bootstrap_overall(by_show, bootstrap_by_show)


util.print_aggregated_metrics(overall, by_show, bootstrap_by_show)
util.print_bootstrap_overall(overall_boot)