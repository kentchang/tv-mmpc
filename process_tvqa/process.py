########################################################################
#  process.py - Preprocessing script for TV-MMPC
#----------------------------------------------------------------------
#  This script performs various preprocessing tasks on video clips,
#  including face detection, recognition, subtitle enhancement, and
#  generation of data structures for downstream tasks.
#
#  -- WRANING --
#
#  This is not quite cleaned up atm -- 
#
#  Usage:
#      python process.py --config cfg.yaml
########################################################################

import io
import os
import gc
import re
import sys
import cv2
import json
import math
import yaml
import base64
import torch
import shutil
import pickle
import pathlib
import imageio
import librosa
import requests
import argparse
import whisperx
import subprocess
import insightface
import numpy as np
import pandas as pd
import onnxruntime as ort

from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from copy import deepcopy 
from numpy import linalg as LA
from pydub import AudioSegment
from collections import defaultdict
from rapidfuzz import process, fuzz
from PIL import Image, ImageSequence
from insightface.app.common import Face
from insightface.app import FaceAnalysis
from collections import defaultdict, Counter
from moviepy import VideoFileClip , AudioFileClip
from sklearn.feature_extraction.text import TfidfVectorizer

# download the files from https://github.com/dbamman/movie-representation/
from imdb_vecs import IMDBVecs
from viz_recog import read_names


#---------------------------------------------#
# Util, metadata
#---------------------------------------------#
def load_cfg(cfg_path):
    with open(cfg_path) as f:
        raw = os.path.expandvars(f.read())   # enables ${root} interpolation
    return yaml.safe_load(raw)

def extract_season_episode(clip_id):
    match = re.search(r's(\d{2})e(\d{2})', clip_id, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_json(path):
    with open(path) as o:
        return json.load(o)    

def remove_parentheses(text):
    return re.sub(r'\s*\(.*?\)\s*', '', text)
    
def get_tmdb_json(query):
    tmdb_response=requests.get(query)

    if tmdb_response.status_code == 200:
        return tmdb_response.json()
    else:
        return ""

def get_actor_name(imdb_id):
    global TMDB_API_KEY
    
    url = f"https://api.themoviedb.org/3/find/{imdb_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'external_source': 'imdb_id'
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Checking if any actors are found
        if data.get("person_results"):
            actor = data['person_results'][0]
            return actor['name']
        else:
            return ""
    else:
        return ""   

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):  # Convert NumPy floats to Python float
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):  # Convert NumPy integers to Python int
        return int(obj)
    elif isinstance(obj, dict):  # Recursively convert dictionaries
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # Recursively convert lists
        return [convert_to_json_serializable(v) for v in obj]
    return obj  # Return as is if not a special type

#---------------------------------------------#
# Frames
#---------------------------------------------#
def timestamp_to_seconds(timestamp):
    """Convert SRT timestamp (HH:MM:SS,ms) to seconds."""
    h, m, s_ms = timestamp.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def process_frames(show_ids, clip_id2files, face_detection_model_path, app):
    def yunet_streamline(app, yunet, image):
        h, w = image.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(image)

        if faces is None:
            return None
            
        output = []
        for face in faces:
            x1, y1, w, h =  map(int, face[:4])
            x2, y2 = x1 + w, y1 + h
            
            landmarks=face[4:-1]
            conf = face[-1]  # Confidence score
            
            img_region =[x1, y1, w, h]
            face = Face(bbox=img_region, kps=landmarks.reshape((5,2)), det_score=conf)
            for taskname, model in app.models.items():
                if taskname=='recognition':
                    model.get(image, face)   
                    output.append(face)
        return output

    def find_closest_actor(query_name, actor_dict):
        best_match, score, _ = process.extractOne(query_name, actor_dict.keys())  # Find closest match
        if score > 80: 
            return actor_dict[best_match] 
        return None

    global ROOT, castDir, nameFile, imdb_reps

    SHOW_JSON_ROOT = ROOT / 'show_json'

    EPISODE_ID_ROOT = SHOW_JSON_ROOT / show_id / 'episode_external_ids'
    EPISODE_ID_ROOT.mkdir(exist_ok=True, parents=True)
    
    EPISODE_JSON_ROOT = SHOW_JSON_ROOT / show_id / 'episodes'
    EPISODE_JSON_ROOT.mkdir(exist_ok=True, parents=True)
    
    CAST_JSON_ROOT = SHOW_JSON_ROOT / show_id / 'cast'
    CAST_JSON_ROOT.mkdir(exist_ok=True, parents=True)
    
    EPISODE_ACTOR_MAPPING_ROOT = SHOW_JSON_ROOT / show_id / 'episode_actor_mapping'
    EPISODE_ACTOR_MAPPING_ROOT.mkdir(exist_ok=True, parents=True)

    RECOGNIZED_FACES_ROOT = SHOW_JSON_ROOT / show_id / 'recognized_faces'
    RECOGNIZED_FACES_ROOT.mkdir(exist_ok=True, parents=True)

    CAPTIONED_FRAMES_ROOT = SHOW_JSON_ROOT / show_id / 'captioned_frames'
    CAPTIONED_FRAMES_ROOT.mkdir(exist_ok=True, parents=True)

    show_id2imdb_id = {
        'grey': 'tt0413573', 
        'house': 'tt0412142', 
        'friends': 'tt0108778', 
        'bbt': 'tt0898266', 
        'met': 'tt0460649', 
        'castle': 'tt1219024'
    }

    show_id2tmdb_id = {
        'grey': 1416,
        'house': 1408,
        'friends': 1668,
        'bbt': 1418,
        'met': 1100,
        'castle': 1419
    }

    names = read_names(nameFile) # space around `=` to breathe 
    imdbvecs = IMDBVecs(imdb_reps, castDir)
    
    yunet = cv2.FaceDetectorYN.create(
        model=face_detection_model_path,
        config='',
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )

    for show_id in tqdm(show_ids): 
        for clip_id, path_dict in (pbar := tqdm(clip_id2files[show_id].items(), total=len(clip_id2files[show_id]))):   
            season, episode = extract_season_episode(clip_id)
            tmdb_id = show_id2tmdb_id[show_id]
            
            pbar.set_description(f"Processing {show_id}: s{season}, e{episode} -- {clip_id}")
            
            episode_imdb_id = None        
            
            ##################################################
            
            if (EPISODE_ID_ROOT / f"s{season}_e{episode}.json").exists():
                with open(EPISODE_ID_ROOT / f"s{season}_e{episode}.json") as o:
                    tv_episode_external_id_json = json.load(o)
            else:
                tv_episode_external_id_query = f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{season}/episode/{episode}/external_ids?api_key={TMDB_API_KEY}"
                tv_episode_external_id_json = get_tmdb_json(tv_episode_external_id_query)
                
                if tv_episode_external_id_json:
                    with open(EPISODE_ID_ROOT / f"s{season}_e{episode}.json", 'w') as o:
                        json.dump(tv_episode_external_id_json, o)

            if tv_episode_external_id_json:
                episode_imdb_id = tv_episode_external_id_json['imdb_id']

            if not episode_imdb_id:
                continue      
            
            if (EPISODE_JSON_ROOT / f"s{season}_e{episode}.json").exists():
                with open(EPISODE_JSON_ROOT / f"s{season}_e{episode}.json") as o:
                    tv_episode_json = json.load(o)
            else:
                tv_episode_query=f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{season}/episode/{episode}?api_key={TMDB_API_KEY}&language=en-US%22"
                tv_episode_json=get_tmdb_json(tv_episode_query)
                
                if tv_episode_json:
                    with open(EPISODE_JSON_ROOT / f"s{season}_e{episode}.json", 'w') as o:
                        json.dump(tv_episode_json, o)
            
            if (CAST_JSON_ROOT / f"s{season}_e{episode}.json").exists():
                with open(CAST_JSON_ROOT / f"s{season}_e{episode}.json") as o:
                    tv_cast_json = json.load(o)
            else:
                tv_cast_query=f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/{season}/episode/{episode}/credits?api_key={TMDB_API_KEY}&language=en-US%22"
                tv_cast_json=get_tmdb_json(tv_cast_query)
                
                if tv_cast_json:
                    with open(CAST_JSON_ROOT / f"s{season}_e{episode}.json", 'w') as o:
                        json.dump(tv_cast_json, o)

            ##################################################
            
            if (EPISODE_ACTOR_MAPPING_ROOT / f"s{season}_e{episode}.json").exists():
                with open(EPISODE_ACTOR_MAPPING_ROOT / f"s{season}_e{episode}.json") as o:
                    episode_actor_mapping = json.load(o)
            else:
                episode_actor_mapping = {}
                
                if tv_cast_json and 'cast' in tv_cast_json:
                    tv_cast_results=tv_cast_json['cast']
                
                    for cast_data_item in tv_cast_results:
                        episode_actor_mapping.setdefault(cast_data_item['character'], cast_data_item['name'])
                
                if tv_episode_json and 'guest_stars' in tv_episode_json:
                    tv_episode_results = tv_episode_json['guest_stars']
                    for guest_star in tv_episode_results:
                        episode_actor_mapping.setdefault(guest_star['character'], guest_star['name'])    
                        
                if episode_actor_mapping:
                    with open(EPISODE_ACTOR_MAPPING_ROOT / f"s{season}_e{episode}.json", 'w') as o:
                        json.dump(episode_actor_mapping, o)
            
            if episode_actor_mapping:
                reversed_episode_actor_mapping = {v: k for k, v in episode_actor_mapping.items()}

            ##################################################

            CAPTIONED_CLIP_FRAME_ROOT = CAPTIONED_FRAMES_ROOT / clip_id
            CAPTIONED_CLIP_FRAME_ROOT.mkdir(exist_ok=True, parents=True)

            episode_folder = RECOGNIZED_FACES_ROOT / f"s{season}_e{episode}" / clip_id
            episode_folder.mkdir(exist_ok=True, parents=True)
            face_images_folder = episode_folder / "face_images"
            face_images_folder.mkdir(exist_ok=True, parents=True)           

            if (episode_folder / "faces.json").exists() and (episode_folder / "face_embeddings.json").exists():
                continue                    

            frame_paths = sorted(path_dict['frame_paths'])

            faces_data = {}
            embeddings_data = {}
            
            for frame_path in frame_paths:
                frame_stem = frame_path.stem             
                captioned_frame_path = CAPTIONED_CLIP_FRAME_ROOT / frame_path.parts[-1]

                try:
                    image = cv2.imread(str(frame_path))
                    recognized_faces = yunet_streamline(yunet, image)

                except:
                    print(f"{show_id}, {season}, {episode}")        
                    continue
                    
                # Always store even if no faces are detected.
                # Important! So frames line up nicely
                faces_data[frame_stem] = []
                embeddings_data[frame_stem] = []
                
                # If no faces, continue to the next frame.
                if not recognized_faces:
                    continue

                for face in recognized_faces:
                    x1, y1, w, h = face['bbox']      
                    x2, y2 = x1 + w, y1 + h
                
                    caption = ''
                    color = (0, 0, 255)
                    
                    imdb_id, top_actors, top_actor_name, top_actor_character_name = None, None, None, None
                    top_actors=imdbvecs.get_actors_ep_first(face['embedding'], episode_imdb_id, show_id2imdb_id[show_id], num_to_get=10, aggregator=np.mean)
                    
                    if not top_actors:
                        top_actors=imdbvecs.get_actors(face['embedding'], show_id2imdb_id[show_id], num_to_get=10, aggregator=np.mean)
        
                    if top_actors:
                        imdb_id = top_actors[0].split(':')[0]

                        if imdb_id in names:
                            top_actor_name = names[imdb_id]
                        else:
                            top_actor_name = get_actor_name(imdb_id)

                        if top_actor_name:
                            top_actor_character_name = find_closest_actor(top_actor_name, reversed_episode_actor_mapping)
                            
                        if not top_actor_character_name or not top_actor_name:
                            pass
                        elif top_actor_character_name == "None" or top_actor_name == "None":
                            pass                        
                        else:
                            caption = (top_actor_character_name, top_actor_name) 
                            color = (255, 0, 255)

                    face['imdb_id'] = imdb_id
                    face['top_actor_name'] = top_actor_name
                    face['top_actor_character_name'] = top_actor_character_name

                    # Ensure the embedding is JSON serializable (convert to list if necessary).
                    if hasattr(face['embedding'], 'tolist'):
                        face_embedding = face['embedding'].tolist()
                    else:
                        face_embedding = face['embedding']                
                    
                    # Prepare the complete face info (for faces.json)
                    face_info = {
                        "bbox": face['bbox'],
                        "kps": face.get('kps', []),
                        "det_score": face.get('det_score', None),
                        # "embedding": face_embedding,
                        "imdb_id": imdb_id,
                        "top_actor_name": top_actor_name,
                        "top_actor_character_name": top_actor_character_name
                    }
                    faces_data[frame_stem].append(face_info)

                    # Prepare the embedding-specific info (for face_embeddings.json)
                    embedding_info = {
                        "embedding": face_embedding,
                        "imdb_id": imdb_id,
                        "det_score": face.get('det_score', None),
                        "top_actor_name": top_actor_name,
                        "top_actor_character_name": top_actor_character_name
                    }
                    embeddings_data[frame_stem].append(embedding_info)                

                    # Crop the face from the image using the bbox info.
                    face_crop = image[y1:y2, x1:x2]
                    out_imdb_id = imdb_id if imdb_id is not None else "none"
                    face_crop_filename = f"{frame_stem}_{out_imdb_id}.jpg"
                    face_crop_path = face_images_folder / face_crop_filename
                    if face_crop_path.exists():
                        continue
                    else:
                        try:
                            cv2.imwrite(str(face_crop_path), face_crop)
                        except KeyboardInterrupt:
                            sys.exit(1)    
                        except:
                            continue                
                    
                    # Update frames with caption
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

                    if caption:
                        caption_y_second_line = y1 - 10
                        caption_y_first_line = caption_y_second_line - 15
                        
                        if caption_y_second_line < 30:  # If too close to the top, move below the rectangle
                            caption_y_first_line = y1 + h + 15      
                            caption_y_second_line = caption_y_first_line + 12
                            
                        cv2.putText(image, caption[0], (x1, caption_y_first_line), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                        cv2.putText(image, f"({caption[1]})", (x1, caption_y_second_line), cv2.FONT_HERSHEY_SIMPLEX, .40, color, 1)                    
                    else:
                        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)

                cv2.imwrite(str(captioned_frame_path), image)
                
            with open(episode_folder / "faces.json", 'w') as f:
                json.dump(convert_to_json_serializable(faces_data), f, indent=4)
            
            with open(episode_folder / "face_embeddings.json", 'w') as f:
                json.dump(convert_to_json_serializable(embeddings_data), f, indent=4)


def copy_best_faces_for_clip(faces_data, season, episode, show_id, clip_id):
    global RECOGNIZED_FACES_ROOT, INFERENCE_DATA_ROOT

    src_root = (RECOGNIZED_FACES_ROOT 
                / f"s{season}_e{episode}" 
                / clip_id 
                / "face_images")

    # where you want to drop your multiple_choice images:
    dst_root = INFERENCE_DATA_ROOT / show_id / f"{clip_id}.multiple_choice"
    dst_root.mkdir(exist_ok=True, parents=True)

    overall_candidates = {}

    for frame_key, faces in faces_data.items():
        for face in faces:
            actor_id = face.get("imdb_id")
            # skip empty / 'none'
            if not actor_id or actor_id.lower() == "none":
                continue
            # initialise per-actor dict
            if actor_id not in overall_candidates:
                overall_candidates[actor_id] = {
                    "top_actor_name": face.get("top_actor_name"),
                    "top_character_name": face.get("top_actor_character_name"),
                    "count": 0,
                    "best_det_score": face.get("det_score", 0),
                    "best_face": frame_key
                }
            # running stats
            overall_candidates[actor_id]["count"] += 1
            cur_score = face.get("det_score", 0)
            if cur_score > overall_candidates[actor_id]["best_det_score"]:
                overall_candidates[actor_id]["best_det_score"] = cur_score
                overall_candidates[actor_id]["best_face"] = frame_key

    # copy one jpeg per actor into .multiple_choice/
    for actor_id, candidate in overall_candidates.items():
        if actor_id.lower() == "none":
            continue
        src = src_root / f"{candidate['best_face']}_{actor_id}.jpg"
        if not src.exists():
            # rare, but fail-safe: warn and skip
            print(f"[warning] missing source face {src}")
            continue
        char_name = candidate.get("top_character_name", "unknown")
        if char_name == "unknown":
            # we only keep faces with usable character labels
            continue
        safe_char = char_name.replace(" ", "_")
        dst = dst_root / f"{actor_id}_{safe_char}.jpg"
        shutil.copy2(src, dst)  # copy2 = keep mtime

#---------------------------------------------#
# Audio and subtitles
#---------------------------------------------#

def enhance_subtitles(show_ids, clip_id2files, device, device_index):
    def get_whisperx_segments(audio_file, model, diarize_model, device=device, batch_size=batch_size):
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        gc.collect() 
        torch.cuda.empty_cache()
        return result['segments']

    def build_set2_tfidf(set2, preferred_speaker_names):
        """
        Projects TVQA speaker names to Whisper transcription and standardize them,
        using preferrewd TMDB names (first + last) -- they are preferred bc
        they represent (more) canonical forms of the names and are usually associated
        with external IDs like IMDb -- presents opportunities for further 
        downstream analysis.

        set2: 
            TVQA subtitles

        preferred_speaker_names: 
            TMDb cast list names
        """

        set2_texts = [seg.get('sentence_text', '') for seg in set2]
        vectorizer = TfidfVectorizer(norm='l2')
        set2_matrix = vectorizer.fit_transform(set2_texts)  # shape (M, vocab_size)

        real_speakers_in_set2 = {
            seg['speaker']
            for seg in set2
            if seg.get('speaker') and seg['speaker'].lower() != 'unknown'
        }

        # Builds speaker_map
        speaker_map = {}
        preferred_tokenized = {name: set(name.lower().split()) for name in preferred_speaker_names}
        
        for sp in real_speakers_in_set2:
            sp_tokens = set(sp.lower().split())
        
            # Step 1: Find all preferred names with at least one overlapping token
            candidates = [name for name, tokens in preferred_tokenized.items() if sp_tokens & tokens]
        
            # Step 2: If only one candidate is found, assign it directly
            if len(candidates) == 1:
                speaker_map[sp] = candidates[0]
            elif candidates:  # Step 3: Use fuzzy matching if multiple candidates exist
                best = process.extractOne(sp, candidates, scorer=fuzz.ratio)
                if best:
                    speaker_map[sp] = best[0]
            else:
                speaker_map[sp] = sp  # Fallback: Keep original if no match
                
        return vectorizer, set2_matrix, speaker_map

    def align_set1_with_tfidf(set1, set2, vectorizer, set2_matrix, speaker_map):
        """
        Aligns TVQA subtitles with Whisper transcriptions, based on algorithm 1 in:

            Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. "MovieNet: A Holistic Dataset for Movie understanding" [ECCV 2020]

        Args:
            set1: the texts to be aligned to, like whisper segments
            set2: the texts with more info, like original TVQA subtitles
        Returns: a new list of dict for set1 with improved 'speaker'.
        """
        index_to_speaker = [seg.get('speaker', 'unknown') for seg in set2]

        set1_texts = [seg.get('text', '') for seg in set1]
        set1_matrix = vectorizer.transform(set1_texts)  # shape (N, vocab_size)
        sim_matrix = set1_matrix.dot(set2_matrix.T)  # shape (N, M)
        sim_array = sim_matrix.toarray()
        best_indices = np.argmax(sim_array, axis=1)  # shape (N,)

        # First pass:
        updated_set1 = []
        for i, seg1 in enumerate(set1):
            j = best_indices[i]
            candidate_spk = index_to_speaker[j] if 0 <= j < len(index_to_speaker) else 'unknown'
            seg_copy = dict(seg1)
            seg_copy['speaker'] = candidate_spk
            updated_set1.append(seg_copy)

        # Second pass: 

        # The general idea is that we want to increase recall and get as many names
        # as possible, because they will all be manually fixed later.
        # So, we want to reduce the number of `unknown` speakers.
        # For each segment that ended withunknown, check: 
        #   if we previously found a real speaker for that same generic label
        #   in an earlier segment:
        #       if so, reuse it :)

        last_known_for_label = {}
        for seg in updated_set1:
            generic_label = seg.get('speaker', 'unknown')  # after first pass, might be 'SPEAKER_00' -- generic bc it's whisper
            if generic_label.lower().startswith('speaker_'):  
                # It's still the original label, or we can keep track of it differently.
                pass
            original_label = seg1.get('speaker', 'unknown')  

        second_pass_result = []
        last_known_for_label = {}
        for seg_orig, seg_upd in zip(set1, updated_set1):
            gen_label = seg_orig.get('speaker', 'unknown')
            assigned_spk = seg_upd.get('speaker', 'unknown') or 'unknown'
            if assigned_spk.lower() != 'unknown':
                # Real speaker, record it for future reuse
                last_known_for_label[gen_label] = assigned_spk
                second_pass_result.append(seg_upd)
            else:
                if gen_label in last_known_for_label:
                    seg_upd['speaker'] = last_known_for_label[gen_label]
                second_pass_result.append(seg_upd)

        # Map final (non-unknown) speakers to preferred names
        for seg in second_pass_result:
            spk = seg['speaker']
            if spk.lower() != 'unknown' and spk in speaker_map:
                seg['speaker'] = speaker_map[spk]

        return updated_set1
        
    global HF_TOKEN, EPISODE_ACTOR_MAPPING_ROOT

    batch_size = 512 
    compute_type = "float16" 

    model = whisperx.load_model(
        "large-v2",
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        language='en'
    )

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)    

    for show_id in tqdm(show_ids): 
        for clip_id, path_dict in tqdm(clip_id2files[show_id].items(), total=len(clip_id2files[show_id])):
            season, episode = extract_season_episode(clip_id)
            srt_path = path_dict['srt_path']
            audio_path = path_dict['audio_path']        
            srt_json_path = str(path_dict['srt_path']).replace('/tvqa_subtitles/', '/tvqa_subtitles_json/')
            enhanced_path = pathlib.Path(str(path_dict['srt_path']).replace('/tvqa_subtitles/', '/enhanced_subtitles/').replace('.srt', '.json'))
            whisperx_segments_path = pathlib.Path(str(path_dict['srt_path']).replace('/tvqa_subtitles/', '/whisperx_segments/').replace('.srt', '.json'))

            if enhanced_path.exists():
                continue
            
            episode_actor_mapping_path = EPISODE_ACTOR_MAPPING_ROOT / f"s{season}_e{episode}.json"

            if not episode_actor_mapping_path.exists():
                print(f"File does not exist: {episode_actor_mapping_path}")
                continue
            
            episode_actor_mapping = load_json(episode_actor_mapping_path)

            if whisperx_segments_path.exists():
                with open(whisperx_segments_path) as whis:
                    set1 = json.load(whis)
            else:
                set1 = get_whisperx_segments(path_dict[show_id][clip_id]['audio_path'], model, diarize_model, device, batch_size)
                with open(whisperx_segments_path, 'w') as o:
                    json.dump(set1, o)
                    
            set2 = load_json(srt_json_path)
            preferred_speaker_names = list(episode_actor_mapping.keys())

            new_set2 = []
            for seg in set2:
                text = seg['sentence_text']
                if ' - ' in text:
                    parts = text.lstrip("- ").split(" - ")
                    for part in parts:
                        part = part.strip() 
                        if part:  
                            new_seg = seg.copy()
                            new_seg['sentence_text'] = part
                            new_set2.append(new_seg)
                else:
                    new_set2.append(seg)
            
            set2 = new_set2

            vectorizer, set2_matrix, speaker_map = build_set2_tfidf(set2, preferred_speaker_names)
            improved_set1 = align_set1_with_tfidf(set1, set2, vectorizer, set2_matrix, speaker_map)

            with open(enhanced_path, 'w') as o:
                json.dump(improved_set1, o)

def count_words(text):
    WORD_RE = re.compile(r"\b\w+\b")

    return len(WORD_RE.findall(text))

def extract_acoustic_features(y, sr):
    energy = float(np.mean(librosa.feature.rms(y=y)[0]))

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = pitches[mags > 0]
    pitch_min = float(valid.min()) if valid.size else 0.0
    pitch_max = float(valid.max()) if valid.size else 0.0
    pitch_mean = float(valid.mean()) if valid.size else 0.0

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    return {
        "energy_rms": energy,
        "pitch_min_hz": pitch_min,
        "pitch_max_hz": pitch_max,
        "pitch_mean_hz": pitch_mean,
        "spectral_centroid_hz": centroid,
    }

def bucket_acoustic_features(raw, ref):
    pitch_range = raw["pitch_max_hz"] - raw["pitch_min_hz"]
    def _cmp(val, ref_val):
        if val > ref_val:
            return "higher"
        if val < ref_val:
            return "lower"
        return "average"

    buckets = {
        "energy": _cmp(raw["energy_rms"], ref["energy_rms"]),
        "speech_rate": _cmp(raw["speech_rate_wps"], ref["speech_rate_wps"]),
        "pitch_range": "narrow" if pitch_range < 80 else "medium" if pitch_range < 180 else "wide",
        "pitch_mean": _cmp(raw["pitch_mean_hz"], ref["pitch_mean_hz"]),
        "pause_before": "none" if raw["pause_before_sec"] < 0.2 else "short" if raw["pause_before_sec"] < 1 else "long",
        "spectral_centroid": "brighter" if raw["spectral_centroid_hz"] > ref["spectral_centroid_hz"] else "darker",
    }
    return buckets

def timestamp_to_frame_range(start: float, end: float, fps: int) -> Tuple[str, str]:
    # !! start and end are floats, and fps is 3
    # !! tvqa frames are named like this: frame 1 -> 00001
    s = int(math.ceil(start * fps)) + 1
    e = int(math.floor(end * fps)) + 1
    return str(s).zfill(5), str(e).zfill(5)


#---------------------------------------------#
# Reconstruct clips for annotation
#---------------------------------------------#

def reconstruct_clips(show_ids,clip_id2files, fps ):
    global TEMP_VIDEO_ROOT, VIDEO_ROOT

    def create_video_from_frames(frame_paths, output_video, fps):
        frame_paths = sorted(frame_paths)  # Ensure correct order
        
        if not frame_paths:
            raise ValueError("No frames found!")

        first_frame = cv2.imread(frame_paths[0])
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format

        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            video.write(img)

        video.release()

    def add_audio_to_video(video_path, audio_path, output_path):
        command = [
            "ffmpeg", "-y",
            "-i", video_path,                   # Input video
            "-i", audio_path,                   # Input audio
            "-vf", f"fps=30,format=yuv420p",    # Forces FPS and corrects format
            "-c:v", "libx264",                  # Re-encode video as H.264
            "-preset", "fast",                  # Faster encoding
            "-pix_fmt", "yuv420p",              # Ensures proper pixel format
            "-c:a", "aac",                      # Encode audio
            "-strict", "experimental",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


    for show_id in tqdm(show_ids):
        for clip_id, path_dict in tqdm(clip_id2files[show_id].items(), total=len(clip_id2files[show_id])):            
            temp_video = TEMP_VIDEO_ROOT / f"{clip_id}_temp.mp4"
            final_video = VIDEO_ROOT / f"{clip_id}.mp4"

            if len(set(path_dict.keys()) & {'srt_path', 'audio_path', 'frame_paths'}) != 3:
                continue
                
            srt_path = path_dict['srt_path']
            audio_path = path_dict['audio_path']
            frame_paths = path_dict['frame_paths']   

            if not frame_paths:
                continue

            captioned_frame_paths = []

            for frame_path in frame_paths:            
                captioned_frame_path = pathlib.Path(str(frame_path).replace(
                    f'/tvqa_video_frames/frames_hq/{show_id}_frames/',
                    f'/show_json/{show_id}/captioned_frames/'
                ))
                if captioned_frame_path.exists():
                    captioned_frame_paths.append(captioned_frame_path)
                else:
                    captioned_frame_paths.append(frame_path)
                
            try:
                create_video_from_frames(captioned_frame_paths, temp_video, fps)
                add_audio_to_video(temp_video, audio_path, final_video)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to cfg.yaml")
    args = parser.parse_args()

    user_cfg = load_cfg(args.config)

    ROOT = Path(user_cfg["root"])
    AUDIO_ROOT = Path(user_cfg["audio_root"])
    SUBTITLE_ROOT = Path(user_cfg["subtitle_root"])
    FRAMES_ROOT = Path(user_cfg["frames_root"])

    FPS = user_cfg["fps"]

    TMDB_API_KEY = user_cfg["tmdb_api_key"]
    HF_TOKEN = user_cfg["hf_token"]

    show_ids = user_cfg["show_ids"]

    face_detection_model_path = Path(user_cfg["face_detection_model"])
    imdb_reps = Path(user_cfg["imdb_reps"])
    castDir = Path(user_cfg["cast_dir"])
    nameFile = Path(user_cfg["name_file"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_index = user_cfg["whisper"]["device_index"]

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=tuple(user_cfg["yunet"]["input_size"]))

    assert AUDIO_ROOT.exists() == SUBTITLE_ROOT.exists() == FRAMES_ROOT.exists() == True

    audio_files = list(AUDIO_ROOT.glob('*/*/*.mp3'))
    subtitle_files = list(SUBTITLE_ROOT.glob('*.srt'))
    frame_files = list(FRAMES_ROOT.glob('*/*/*/*.jpg'))

    clip_id2files = {}

    for file in audio_files:
        if 'bbt' in str(file):
            show_id = 'bbt'
        else:
            show_id = file.parent.parent.stem
            
        clip_id = file.stem    
        clip_id2files.setdefault(show_id, {}).setdefault(clip_id, {
            'audio_path': str(file)
        })  

    for file in subtitle_files:
        clip_id = file.stem
        show_id = clip_id.split('_')[0]

        if show_id not in clip_id2files and show_id[0] == 's':
            show_id = 'bbt'
        
        clip_id2files[show_id][clip_id]['srt_path'] = str(file)

    for frame_file in frame_files:
        show_id = frame_file.parent.parent.stem.split('_')[0]
        clip_id = frame_file.parent.stem
        
        if show_id not in clip_id2files:
            continue
        if clip_id not in clip_id2files[show_id]:
            continue
        
        if 'frame_paths' not in clip_id2files[show_id][clip_id]:
            clip_id2files[show_id][clip_id]['frame_paths'] = []
        clip_id2files[show_id][clip_id]['frame_paths'].append(frame_file)

    process_frames(show_ids, clip_id2files, face_detection_model_path, app)
    enhance_subtitles(show_ids, clip_id2files, device, device_index)
    reconstruct_clips(show_ids, clip_id2files, FPS)