import sys
import json
import math
import base64
import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def timestamp_to_frame_range(start_time, end_time, fps=3):
    start_frame_num = int(math.ceil(start_time * fps)) + 1
    end_frame_num = int(math.floor(end_time * fps)) + 1
    return str(start_frame_num).zfill(5), str(end_frame_num).zfill(5)

def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="data/")
    ap.add_argument("--output_root", type=Path, required=True, help="output/")
    ap.add_argument("--model", default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    data_root = args.data_root
    output_root = args.output_root
    MODEL_ID = args.model
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # This runs your model via vLLM as an OpenAI compatible server
    # cf. https://docs.vllm.ai/en/stable/getting_started/quickstart.html 
    
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    system_instruction = """You are a conversational analysis assistant tasked with analyzing video clips **through a frame-based context. Instead of full-motion video with subtitles, you are provided with a sequence of sampled frames along with accompanying audio feature descriptors and face crop information.** Each task includes two stages:

**1. Context Analysis:**

*   Review the entire clip context, including the sampled frames, associated audio feature buckets (e.g., speech rate, pitch range,, etc.), and face crop cues, to understand conversational threads and participant dynamics.

**2. Targeted Analysis (Specific Line):**

You will be given a specific dialogue line from the clip context to:

*   determine what previous line it is replying to
*   determine the speaker, addressees, and side-participants

Here's how to determine the reply-to relationship between utterances to resolve conversational threads:

*   The reply-to structure gives us information about floor-claiming and topical change within the clip.
*   The character is saying this line because they want to respond to that previous line. What previous line is this current line replying to?
*   If the speaker of the last line is the same, you can treat it as continuation and put the index of the last line as the reply-to.
*   When there is a noticeable change in topic and a shift in participant focus – and no previous line clearly triggers this current line – write the current line index, indicating the current line replies to itself.

Here's how to determine each role:

*   **Speaker:**  The character who is speaking the line.  Infer this from visual cues in the frames (e.g., lip movements, body language, face crops) and the dialogue context. If a character finishes one line and immediately starts another (with only a very short pause), assume it's the same speaker, UNLESS there is clear evidence of a scene or speaker change.
*   **Addressee(s):** The character(s) the speaker is *directly* addressing. Use these cues:
    *   **Eye Contact and Gaze Direction:** Identify whom the speaker is looking at in the frames.
    *   **Body Orientation:** Is the speaker's body oriented toward one person or a group?
    *   **Dialogue Context:** Does the line contain direct address words (like a name or specific pronoun), or is it clearly directed at an individual or group?
    *   **Immediate Reactions:** Characters who react visibly (e.g., changes in expression or body movement) likely are being addressed.
    *   If the speaker appears to talk to everyone present, list all characters who are visibly engaged.
    *   If the speaker is addressing a crowd of unidentifiable individuals, write "crowd".
    *   If speaking to themselves or if no one is directly addressed, write "none".
*   **Side-Participant(s):** Any character(s) visible in the frames during the line's timeframe who are *not* the speaker or addressees. They are present and aware of the conversation and may join later.
    *   If it is not possible to confidently determine if someone is a side-participant, write "unknown".
    *   If there are no side-participants, write "none".

**Input:**

You will receive the CONTEXT information and a specific dialogue line to analyze:

*   **Context information:** 
    *   **A sequence of sampled frames with corresponding audio feature descriptors and face crop information. This context includes, for each segment, keys such as `start_time`, `end_time`, `sentence` (text), `start_frame`, `end_frame`, and audio feature buckets (e.g., `audio_speech_rate_bucket`, `audio_pitch_range_bucket`, `audio_pitch_mean_bucket`, `audio_pause_before_bucket`, `audio_spectral_centroid_bucket`).**

*   **Dialogue line to analyze:**
    *   `"line index"`: (int) The index of the focus dialogue line.
    *   `"start time"`: (float) The start time (in seconds) of the dialogue line.
    *   `"end time"`: (float) The end time (in seconds) of the dialogue line.
    *   `"sentence"`: (string) The text of the dialogue line.

Based on the dialogue line, answer the questions given to you.

**Output:**

Provide your answers in JSON format:
*   `"reply_to"`: (int) The line index that this current line replies to, which could be the same as the current line index or that of a previous line.
*   `"speaker"`: (string) The name of the speaker. If you cannot determine the speaker, use "unknown".
*   `"addressees"`: (list of strings) A list of the names of the addressee(s). This may be an empty list (`[]`), or a list such as `["none"]` if the speaker is addressing no one in particular, or `["crowd"]` if addressing an undefined crowd.
*   `"side_participants"`: (list of strings) A list of the names of the side-participant(s), which can be an empty list (`[]`), `["none"]`, or `["unknown"]`.

This corresponds to the data model format:

```python
class ConversationalRoles(BaseModel):
    reply_to: int  # Index of the line being replied to
    speaker: str   # Speaker of the line, or "unknown"
    addressees: list[str]  # List of names, ["crowd"], ["none"], or ["unknown"]
    side_participants: list[str]  # List of names, ["crowd"], ["none"], or ["unknown"]
"""

    file_dict = {}
    for show_dir in data_root.iterdir():
        if not show_dir.is_dir():
            continue
        show_id = show_dir.name
        clips = {}
        for faces_json in show_dir.glob("*.frame_faces.json"):
            clip_id = faces_json.stem.replace(".frame_faces", "")
            paths = {
                'frame_paths': [p for p in sorted(show_dir.glob(f"{clip_id}.frames/*.jpg"))],
                'multiple_choice_path': show_dir / f"{clip_id}.multiple_choice.json",
                'multiple_choice_image_paths': list((show_dir / f"{clip_id}.multiple_choice").glob("*.jpg"))
            }
        if clips:
            file_dict[show_id] = clips
            
    for show_id, clips in tqdm(file_dict.items(), desc="shows"):
        for clip_id, paths in tqdm(clips.items(), desc=show_id, leave=False):
            out_dir = args.output_root / show_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{clip_id}.{args.model.split('/')[-1]}.json"

            if out_file.exists():
                continue

            script = load_json(paths["script"])
            mc_data = load_json(paths["mc_json"])
            mc_imgs = list(paths["mc_img_dir"].glob("*.jpg"))

            # Build shared context (frames, audio buckets, subtitle text)
            context: list[dict[str, Any]] = []
            context.append({"type": "text", "text": "Context:"})
            for idx, utt in enumerate(script):
                if utt.get("start_time") is None:
                    continue

                # representative frame (end frame)
                _, end_f = timestamp_to_frame_range(
                    utt["start_time"], utt["end_time"], args.fps
                )
                frame_path = paths.get("frames_dir") / f"{end_f}.jpg" if paths.get("frames_dir") else None
                if frame_path and frame_path.exists():
                    context.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_image(frame_path)}})

                audio_txt = (
                    f"Utt {idx} audio:\n"
                    f"- speech_rate: {utt.get('audio_speech_rate_bucket')}\n"
                    f"- pitch_range: {utt.get('audio_pitch_range_bucket')}\n"
                    f"- pitch_mean: {utt.get('audio_pitch_mean_bucket')}\n"
                    f"- pause_before: {utt.get('audio_pause_before_bucket')}\n"
                    f"- spectral_centroid: {utt.get('audio_spectral_centroid_bucket')}\n"
                )
                context.append({"type": "text", "text": audio_txt})
                context.append({"type": "text", "text": f'Utt {idx} text: "{utt.get("sentence")}"'})

            # candidates
            for actor in mc_data:
                safe = actor.replace(" ", "_")
                img_path = next((p for p in mc_imgs if p.name.endswith(f"_{safe}.jpg")), None)
                if img_path:
                    context.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_image(img_path)}})
                context.append({"type": "text", "text": f"Candidate: {actor}"})

            results = {"clip_roles": []}
            for idx, utt in enumerate(script):
                q = f'Analyze line {idx}: "{utt.get("sentence")}".'
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": context + [{"type": "text", "text": q}]},
                ]
                try:
                    resp = client.chat.completions.create(model=args.model, messages=messages)
                    txt = resp.choices[0].message.content
                    try:
                        pred = json.loads(txt)
                    except json.JSONDecodeError:
                        pred = {"raw": txt}
                except Exception as e:
                    pred = {"error": str(e)}
                pred["line_index"] = idx
                results["clip_roles"].append(pred)

            out_file.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()