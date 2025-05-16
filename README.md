# TV-MMPC

Code and data to support:

Kent K. Chang, Mackenzie Hanh Cramer, Anna Ho, Ti Ti Nguyen, Yilin Yuan, David Bamman, "Multimodal Conversation Structure Understanding" (2025)

Alpha release -- reach out to `kentkchang@berkeley.edu` if you have any questions.

## TVQA

Our annotated dataset is built on [TVQA](https://aclanthology.org/D18-1167/) -- you need to download their data first; see [here](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) for instructions.

## Data structure

You can download the data from [Harvard Dataverse](https://doi.org/10.7910/DVN/4KUKUL) and put it in the `data/` directory:

```
data/
└── {show_id}/
    ├── {clip_id}.annotation.json
    ├── {clip_id}.episode_cast.json
    └── {clip_id}.frame_faces.json
```

Here are some examples:

### `.annotation.json`

List of annotated utterances with conversational role/reply-to labels:

```json
[
  {
    "line_idx": 1,
    "speaker": "leonard hofstadter",
    "addressee": ["sheldon cooper"],
    "side_participant": [],
    "reply_to": 1
  }
]
```

### `.episode_cast.json`

Maps character names to actor names for the episode, public information from [TMDb](https://www.themoviedb.org/) and [IMDb](imdb.com):

```json
{
  "Leonard Hofstadter": "Johnny Galecki",
  "Sheldon Cooper": "Jim Parsons",
  ...
}
```

### `.frame_faces.json`

Detected faces per frame with bounding boxes, keypoints, detection scores, and actor identity predictions -- face detection follows the pipeline described in [https://github.com/dbamman/movie-representation/tree/main/pipeline/scripts](https://github.com/dbamman/movie-representation/tree/main/pipeline/scripts):

```json
{
  "00001": [
    {
      "bbox": [336, 36, 75, 88],
      "kps": [[347.2, 68.6], ...],
      "det_score": 0.94,
      "imdb_id": "nm0301959",
      "top_actor_name": "Johnny Galecki",
      "top_actor_character_name": "Leonard Hofstadter"
    }
  ]
}
```

## Scripts

### `process_tvqa/`

Preprocessing pipeline for TVQA data:

* `cfg.yaml`: YAML configuration file containing root paths, show list, API keys, etc.
* `process.py`: Extracts frames, detectss faces, processes subtitles, and reconstructs annotated video clips.
 

### `inference/`

You can swap out the model name in `vlm-inference.py` to do this with other models via vLLM for vision–language models; or you can check out the `gemini-demo.ipynb` (which requires Google Cloud Platfrm). 

By default, output JSON files are saved under:

```
inference/output/{model_name}/{clip_id}.json
```

Each prediction contains a list of roles per utterance.

### `eval/`

Evaluates predictions against `.annotation.json` files:

* `util.py`: contains all evaluation and helper functions.
* `eval.py`: main entry point; loads gold and predicted data, computes metrics, and prints aggregate results.