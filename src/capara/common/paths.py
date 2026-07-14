"""Filesystem layout.

Every path is derived from a single root so the project can be relocated without
editing code. Override the root with the ``CAPARA_ROOT`` environment variable;
it defaults to the repository checkout.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Root under which datasets and generated artefacts live.
DATA_ROOT = Path(os.environ.get("CAPARA_ROOT", REPO_ROOT))

# Raw benchmark datasets (not tracked in git; see README for download instructions).
DATASETS_DIR = DATA_ROOT / "datasets"
COCO_ROOT = DATASETS_DIR / "coco2017"
FLICKR_ROOT = DATASETS_DIR / "flickr30k"
DOCCI_ROOT = DATASETS_DIR / "docci"
LLAVA_PRETRAIN_ROOT = DATASETS_DIR / "LLaVA-Pretrain"
SHAREGPT4V_SAMPLES_DIR = DATASETS_DIR / "sharegpt4v_samples"
SHAREGPT4V_JSON = DATA_ROOT / "ShareGPT4v_json" / "sharegpt4v_instruct_gpt4-vision_cap100k.json"

# Generated CC3M annotations and the BLIP image-embedding shards built from them.
ANNOTATIONS_DIR = DATA_ROOT / "cc3m_qwen_llama_outputs"
EMBED_DATASET_DIR = DATA_ROOT / "cc3m_blip_embed_dataset"
SHARDS_768_DIR = EMBED_DATASET_DIR / "shards"
SHARDS_256_DIR = EMBED_DATASET_DIR / "shards_256"

# Training runs and evaluation artefacts.
TRAIN_RUNS_DIR = DATA_ROOT / "blip_text_train"
RESULTS_DIR = REPO_ROOT / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ABLATIONS_DIR = RESULTS_DIR / "ablations"

BLIP_MODEL = "Salesforce/blip-itm-base-coco"
CLIP_MODEL = "openai/clip-vit-base-patch32"
