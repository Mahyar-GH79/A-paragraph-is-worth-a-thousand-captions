"""Image-text sample loading for the analysis figures.

Every analysis reduces a benchmark to the same shape: a list of ``{"image_path", "text"}``
pairs, one text per image. Samples whose image is missing on disk are dropped.

The random subsets are drawn with ``random.Random(seed)`` so that repeated runs -- and
the different analyses -- see the same images.
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from capara.common.paths import (
    COCO_ROOT,
    DATASETS_DIR,
    DOCCI_ROOT,
    FLICKR_ROOT,
    SHAREGPT4V_SAMPLES_DIR,
)

Sample = dict[str, str]
PathLike = str | Path

# Default dataset locations; every module exposes these as argparse overrides.
SHAREGPT4V_SAMPLE_JSON = SHAREGPT4V_SAMPLES_DIR / "sample_coco50000_llava30000_seed12345.json"
SHAREGPT4V_IMAGE_ROOT = DATASETS_DIR
DOCCI_JSONL = DOCCI_ROOT / "docci_descriptions.jsonlines"
DOCCI_IMAGE_DIR = DOCCI_ROOT / "images"
COCO_VAL_ANN = COCO_ROOT / "annotations" / "captions_val2017.json"
COCO_VAL_IMAGE_DIR = COCO_ROOT / "images" / "val2017"
COCO_TRAIN_ANN = COCO_ROOT / "annotations" / "captions_train2017.json"
COCO_TRAIN_IMAGE_DIR = COCO_ROOT / "images" / "train2017"
FLICKR_IMAGE_DIR = FLICKR_ROOT / "flickr30k_images" / "flickr30k_images"
FLICKR_CAPTIONS_CSV = FLICKR_ROOT / "flickr30k_images" / "results.csv"


def _sharegpt4v_text(entry: dict) -> str:
    """The long description of a ShareGPT4V entry, from either schema."""
    text = (entry.get("text") or "").strip()
    if text:
        return text
    for turn in entry.get("conversations", []):
        if turn.get("from") in ("gpt", "assistant"):
            return (turn.get("value") or "").strip()
    return ""


def load_sharegpt4v_samples(
    json_path: PathLike,
    image_root: PathLike,
    num_samples: int,
    seed: int = 42,
) -> list[Sample]:
    """Load a random subset of ShareGPT4V (either the raw instruct JSON or a sample cache)."""
    with open(json_path) as handle:
        data = json.load(handle)

    random.Random(seed).shuffle(data)

    samples: list[Sample] = []
    for entry in data:
        image_path = entry.get("image_path") or os.path.join(image_root, entry.get("image", ""))
        if not os.path.isfile(image_path):
            continue
        text = _sharegpt4v_text(entry)
        if not text:
            continue
        samples.append({"image_path": str(image_path), "text": text})
        if len(samples) >= num_samples:
            break
    return samples


def load_docci_records(jsonl_path: PathLike, image_dir: PathLike) -> list[Sample]:
    """Load every DOCCI description, in file order.

    ``image_file`` is kept alongside the path: the qualitative figures cite it.
    """
    records: list[Sample] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            description = (record.get("description") or "").strip()
            image_file = record.get("image_file", "")
            if not description or not image_file:
                continue
            image_path = os.path.join(image_dir, image_file)
            if not os.path.isfile(image_path):
                continue
            records.append(
                {"image_path": image_path, "text": description, "image_file": image_file}
            )
    return records


def load_docci_samples(
    jsonl_path: PathLike,
    image_dir: PathLike,
    num_samples: int,
    seed: int = 42,
) -> list[Sample]:
    """Load a random subset of DOCCI."""
    raw: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    random.Random(seed).shuffle(raw)

    samples: list[Sample] = []
    for record in raw:
        description = (record.get("description") or "").strip()
        image_file = record.get("image_file", "")
        if not description or not image_file:
            continue
        image_path = os.path.join(image_dir, image_file)
        if not os.path.isfile(image_path):
            continue
        samples.append({"image_path": image_path, "text": description})
        if len(samples) >= num_samples:
            break
    return samples


def load_coco_samples(
    ann_path: PathLike,
    image_dir: PathLike,
    num_samples: int,
    seed: int = 42,
) -> list[Sample]:
    """Load a random subset of COCO, keeping the first caption of each image."""
    with open(ann_path) as handle:
        data = json.load(handle)

    id_to_file = {image["id"]: image["file_name"] for image in data["images"]}
    image_to_captions = defaultdict(list)
    for annotation in data["annotations"]:
        image_to_captions[annotation["image_id"]].append(annotation["caption"])

    items: list[Sample] = []
    for image_id, captions in image_to_captions.items():
        file_name = id_to_file.get(image_id)
        if not file_name:
            continue
        image_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(image_path):
            continue
        items.append({"image_path": image_path, "text": captions[0].strip()})

    random.Random(seed).shuffle(items)
    return items[:num_samples]


def _read_flickr_csv(csv_path: PathLike, image_dir: PathLike) -> list[Sample]:
    """Parse Flickr30k's pipe-separated ``results.csv``, keeping one caption per image."""
    first_caption: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|", skipinitialspace=True)
        columns = {name.strip(): name for name in (reader.fieldnames or [])}
        if "image_name" not in columns or "comment" not in columns:
            return []
        for row in reader:
            file_name = (row.get(columns["image_name"]) or "").strip()
            caption = (row.get(columns["comment"]) or "").strip()
            if not file_name or not caption:
                continue
            file_name = os.path.basename(file_name)
            first_caption.setdefault(file_name, caption)

    items: list[Sample] = []
    for file_name, caption in first_caption.items():
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            items.append({"image_path": image_path, "text": caption})
    return items


def _read_flickr_karpathy(json_path: PathLike, image_dir: PathLike) -> list[Sample]:
    """Parse a Karpathy-split ``dataset_flickr30k.json``, test split only."""
    with open(json_path) as handle:
        data = json.load(handle)

    items: list[Sample] = []
    for image in data.get("images", []):
        if image.get("split") != "test":
            continue
        image_path = os.path.join(image_dir, image.get("filename", ""))
        if not os.path.isfile(image_path):
            continue
        captions = [s["raw"] for s in image.get("sentences", []) if s.get("raw")]
        if captions:
            items.append({"image_path": image_path, "text": captions[0].strip()})
    return items


def load_flickr_samples(
    image_dir: PathLike,
    num_samples: int,
    seed: int = 42,
    captions_path: PathLike | None = None,
) -> list[Sample]:
    """Load a random subset of Flickr30k.

    Uses ``captions_path`` when given; otherwise looks for a Karpathy-split JSON and then
    for ``results.csv`` next to the image directory. Returns ``[]`` when neither is found.
    """
    image_dir = str(image_dir)

    if captions_path is not None:
        captions_path = str(captions_path)
        items = (
            _read_flickr_karpathy(captions_path, image_dir)
            if captions_path.endswith(".json")
            else _read_flickr_csv(captions_path, image_dir)
        )
        random.Random(seed).shuffle(items)
        return items[:num_samples]

    parent = os.path.dirname(image_dir)
    karpathy_candidates = [
        os.path.join(parent, "dataset_flickr30k.json"),
        os.path.join(parent, "karpathy", "dataset_flickr30k.json"),
    ]
    for candidate in karpathy_candidates:
        if os.path.isfile(candidate):
            items = _read_flickr_karpathy(candidate, image_dir)
            random.Random(seed).shuffle(items)
            print(f"  Loaded Flickr30k from {candidate}: {len(items)} items")
            return items[:num_samples]

    csv_candidates = [
        os.path.join(parent, "results.csv"),
        os.path.abspath(os.path.join(image_dir, "..", "results.csv")),
        os.path.join(os.path.dirname(parent), "results.csv"),
    ]
    for candidate in csv_candidates:
        if os.path.isfile(candidate):
            items = _read_flickr_csv(candidate, image_dir)
            random.Random(seed).shuffle(items)
            print(f"  Loaded Flickr30k from {candidate}: {len(items)} items")
            return items[:num_samples]

    print("  WARNING: Flickr30k annotations not found. Skipping.")
    return []
