"""Image-text retrieval evaluation on Flickr30k, COCO, DOCCI and ShareGPT4V.

Evaluates one model over one or more benchmarks and writes Recall@1/5/10 in both
directions to JSON:

    python -m capara.evaluate --model blip --datasets flickr coco
    python -m capara.evaluate --model blip-finetuned \\
        --checkpoint blip_text_train/cfg5_.../final_model.pt \\
        --datasets docci coco --output results/eval

Text truncation
---------------
``--max-text-length`` defaults reproduce the published numbers: 77 tokens for
Flickr30k, COCO and ShareGPT4V, 128 for DOCCI.

ShareGPT4V is the trap. Its descriptions are paragraphs, and the published
ShareGPT4V numbers were produced with BLIP's text encoder capped at 77 tokens,
which truncates most of them mid-paragraph. Raising the cap to 128 -- as the
data-efficiency ablation does -- yields substantially higher recall on the same
checkpoints, which is why the two report different numbers for the same
benchmark. The default stays at 77 so this module reproduces the paper; pass
``--max-text-length sharegpt4v=128`` to measure untruncated recall instead.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from capara.common import paths
from capara.common.blip import encode_image_paths, encode_texts_batched, load_blip
from capara.common.metrics import DEFAULT_KS, owner_from_lists, owner_one_to_one, recall_at_k
from capara.common.text import clean_text

# Flickr30k ships a handful of images with truncated EOFs.
ImageFile.LOAD_TRUNCATED_IMAGES = True

#: A benchmark: image paths, flat text list, and the image each text describes.
Benchmark = tuple[list[Path], list[str], torch.Tensor]

# The ShareGPT4V evaluation sample is fixed: these values name the cache file and
# must not drift, or the 80k sample stops being comparable across runs.
SHAREGPT4V_COCO_TARGET = 50_000
SHAREGPT4V_LLAVA_TARGET = 30_000
SHAREGPT4V_SEED = 12345

CLIP_MAX_TEXT_LENGTH = 77

_LAION_REL_RE = re.compile(r"^\d{5}/\d+\.jpg$")


# --------------------------------------------------------------------------
# Dataset loaders
# --------------------------------------------------------------------------


def load_flickr30k(
    images_dir: Path | None = None,
    captions_csv: Path | None = None,
) -> Benchmark:
    """Flickr30k: every image with its ~5 captions from the pipe-delimited CSV.

    Images that are missing or unreadable are skipped, as are images left with
    no caption.
    """
    images_dir = images_dir or paths.FLICKR_ROOT / "flickr30k_images" / "flickr30k_images"
    captions_csv = captions_csv or paths.FLICKR_ROOT / "flickr30k_images" / "results.csv"
    if not captions_csv.exists():
        raise FileNotFoundError(f"Flickr30k captions not found: {captions_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Flickr30k image directory not found: {images_dir}")

    # image file name -> captions, ordered by comment_number.
    numbered: dict[str, list[tuple[int, str]]] = {}
    with open(captions_csv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|", skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"No header in {captions_csv}")
        columns = {name.strip(): name for name in reader.fieldnames}
        required = {"image_name", "comment_number", "comment"}
        if not required.issubset(columns):
            raise ValueError(
                f"{captions_csv} has columns {sorted(columns)}, expected {sorted(required)}"
            )

        for row in reader:
            name = (row.get(columns["image_name"]) or "").strip()
            caption = clean_text(row.get(columns["comment"]))
            if not name or caption is None:
                continue
            raw_index = (row.get(columns["comment_number"]) or "0").strip()
            try:
                index = int(raw_index)
            except ValueError:
                index = 0
            numbered.setdefault(Path(name).name, []).append((index, caption))

    image_paths: list[Path] = []
    texts: list[str] = []
    cap_to_img: list[int] = []
    missing = unreadable = 0

    for name in tqdm(sorted(numbered), desc="flickr30k", leave=False):
        path = images_dir / name
        if not path.exists():
            missing += 1
            continue
        try:
            Image.open(path).convert("RGB")
        except OSError:
            unreadable += 1
            continue

        captions = [caption for _, caption in sorted(numbered[name], key=lambda pair: pair[0])]
        if not captions:
            continue

        image_id = len(image_paths)
        image_paths.append(path)
        for caption in captions:
            texts.append(caption)
            cap_to_img.append(image_id)

    if not image_paths:
        raise RuntimeError(f"No usable Flickr30k images under {images_dir}")
    print(
        f"Flickr30k: {len(image_paths)} images, {len(texts)} captions "
        f"({missing} missing, {unreadable} unreadable)"
    )
    return image_paths, texts, owner_from_lists(cap_to_img)


def load_coco(coco_root: Path | None = None) -> Benchmark:
    """COCO train2017: every image with its ~5 captions.

    train2017 -- not val2017 -- is the published evaluation split.
    """
    coco_root = coco_root or paths.COCO_ROOT
    annotations = coco_root / "annotations" / "captions_train2017.json"
    images_dir = coco_root / "images" / "train2017"
    if not annotations.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {images_dir}")

    data = json.loads(annotations.read_text(encoding="utf-8"))
    file_by_id = {int(image["id"]): image["file_name"] for image in data["images"]}

    texts: list[str] = []
    coco_ids: list[int] = []
    for annotation in data["annotations"]:
        caption = clean_text(annotation.get("caption"))
        if caption is None:
            continue
        texts.append(caption)
        coco_ids.append(int(annotation["image_id"]))

    used_ids = sorted(set(coco_ids))
    local_by_coco_id = {coco_id: index for index, coco_id in enumerate(used_ids)}
    image_paths = [images_dir / file_by_id[coco_id] for coco_id in used_ids]
    cap_to_img = [local_by_coco_id[coco_id] for coco_id in coco_ids]

    print(f"COCO train2017: {len(image_paths)} images, {len(texts)} captions")
    return image_paths, texts, owner_from_lists(cap_to_img)


def load_docci(
    jsonlines: Path | None = None,
    images_dir: Path | None = None,
) -> Benchmark:
    """DOCCI: one long description per image, across all splits."""
    jsonlines = jsonlines or paths.DOCCI_ROOT / "docci_descriptions.jsonlines"
    images_dir = images_dir or paths.DOCCI_ROOT / "images"
    if not jsonlines.exists():
        raise FileNotFoundError(f"DOCCI descriptions not found: {jsonlines}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"DOCCI image directory not found: {images_dir}")

    image_paths: list[Path] = []
    texts: list[str] = []
    with open(jsonlines, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            description = clean_text(record.get("description"))
            image_file = record.get("image_file")
            if description is None or not image_file:
                continue
            path = images_dir / image_file
            if not path.is_file():
                continue
            image_paths.append(path)
            texts.append(description)

    if not image_paths:
        raise RuntimeError(f"No usable DOCCI records under {images_dir}")
    print(f"DOCCI: {len(image_paths)} image-description pairs")
    return image_paths, texts, owner_one_to_one(len(image_paths))


def load_sharegpt4v(
    samples_dir: Path | None = None,
    sharegpt4v_json: Path | None = None,
    coco_root: Path | None = None,
    llava_images_dir: Path | None = None,
) -> Benchmark:
    """ShareGPT4V: the fixed 80k sample of 50k COCO + 30k LLaVA-Pretrain pairs.

    The sample is cached at ``sample_coco50000_llava30000_seed12345.json`` and is
    rebuilt from the raw ShareGPT4V JSON only when that cache is absent, so every
    model is scored on identical items. Image paths are re-resolved from each
    item's ``image_field`` against the configured roots, which keeps a cache
    written on another machine usable.
    """
    samples_dir = samples_dir or paths.SHAREGPT4V_SAMPLES_DIR
    coco_root = coco_root or paths.COCO_ROOT
    llava_images_dir = llava_images_dir or paths.LLAVA_PRETRAIN_ROOT / "images"
    cache = (
        samples_dir
        / f"sample_coco{SHAREGPT4V_COCO_TARGET}_llava{SHAREGPT4V_LLAVA_TARGET}"
        f"_seed{SHAREGPT4V_SEED}.json"
    )

    if cache.exists():
        print(f"ShareGPT4V: loading cached sample {cache}")
        sample = json.loads(cache.read_text(encoding="utf-8"))
    else:
        sample = _build_sharegpt4v_sample(
            sharegpt4v_json or paths.SHAREGPT4V_JSON, coco_root, llava_images_dir
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        print(f"ShareGPT4V: cached sample to {cache}")

    image_paths: list[Path] = []
    texts: list[str] = []
    for item in sample:
        path = _resolve_sharegpt4v_image(item["image_field"], coco_root, llava_images_dir)
        if path is None or not path.is_file():
            path = Path(item["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"ShareGPT4V image missing: {item['image_field']}")
        image_paths.append(path)
        texts.append(item["text"])

    print(f"ShareGPT4V: {len(image_paths)} image-description pairs")
    return image_paths, texts, owner_one_to_one(len(image_paths))


def _sharegpt4v_bucket(image_field: str) -> str | None:
    field = image_field.strip().lstrip("/")
    if field.startswith("coco/train2017/"):
        return "COCO"
    if field.startswith("llava/llava_pretrain/images/") or _LAION_REL_RE.match(field):
        return "LLAVA"
    return None


def _resolve_sharegpt4v_image(
    image_field: str, coco_root: Path, llava_images_dir: Path
) -> Path | None:
    field = image_field.strip().lstrip("/")
    if field.startswith("coco/train2017/"):
        return coco_root / "images" / "train2017" / field.split("coco/train2017/", 1)[1]
    if field.startswith("llava/llava_pretrain/images/"):
        return llava_images_dir / field.split("llava/llava_pretrain/images/", 1)[1]
    if _LAION_REL_RE.match(field):
        return llava_images_dir / field
    return None


def _first_assistant_turn(example: dict[str, object]) -> str | None:
    conversation = example.get("conversations")
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if isinstance(message, dict) and message.get("from") in ("gpt", "assistant"):
            text = clean_text(message.get("value"))
            if text is not None:
                return text
    return None


def _build_sharegpt4v_sample(
    sharegpt4v_json: Path, coco_root: Path, llava_images_dir: Path
) -> list[dict[str, str]]:
    """Draw the deterministic 50k COCO + 30k LLaVA-Pretrain sample."""
    if not sharegpt4v_json.exists():
        raise FileNotFoundError(f"ShareGPT4V JSON not found: {sharegpt4v_json}")

    data = json.loads(sharegpt4v_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level list in {sharegpt4v_json}")

    pools: dict[str, list[dict[str, str]]] = {"COCO": [], "LLAVA": []}
    for example in tqdm(data, desc="indexing ShareGPT4V", leave=False):
        if not isinstance(example, dict):
            continue
        image_field = example.get("image")
        if not isinstance(image_field, str) or not image_field.strip():
            continue
        bucket = _sharegpt4v_bucket(image_field)
        if bucket is None:
            continue
        text = _first_assistant_turn(example)
        if text is None:
            continue
        path = _resolve_sharegpt4v_image(image_field, coco_root, llava_images_dir)
        if path is None or not path.is_file():
            continue
        pools[bucket].append(
            {
                "id": example.get("id"),
                "image_field": image_field,
                "image_path": str(path),
                "text": text,
            }
        )

    targets = {"COCO": SHAREGPT4V_COCO_TARGET, "LLAVA": SHAREGPT4V_LLAVA_TARGET}
    # Offsets +1/+2 keep the two pools' draws independent, as when the cache was built.
    seeds = {"COCO": SHAREGPT4V_SEED + 1, "LLAVA": SHAREGPT4V_SEED + 2}

    sample: list[dict[str, str]] = []
    for bucket, target in targets.items():
        pool = pools[bucket]
        if len(pool) < target:
            raise RuntimeError(
                f"ShareGPT4V needs {target} usable {bucket} items but found {len(pool)}"
            )
        order = list(range(len(pool)))
        random.Random(seeds[bucket]).shuffle(order)
        sample.extend(pool[index] for index in order[:target])

    random.Random(SHAREGPT4V_SEED).shuffle(sample)
    return sample


# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """How a benchmark is loaded, scored and filed on disk."""

    loader: Callable[[argparse.Namespace], Benchmark]
    #: Benchmark key inside the result JSON, e.g. ``"COCO_train2017"``.
    bench_key: str
    #: Subdirectory of the output root; the analysis scripts dispatch on it.
    out_subdir: str
    #: Stem of the result filename.
    file_stem: str
    #: Token cap that reproduces the published numbers.
    default_max_text_length: int
    #: Whether trained results are filed under a per-checkpoint subdirectory.
    nest_by_checkpoint: bool


DATASET_SPECS: dict[str, DatasetSpec] = {
    "flickr": DatasetSpec(
        loader=lambda args: load_flickr30k(args.flickr_images_dir, args.flickr_captions),
        bench_key="Flickr30k",
        out_subdir="Coco-Flickr",
        file_stem="flickr",
        default_max_text_length=77,
        nest_by_checkpoint=True,
    ),
    "coco": DatasetSpec(
        loader=lambda args: load_coco(args.coco_root),
        bench_key="COCO_train2017",
        out_subdir="Coco-Flickr",
        file_stem="coco",
        default_max_text_length=77,
        nest_by_checkpoint=True,
    ),
    "docci": DatasetSpec(
        loader=lambda args: load_docci(args.docci_jsonlines, args.docci_images_dir),
        bench_key="DOCCI",
        out_subdir="Docci",
        file_stem="docci",
        default_max_text_length=128,
        nest_by_checkpoint=False,
    ),
    "sharegpt4v": DatasetSpec(
        loader=lambda args: load_sharegpt4v(
            args.sharegpt4v_samples_dir,
            args.sharegpt4v_json,
            args.coco_root,
            args.llava_images_dir,
        ),
        bench_key=(
            f"ShareGPT4V_COCO{SHAREGPT4V_COCO_TARGET}_LLaVA{SHAREGPT4V_LLAVA_TARGET}_80k"
        ),
        out_subdir="sharegpt4v",
        file_stem=(
            f"sharegpt4v_coco{SHAREGPT4V_COCO_TARGET}_llava{SHAREGPT4V_LLAVA_TARGET}"
            f"_seed{SHAREGPT4V_SEED}"
        ),
        default_max_text_length=77,
        nest_by_checkpoint=True,
    ),
}


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


@contextlib.contextmanager
def autocast_fp16(device: str, fp16: bool) -> Iterator[None]:
    """Half-precision forward passes on CUDA, matching the original scripts."""
    if fp16 and device.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield


class Encoder:
    """A model that embeds images and texts into one normalised space."""

    def encode_images(self, image_paths: Sequence[Path]) -> torch.Tensor:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        raise NotImplementedError


class BlipEncoder(Encoder):
    """BLIP's 256-d ITC space, pretrained or with a fine-tuned text tower."""

    def __init__(
        self,
        model_name: str,
        checkpoint: Path | None,
        device: str,
        fp16: bool,
        batch_size_images: int,
        batch_size_texts: int,
    ) -> None:
        # load_blip raises if the checkpoint matches no parameter, so a wrong
        # checkpoint cannot quietly be scored as the pretrained baseline.
        self.model, self.processor = load_blip(model_name, checkpoint, device)
        self.device = device
        self.fp16 = fp16
        self.batch_size_images = batch_size_images
        self.batch_size_texts = batch_size_texts

    def encode_images(self, image_paths: Sequence[Path]) -> torch.Tensor:
        with autocast_fp16(self.device, self.fp16):
            embeddings = encode_image_paths(
                self.model, self.processor, image_paths, self.device, self.batch_size_images
            )
        return embeddings.float()

    def encode_texts(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        with autocast_fp16(self.device, self.fp16):
            embeddings = encode_texts_batched(
                self.model,
                self.processor,
                texts,
                self.device,
                max_length,
                self.batch_size_texts,
            )
        return embeddings.float()


class ClipEncoder(Encoder):
    """CLIP's joint space. Its text encoder is capped at 77 tokens by design."""

    def __init__(
        self,
        model_name: str,
        device: str,
        fp16: bool,
        batch_size_images: int,
        batch_size_texts: int,
    ) -> None:
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.fp16 = fp16
        self.batch_size_images = batch_size_images
        self.batch_size_texts = batch_size_texts

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path]) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in tqdm(
            range(0, len(image_paths), self.batch_size_images), desc="images", leave=False
        ):
            batch = [
                Image.open(path).convert("RGB")
                for path in image_paths[start : start + self.batch_size_images]
            ]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with autocast_fp16(self.device, self.fp16):
                pooled = self.model.vision_model(
                    pixel_values=pixel_values, return_dict=True
                ).pooler_output
                projected = self.model.visual_projection(pooled)
            chunks.append(F.normalize(projected, dim=-1).float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in tqdm(
            range(0, len(texts), self.batch_size_texts), desc="texts", leave=False
        ):
            inputs = self.processor(
                text=list(texts[start : start + self.batch_size_texts]),
                padding=True,
                truncation=True,
                max_length=CLIP_MAX_TEXT_LENGTH,
                return_tensors="pt",
            )
            with autocast_fp16(self.device, self.fp16):
                pooled = self.model.text_model(
                    input_ids=inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                    return_dict=True,
                ).pooler_output
                projected = self.model.text_projection(pooled)
            chunks.append(F.normalize(projected, dim=-1).float().cpu())
        return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class ModelSpec:
    """How a model is built and how it is keyed in the result JSON."""

    build: Callable[[argparse.Namespace], Encoder]
    #: Top-level key the analysis scripts read the metrics from.
    json_key: str
    needs_checkpoint: bool
    #: Text cap actually applied; CLIP ignores --max-text-length.
    honours_max_text_length: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "blip": ModelSpec(
        build=lambda args: BlipEncoder(
            args.blip_model,
            None,
            args.device,
            args.fp16,
            args.batch_size_images,
            args.batch_size_texts,
        ),
        json_key="BLIP",
        needs_checkpoint=False,
        honours_max_text_length=True,
    ),
    "clip": ModelSpec(
        build=lambda args: ClipEncoder(
            args.clip_model,
            args.device,
            args.fp16,
            args.batch_size_images,
            args.batch_size_texts,
        ),
        json_key="CLIP",
        needs_checkpoint=False,
        honours_max_text_length=False,
    ),
    "blip-finetuned": ModelSpec(
        build=lambda args: BlipEncoder(
            args.blip_model,
            args.checkpoint,
            args.device,
            args.fp16,
            args.batch_size_images,
            args.batch_size_texts,
        ),
        json_key="TRAINED_BLIP",
        needs_checkpoint=True,
        honours_max_text_length=True,
    ),
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate_benchmark(
    encoder: Encoder,
    benchmark: Benchmark,
    max_text_length: int,
    device: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Recall@K in both directions for one benchmark."""
    image_paths, texts, text_owner = benchmark
    image_embeddings = encoder.encode_images(image_paths)
    text_embeddings = encoder.encode_texts(texts, max_text_length)
    return recall_at_k(image_embeddings, text_embeddings, text_owner, ks=ks, device=device)


def checkpoint_tag(checkpoint: Path) -> str:
    """Name a run by its checkpoint's directory, e.g. ``cfg5_cap_plus_para_20260227_181332``."""
    return checkpoint.parent.name or checkpoint.stem


def result_path(
    output_root: Path, spec: DatasetSpec, model: str, tag: str | None
) -> Path:
    """Where a result is filed.

    The analysis scripts dispatch on the subdirectory and on a ``baseline_`` or
    ``eval_`` filename prefix, so both are part of the output contract.
    """
    directory = output_root / spec.out_subdir
    if tag is None:
        return directory / f"baseline_{model}_{spec.file_stem}.json"
    if spec.nest_by_checkpoint:
        directory = directory / tag
    return directory / f"eval_{spec.file_stem}__{tag}.json"


def build_payload(
    spec: DatasetSpec,
    model: str,
    model_spec: ModelSpec,
    model_name: str,
    metrics: dict[str, float],
    benchmark: Benchmark,
    max_text_length: int,
    checkpoint: Path | None,
    tag: str | None,
) -> dict[str, object]:
    """Assemble the result JSON.

    The metrics live under the model's key mapped by benchmark key -- e.g.
    ``{"TRAINED_BLIP": {"DOCCI": {"I2T_R@1": ...}}}`` -- and are repeated flat
    under ``metrics``, which is how the ShareGPT4V results are read back.
    """
    image_paths, texts, _ = benchmark
    payload: dict[str, object] = {
        "dataset": spec.bench_key,
        "model": model,
        "model_name": model_name,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "ckpt_tag": tag,
        "num_samples": len(image_paths),
        "num_images": len(image_paths),
        "num_texts": len(texts),
        "max_text_length": max_text_length,
        model_spec.json_key: {spec.bench_key: metrics},
        "metrics": metrics,
    }
    if spec.file_stem.startswith("sharegpt4v"):
        payload["sample_seed"] = SHAREGPT4V_SEED
        payload["coco_target"] = SHAREGPT4V_COCO_TARGET
        payload["llava_target"] = SHAREGPT4V_LLAVA_TARGET
    return payload


def _resolved_max_text_length(
    args: argparse.Namespace, dataset: str, spec: DatasetSpec, model_spec: ModelSpec
) -> int:
    if not model_spec.honours_max_text_length:
        return CLIP_MAX_TEXT_LENGTH
    return args.max_text_length.get(dataset, spec.default_max_text_length)


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Evaluate one model over the selected benchmarks, writing one JSON each."""
    model_spec = MODEL_SPECS[args.model]
    tag = checkpoint_tag(args.checkpoint) if args.checkpoint else None
    model_name = args.clip_model if args.model == "clip" else args.blip_model

    print(f"Model: {args.model} ({model_name}) on {args.device}")
    if tag:
        print(f"Checkpoint: {args.checkpoint} (tag {tag})")

    encoder = model_spec.build(args)

    summary: dict[str, dict[str, float]] = {}
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        max_text_length = _resolved_max_text_length(args, dataset, spec, model_spec)

        print(f"\n=== {spec.bench_key} (max_text_length={max_text_length}) ===")
        benchmark = spec.loader(args)
        metrics = evaluate_benchmark(encoder, benchmark, max_text_length, args.device)

        for key, value in metrics.items():
            print(f"  {key}: {value * 100:.2f}%")

        destination = result_path(args.output, spec, args.model, tag)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = build_payload(
            spec,
            args.model,
            model_spec,
            model_name,
            metrics,
            benchmark,
            max_text_length,
            args.checkpoint,
            tag,
        )
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  saved: {destination}")

        summary[spec.bench_key] = metrics

    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_max_text_length(values: list[str] | None) -> dict[str, int]:
    """Parse ``--max-text-length 128`` or ``--max-text-length docci=128 coco=77``."""
    overrides: dict[str, int] = {}
    for value in values or []:
        if "=" in value:
            dataset, _, raw = value.partition("=")
            if dataset not in DATASET_SPECS:
                raise argparse.ArgumentTypeError(f"Unknown dataset: {dataset}")
            overrides[dataset] = int(raw)
        else:
            for dataset in DATASET_SPECS:
                overrides[dataset] = int(value)
    return overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-text retrieval on Flickr30k, COCO, DOCCI and ShareGPT4V.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Fine-tuned weights; required for blip-finetuned."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument("--output", type=Path, default=paths.EVAL_RESULTS_DIR)
    parser.add_argument(
        "--max-text-length",
        nargs="+",
        metavar="N|DATASET=N",
        help=(
            "Token cap for the text encoder. Defaults reproduce the paper: "
            "flickr 77, coco 77, sharegpt4v 77, docci 128. Note that 77 truncates "
            "most ShareGPT4V descriptions; 128 gives substantially higher recall. "
            "CLIP is always capped at 77."
        ),
    )

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Half-precision forward passes. Defaults to on for CUDA, off for CPU. "
            "The original DOCCI BLIP evaluation was the one path that ran in fp32; "
            "pass --no-fp16 to reproduce it exactly."
        ),
    )
    parser.add_argument("--batch-size-images", type=int, default=64)
    parser.add_argument("--batch-size-texts", type=int, default=256)
    parser.add_argument("--blip-model", default=paths.BLIP_MODEL)
    parser.add_argument("--clip-model", default=paths.CLIP_MODEL)

    parser.add_argument("--flickr-images-dir", type=Path)
    parser.add_argument("--flickr-captions", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--docci-jsonlines", type=Path)
    parser.add_argument("--docci-images-dir", type=Path)
    parser.add_argument("--sharegpt4v-json", type=Path)
    parser.add_argument("--sharegpt4v-samples-dir", type=Path)
    parser.add_argument("--llava-images-dir", type=Path)

    args = parser.parse_args(argv)

    if MODEL_SPECS[args.model].needs_checkpoint and args.checkpoint is None:
        parser.error(f"--model {args.model} requires --checkpoint")
    if args.checkpoint is not None and not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.fp16 is None:
        args.fp16 = args.device.startswith("cuda")

    try:
        args.max_text_length = _parse_max_text_length(args.max_text_length)
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))

    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
