"""Long-CLIP evaluation on Flickr30k, COCO, DOCCI and ShareGPT4V.

Long-CLIP is kept apart from ``capara.evaluate`` because it is not a Hugging Face
model: it is loaded from the vendored upstream checkout (``long_clip/repo``) and
carries its own tokenizer and preprocessing. The benchmarks, the ownership maps
and the recall metric are shared, so the two scripts score identical items.

Long-CLIP's tokenizer has a 248-token context and truncates to it, so
``--max-text-length`` has no counterpart here; long descriptions survive largely
intact, which is the entire point of the comparison.

    python -m capara.eval_longclip --datasets docci --output results/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

from capara.common import paths
from capara.common.metrics import DEFAULT_KS, recall_at_k
from capara.evaluate import DATASET_SPECS, Benchmark, Encoder, autocast_fp16

ImageFile.LOAD_TRUNCATED_IMAGES = True

LONGCLIP_REPO = paths.REPO_ROOT / "long_clip" / "repo"
LONGCLIP_CHECKPOINTS = LONGCLIP_REPO / "checkpoints"

#: One result file per output subdirectory, named as the published ones are.
GROUP_FILE_STEMS: dict[str, str] = {
    "Coco-Flickr": "flickr_coco_results",
    "Docci": "docci",
    "sharegpt4v": "sharegpt4v",
}


@dataclass(frozen=True)
class LongClipCheckpoint:
    """A Long-CLIP variant: ``longclip-B`` at ``long_clip/repo/checkpoints/longclip-B.pt``."""

    tag: str
    path: Path

    @property
    def json_key(self) -> str:
        """Top-level key the analysis scripts read, e.g. ``LONGCLIP_B``."""
        return self.tag.replace("-", "_").upper()

    @property
    def file_tag(self) -> str:
        """Filename fragment, e.g. ``longclip_B``."""
        return self.tag.replace("-", "_")


class LongClipEncoder(Encoder):
    """Long-CLIP's joint space, via the vendored repo's own tokenizer and transforms."""

    def __init__(
        self,
        longclip: ModuleType,
        checkpoint: Path,
        device: str,
        fp16: bool,
        batch_size_images: int,
        batch_size_texts: int,
    ) -> None:
        self.longclip = longclip
        self.model, self.preprocess = longclip.load(str(checkpoint), device=device)
        self.model.eval()
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
            batch = torch.stack(
                [
                    self.preprocess(Image.open(path).convert("RGB"))
                    for path in image_paths[start : start + self.batch_size_images]
                ]
            ).to(self.device)
            with autocast_fp16(self.device, self.fp16):
                features = self.model.encode_image(batch)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], max_length: int) -> torch.Tensor:
        """Embed texts. ``max_length`` is ignored: Long-CLIP truncates at 248 tokens."""
        chunks: list[torch.Tensor] = []
        for start in tqdm(
            range(0, len(texts), self.batch_size_texts), desc="texts", leave=False
        ):
            batch = list(texts[start : start + self.batch_size_texts])
            tokens = self.longclip.tokenize(batch, truncate=True).to(self.device)
            with autocast_fp16(self.device, self.fp16):
                features = self.model.encode_text(tokens)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)


def import_longclip(repo: Path) -> ModuleType:
    """Import ``longclip`` from the vendored checkout."""
    if not (repo / "model").is_dir():
        raise FileNotFoundError(f"Long-CLIP repo has no model/ directory: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import longclip  # type: ignore[import-not-found]

    return longclip


def parse_checkpoints(values: Sequence[str]) -> list[LongClipCheckpoint]:
    """Parse ``tag=path`` arguments, e.g. ``longclip-B=/path/longclip-B.pt``."""
    checkpoints: list[LongClipCheckpoint] = []
    for value in values:
        tag, separator, raw = value.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(f"Expected tag=path, got: {value}")
        path = Path(raw)
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"Checkpoint not found: {path}")
        checkpoints.append(LongClipCheckpoint(tag=tag, path=path))
    return checkpoints


def evaluate_checkpoint(
    checkpoint: LongClipCheckpoint,
    longclip: ModuleType,
    benchmarks: dict[str, Benchmark],
    args: argparse.Namespace,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, dict[str, float]]:
    """Score one Long-CLIP variant over the preloaded benchmarks."""
    print(f"\n### {checkpoint.tag}: {checkpoint.path}")
    encoder = LongClipEncoder(
        longclip,
        checkpoint.path,
        args.device,
        args.fp16,
        args.batch_size_images,
        args.batch_size_texts,
    )

    metrics_by_dataset: dict[str, dict[str, float]] = {}
    for dataset, benchmark in benchmarks.items():
        spec = DATASET_SPECS[dataset]
        print(f"\n=== {spec.bench_key} ===")
        image_paths, texts, text_owner = benchmark
        image_embeddings = encoder.encode_images(image_paths)
        text_embeddings = encoder.encode_texts(texts, spec.default_max_text_length)
        metrics = recall_at_k(
            image_embeddings, text_embeddings, text_owner, ks=ks, device=args.device
        )
        for key, value in metrics.items():
            print(f"  {key}: {value * 100:.2f}%")
        metrics_by_dataset[dataset] = metrics

    return metrics_by_dataset


def write_results(
    checkpoint: LongClipCheckpoint,
    metrics_by_dataset: dict[str, dict[str, float]],
    benchmarks: dict[str, Benchmark],
    output_root: Path,
) -> list[Path]:
    """File one JSON per output subdirectory, keyed as the analysis scripts expect."""
    by_group: dict[str, list[str]] = {}
    for dataset in metrics_by_dataset:
        by_group.setdefault(DATASET_SPECS[dataset].out_subdir, []).append(dataset)

    written: list[Path] = []
    for subdir, datasets in by_group.items():
        payload: dict[str, object] = {
            "model": checkpoint.tag,
            "checkpoint": str(checkpoint.path),
            "num_samples": {
                DATASET_SPECS[dataset].bench_key: len(benchmarks[dataset][0])
                for dataset in datasets
            },
            checkpoint.json_key: {
                DATASET_SPECS[dataset].bench_key: metrics_by_dataset[dataset]
                for dataset in datasets
            },
        }
        # Single-benchmark files also expose the metrics flat, which is how the
        # ShareGPT4V results are read back.
        if len(datasets) == 1:
            payload["metrics"] = metrics_by_dataset[datasets[0]]

        destination = (
            output_root
            / subdir
            / f"baseline_{checkpoint.file_tag}_{GROUP_FILE_STEMS[subdir]}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {destination}")
        written.append(destination)

    return written


def run(args: argparse.Namespace) -> None:
    longclip = import_longclip(args.longclip_repo)
    checkpoints = parse_checkpoints(args.checkpoints)

    # Load each benchmark once and score every checkpoint against it.
    benchmarks: dict[str, Benchmark] = {
        dataset: DATASET_SPECS[dataset].loader(args) for dataset in args.datasets
    }

    for checkpoint in checkpoints:
        metrics_by_dataset = evaluate_checkpoint(checkpoint, longclip, benchmarks, args)
        write_results(checkpoint, metrics_by_dataset, benchmarks, args.output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Long-CLIP on Flickr30k, COCO, DOCCI and ShareGPT4V.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--longclip-repo", type=Path, default=LONGCLIP_REPO)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="TAG=PATH",
        default=[
            f"longclip-B={LONGCLIP_CHECKPOINTS / 'longclip-B.pt'}",
            f"longclip-L={LONGCLIP_CHECKPOINTS / 'longclip-L.pt'}",
        ],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument("--output", type=Path, default=paths.EVAL_RESULTS_DIR)

    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Half-precision forward passes. Defaults to on for CUDA, off for CPU.",
    )
    parser.add_argument("--batch-size-images", type=int, default=64)
    parser.add_argument("--batch-size-texts", type=int, default=64)

    parser.add_argument("--flickr-images-dir", type=Path)
    parser.add_argument("--flickr-captions", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--docci-jsonlines", type=Path)
    parser.add_argument("--docci-images-dir", type=Path)
    parser.add_argument("--sharegpt4v-json", type=Path)
    parser.add_argument("--sharegpt4v-samples-dir", type=Path)
    parser.add_argument("--llava-images-dir", type=Path)

    args = parser.parse_args(argv)
    if args.fp16 is None:
        args.fp16 = args.device.startswith("cuda")

    try:
        parse_checkpoints(args.checkpoints)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
