"""Ablation: scaling positives against hard negatives.

Six runs fine-tune BLIP's text tower on the CC3M shards, with the hyperparameters
of the main experiments and one axis varied:

* ``P1``/``P3``/``P5`` -- the original caption plus the first 1/3/5 generated
  positives, every one of them a retrieval target (multi-positive InfoNCE over
  ``K = 1 + k`` texts).
* ``N1``/``N3``/``N5`` -- the original caption plus the first 1/3/5 hard
  negatives, which only ever enter the image-to-text denominator.

Every run is then scored on Flickr30k, COCO, ShareGPT4V and DOCCI. Results land in
``results/ablations/`` as JSON, a LaTeX table and a 2x2 figure; checkpoints land in
the training-runs directory, which is not tracked in git.

    python -m capara.ablations.pos_neg_scaling
    python -m capara.ablations.pos_neg_scaling --steps table
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from capara.analysis.style import DATASET_ORDER, DATASET_STYLE, use_paper_style
from capara.common.blip import encode_texts, freeze_vision_tower, load_blip
from capara.common.losses import (
    multi_positive_infonce,
    multi_positive_infonce_with_negatives,
)
from capara.common.paths import (
    ABLATIONS_DIR,
    BLIP_MODEL,
    SHARDS_256_DIR,
    TRAIN_RUNS_DIR,
)
from capara.common.shards import (
    ORIGINAL,
    Requirement,
    ShardTextDataset,
    TextBatch,
    TextSource,
    build_text_batch,
    collate_examples,
    count_rows,
    list_shards,
    top_negatives,
    top_positives,
)
from capara.evaluate import (
    Benchmark,
    BlipEncoder,
    evaluate_benchmark,
    load_coco,
    load_docci,
    load_flickr30k,
    load_sharegpt4v,
)
from capara.train import lr_multiplier

RESULTS_FILENAME = "pos_neg_scaling_results.json"
TABLE_FILENAME = "table_pos_neg_scaling.tex"
FIGURE_STEM = "fig_pos_neg_scaling"
CHECKPOINT_FILENAME = "final_model.pt"

#: Only R@1 is reported for this ablation.
KS: tuple[int, ...] = (1,)

#: style.py carries the dataset colours; the markers are local to the line plots.
DATASET_MARKERS: dict[str, str] = {
    "Flickr30k": "o",
    "COCO": "s",
    "ShareGPT4V": "D",
    "DOCCI": "^",
}

STEPS = ("train", "eval", "figure", "table")


@dataclass(frozen=True)
class TrainConfig:
    """The main experiments' hyperparameters, which the ablation reuses verbatim."""

    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    temperature: float = 0.07
    #: Both positives and hard negatives are captions, so one token cap covers them.
    max_length: int = 77
    #: Trailing shards held out from training, as in the main runs.
    val_shards: int = 10
    num_workers: int = 4


TRAIN = TrainConfig()


@dataclass(frozen=True)
class Variant:
    """One point on the positive or the negative axis."""

    tag: str
    count: int
    negative: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("P1", 1, negative=False),
    Variant("P3", 3, negative=False),
    Variant("P5", 5, negative=False),
    Variant("N1", 1, negative=True),
    Variant("N3", 3, negative=True),
    Variant("N5", 5, negative=True),
)


@dataclass(frozen=True)
class EvalSpec:
    """A loaded benchmark and how this ablation scores it."""

    name: str
    benchmark: Benchmark
    max_length: int
    ks: tuple[int, ...] = KS


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def build_dataset(
    variant: Variant, shard_paths: Sequence[str], seed: int
) -> ShardTextDataset:
    """The text sources for one variant.

    P: the original caption is always a positive, joined by the top ``k`` generated
    positives (padded with the original when a record has fewer).
    N: the original caption is the only positive; the negatives are distractors.
    """
    if variant.negative:
        sources: list[TextSource] = [ORIGINAL, top_negatives(variant.count)]
        requires = [Requirement.ORIGINAL, Requirement.NEGATIVES]
    else:
        sources = [ORIGINAL, top_positives(variant.count, pad_with_original=True)]
        requires = [Requirement.ORIGINAL]

    return ShardTextDataset(
        shard_paths,
        sources=sources,
        requires=requires,
        shuffle_shards=True,
        seed=seed,
    )


def compute_loss(
    model,
    processor,
    images: torch.Tensor,
    batch: TextBatch,
    variant: Variant,
    device: str,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(loss, margin_or_None)``; the margin is a hard-negative diagnostic."""
    txt_pos = encode_texts(
        model, processor, batch.positives, device, config.max_length
    )
    pos_index = batch.pos_index.to(device)
    text_owner = batch.text_owner.to(device)

    if not variant.negative:
        loss = multi_positive_infonce(
            images, txt_pos, pos_index, text_owner, config.temperature
        )
        return loss, None

    txt_neg = encode_texts(
        model, processor, batch.negatives, device, config.max_length
    )
    return multi_positive_infonce_with_negatives(
        images,
        txt_pos,
        txt_neg,
        pos_index,
        batch.neg_index.to(device),
        text_owner,
        config.temperature,
    )


def train_variant(
    variant: Variant,
    shard_paths: Sequence[str],
    runs_dir: Path,
    device: str,
    seed: int,
    model_name: str,
    config: TrainConfig = TRAIN,
) -> Path:
    """Fine-tune one variant's text tower and return the checkpoint path."""
    run_dir = runs_dir / variant.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = count_rows(shard_paths)
    steps_per_epoch = rows // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    print(
        f"\n=== {variant.tag}: {len(shard_paths)} shards, {rows} rows, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total ==="
    )

    model, processor = load_blip(model_name, device=device)
    freeze_vision_tower(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    dataset = build_dataset(variant, shard_paths, seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
        collate_fn=collate_examples,
    )

    history: dict[str, list[float]] = {"train_loss": [], "margin": []}
    step_count = 0

    for epoch in range(config.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_margin = 0.0
        batches = 0

        progress = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"  {variant.tag} epoch {epoch + 1}/{config.epochs}",
        )
        for step, (images, examples) in enumerate(progress):
            if step >= steps_per_epoch:
                break

            images = F.normalize(images.to(device), dim=-1)
            batch = build_text_batch(examples)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.startswith("cuda")
            ):
                loss, margin = compute_loss(
                    model, processor, images, batch, variant, device, config
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # The schedule advances after the step, as in the runs that produced
            # the published numbers: the first step therefore sees the full lr.
            step_count += 1
            for group in optimizer.param_groups:
                group["lr"] = config.lr * lr_multiplier(
                    step_count, total_steps, config.warmup_steps
                )

            running_loss += float(loss.item())
            if margin is not None:
                running_margin += float(margin.item())
            batches += 1

            if step % 50 == 0:
                progress.set_postfix(
                    loss=f"{running_loss / max(1, batches):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        history["train_loss"].append(running_loss / max(1, batches))
        history["margin"].append(running_margin / max(1, batches))
        print(f"  epoch {epoch + 1}: loss={history['train_loss'][-1]:.6f}")

    checkpoint = run_dir / CHECKPOINT_FILENAME
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"  saved {checkpoint}")

    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return checkpoint


def already_trained(checkpoint: Path) -> bool:
    """True when an earlier run left a checkpoint whose loss actually moved."""
    history = checkpoint.parent / "history.json"
    if not checkpoint.is_file() or not history.is_file():
        return False
    losses = json.loads(history.read_text()).get("train_loss", [])
    return bool(losses) and any(loss > 0 for loss in losses)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def build_benchmarks(args: argparse.Namespace) -> list[EvalSpec]:
    """Load the four benchmarks once; every variant is scored against the same items."""
    specs: list[EvalSpec] = []
    for name in args.benchmarks:
        if name == "flickr30k":
            benchmark = load_flickr30k(args.flickr_images_dir, args.flickr_captions)
            specs.append(EvalSpec("Flickr30k", benchmark, 77))
        elif name == "coco":
            specs.append(EvalSpec("COCO", load_coco(args.coco_root), 77))
        elif name == "sharegpt4v":
            benchmark = load_sharegpt4v(
                args.sharegpt4v_samples_dir,
                args.sharegpt4v_json,
                args.coco_root,
                args.llava_images_dir,
            )
            specs.append(
                EvalSpec("ShareGPT4V", benchmark, args.sharegpt4v_max_length)
            )
        elif name == "docci":
            benchmark = load_docci(args.docci_jsonlines, args.docci_images_dir)
            specs.append(EvalSpec("DOCCI", benchmark, 128))
    return specs


def evaluate_checkpoint(
    checkpoint: Path, specs: Sequence[EvalSpec], args: argparse.Namespace
) -> dict[str, dict[str, float]]:
    """Score one checkpoint on every benchmark."""
    encoder = BlipEncoder(
        args.blip_model,
        checkpoint,
        args.device,
        fp16=True,
        batch_size_images=args.batch_size_images,
        batch_size_texts=args.batch_size_texts,
    )

    results: dict[str, dict[str, float]] = {}
    for spec in specs:
        results[spec.name] = evaluate_benchmark(
            encoder, spec.benchmark, spec.max_length, args.device, spec.ks
        )
        metrics = results[spec.name]
        print(
            f"    {spec.name}: I2T R@1={metrics['I2T_R@1'] * 100:.1f}%  "
            f"T2I R@1={metrics['T2I_R@1'] * 100:.1f}%"
        )

    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

#: Table and figure rows: the two axes, and the counts along each.
GROUPS: tuple[tuple[str, tuple[Variant, ...]], ...] = (
    ("Positive", tuple(v for v in VARIANTS if not v.negative)),
    ("Negative", tuple(v for v in VARIANTS if v.negative)),
)


def plot(results: dict[str, dict[str, dict[str, float]]], out_dir: Path) -> None:
    """2x2 line plot: positive/negative scaling by row, I2T/T2I R@1 by column."""
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for row, (axis_label, variants) in enumerate(GROUPS):
        counts = [variant.count for variant in variants]
        xlabel = (
            "Number of positive captions"
            if axis_label == "Positive"
            else "Number of hard negatives"
        )

        for col, metric in enumerate(("I2T_R@1", "T2I_R@1")):
            ax = axes[row, col]

            for dataset in DATASET_ORDER:
                colour = DATASET_STYLE[dataset]["color"]
                points = [
                    (
                        count,
                        results.get(variant.tag, {})
                        .get(dataset, {})
                        .get(metric),
                    )
                    for count, variant in zip(counts, variants, strict=True)
                ]
                points = [(x, y * 100) for x, y in points if y is not None]
                if not points:
                    continue

                xs = [x for x, _ in points]
                ys = [y for _, y in points]
                ax.plot(
                    xs,
                    ys,
                    color=colour,
                    marker=DATASET_MARKERS[dataset],
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    linewidth=2.0,
                    label=dataset,
                )
                for x, y in points:
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=6.5,
                        color=colour,
                        fontweight="bold",
                    )

            direction = "I→T" if metric.startswith("I2T") else "T→I"
            ax.set_title(f"{axis_label} scaling — {direction} R@1", fontsize=11, pad=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("R@1 (%)")
            ax.set_xticks(counts)
            ax.grid(True, alpha=0.2)

            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle(
        "Effect of scaling positives vs. negatives on retrieval", fontsize=14, y=1.01
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{FIGURE_STEM}.{suffix}")
    plt.close(fig)
    print(f"Wrote {out_dir / FIGURE_STEM}.pdf/.png")


def build_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """The ICML-style LaTeX table: one row per variant, R@1 in both directions."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Positive/negative scaling ablation. Final R@1 (\%) across retrieval datasets.}",
        r"\label{tab:pos_neg_scaling}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"% Requires \usepackage{multirow}",
        r"\begin{tabular}{ll" + "cc" * len(DATASET_ORDER) + "}",
        r"\toprule",
        r"\textbf{Type} & \textbf{\# Prompts} & "
        + " & ".join(r"\multicolumn{2}{c}{" + name + "}" for name in DATASET_ORDER)
        + r" \\",
    ]
    lines += [
        r"\cmidrule(lr){" + f"{3 + 2 * i}-{4 + 2 * i}" + "}"
        for i in range(len(DATASET_ORDER))
    ]
    lines.append(
        " &  & "
        + " & ".join([r"I$\to$T", r"T$\to$I"] * len(DATASET_ORDER))
        + r" \\"
    )
    lines.append(r"\midrule")

    for group_index, (axis_label, variants) in enumerate(GROUPS):
        for row_index, variant in enumerate(variants):
            cells = [
                r"\multirow{%d}{*}{%s}" % (len(variants), axis_label) # noqa: UP031  (%-format keeps the LaTeX braces readable)
                if row_index == 0
                else "",
                str(variant.count),
            ]
            for dataset in DATASET_ORDER:
                metrics = results[variant.tag][dataset]
                cells.append(f"{100.0 * metrics['I2T_R@1']:.1f}")
                cells.append(f"{100.0 * metrics['T2I_R@1']:.1f}")
            lines.append(" & ".join(cells) + r" \\")
        if group_index != len(GROUPS) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(results: dict[str, dict[str, dict[str, float]]], out_dir: Path) -> None:
    path = out_dir / TABLE_FILENAME
    path.write_text(build_table(results), encoding="utf-8")
    print(f"Wrote {path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=list(STEPS),
        help="Stages to run. Training resumes from any variant already trained.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.tag for variant in VARIANTS],
        default=[variant.tag for variant in VARIANTS],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("flickr30k", "coco", "sharegpt4v", "docci"),
        default=["flickr30k", "coco", "sharegpt4v", "docci"],
    )

    parser.add_argument("--shards-dir", type=Path, default=SHARDS_256_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ABLATIONS_DIR,
        help="Where the JSON, LaTeX table and figure are written.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=TRAIN_RUNS_DIR / "ablations" / "pos_neg_scaling",
        help="Where checkpoints are written; kept out of results/ because they are large.",
    )

    parser.add_argument("--flickr-images-dir", type=Path, default=None)
    parser.add_argument("--flickr-captions", type=Path, default=None)
    parser.add_argument("--coco-root", type=Path, default=None)
    parser.add_argument("--docci-jsonlines", type=Path, default=None)
    parser.add_argument("--docci-images-dir", type=Path, default=None)
    parser.add_argument("--sharegpt4v-samples-dir", type=Path, default=None)
    parser.add_argument("--sharegpt4v-json", type=Path, default=None)
    parser.add_argument("--llava-images-dir", type=Path, default=None)

    parser.add_argument(
        "--sharegpt4v-max-length",
        type=int,
        default=128,
        help=(
            "Token cap for ShareGPT4V descriptions. NOTE: 128 reproduces the published "
            "ablation (~61%% I2T R@1), but the main results table encodes the same "
            "benchmark at 77, which truncates the descriptions and is why it reports "
            "~52%%. Pass 77 to line this ablation up with the main table."
        ),
    )
    parser.add_argument("--blip-model", type=str, default=BLIP_MODEL)
    parser.add_argument("--batch-size-images", type=int, default=64)
    parser.add_argument("--batch-size-texts", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / RESULTS_FILENAME

    steps = set(args.steps)
    results: dict[str, dict[str, dict[str, float]]] = {}
    if results_path.is_file():
        results = json.loads(results_path.read_text())

    if steps & {"train", "eval"}:
        shards = list_shards(str(args.shards_dir))
        if len(shards) <= TRAIN.val_shards:
            raise RuntimeError(
                f"{args.shards_dir} holds {len(shards)} shards, "
                f"which does not cover the {TRAIN.val_shards} held out from training."
            )
        train_shards = shards[: -TRAIN.val_shards]
        specs = build_benchmarks(args) if "eval" in steps else []

        for variant in (v for v in VARIANTS if v.tag in args.variants):
            checkpoint = args.runs_dir / variant.tag / CHECKPOINT_FILENAME

            if "train" in steps:
                if already_trained(checkpoint):
                    print(f"\n{variant.tag}: already trained, reusing {checkpoint}")
                else:
                    checkpoint = train_variant(
                        variant,
                        train_shards,
                        args.runs_dir,
                        args.device,
                        args.seed,
                        args.blip_model,
                    )

            if "eval" in steps:
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"No checkpoint for {variant.tag}: {checkpoint}. "
                        "Run with --steps train first."
                    )
                print(f"\n  evaluating {variant.tag}")
                results[variant.tag] = evaluate_checkpoint(checkpoint, specs, args)
                # Written after every variant so a long run can be interrupted.
                results_path.write_text(json.dumps(results, indent=2))
                print(f"  wrote {results_path}")

    if not results and steps & {"figure", "table"}:
        raise FileNotFoundError(
            f"No results to report: {results_path} does not exist. Run --steps train eval first."
        )

    if "figure" in steps:
        plot(results, args.out_dir)
    if "table" in steps:
        write_table(results, args.out_dir)


if __name__ == "__main__":
    main()
