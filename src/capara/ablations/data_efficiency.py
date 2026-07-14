"""Ablation: how much data the paragraph supervision needs.

CFG5 -- the original caption plus the generated paragraph, as two positives of a
multi-positive InfoNCE -- is trained four times, on 25%, 50%, 75% and 100% of the
CC3M training shards, and each run is scored on Flickr30k, COCO, ShareGPT4V and
DOCCI. The question is whether paragraph supervision only pays off at scale.

The shard subsets are nested: one shuffle, seeded by ``--seed``, then the first
25/50/75/100% of it, so the 50% run sees everything the 25% run saw.

Results land in ``results/ablations/`` as JSON, a LaTeX table and a figure;
checkpoints land in the training-runs directory, which is not tracked in git.

    python -m capara.ablations.data_efficiency
    python -m capara.ablations.data_efficiency --steps table
"""

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from capara.analysis.style import DATASET_ORDER, DATASET_STYLE, use_paper_style
from capara.common.blip import freeze_vision_tower, load_blip
from capara.common.losses import multi_positive_infonce
from capara.common.paths import (
    ABLATIONS_DIR,
    BLIP_MODEL,
    SHARDS_256_DIR,
    TRAIN_RUNS_DIR,
)
from capara.common.shards import (
    ORIGINAL,
    PARAGRAPH,
    Requirement,
    ShardTextDataset,
    build_text_batch,
    collate_examples,
    count_rows,
    list_shards,
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
from capara.train import encode_mixed_lengths, lr_multiplier

RESULTS_FILENAME = "data_efficiency_results.json"
TABLE_FILENAME = "table_data_efficiency.tex"
FIGURE_STEM = "fig_data_efficiency"
CHECKPOINT_FILENAME = "final_model.pt"

FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

#: Flickr30k and COCO are reported at R@1 only; the long-description benchmarks
#: also carry R@5 and R@10, as in the published results file.
CAPTION_KS: tuple[int, ...] = (1,)
DESCRIPTION_KS: tuple[int, ...] = (1, 5, 10)

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
    """CFG5's hyperparameters, which the ablation reuses verbatim."""

    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    temperature: float = 0.07
    max_length_caption: int = 77
    #: A paragraph does not fit in a caption's 77 tokens.
    max_length_paragraph: int = 128
    #: Trailing shards held out from training, as in the main runs.
    val_shards: int = 10
    num_workers: int = 4


TRAIN = TrainConfig()


@dataclass(frozen=True)
class EvalSpec:
    """A loaded benchmark and how this ablation scores it."""

    name: str
    benchmark: Benchmark
    max_length: int
    ks: tuple[int, ...]


def fraction_tag(fraction: float) -> str:
    """``0.25 -> "frac_025"``."""
    return f"frac_{int(fraction * 100):03d}"


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def select_shards(
    shard_paths: Sequence[str], fraction: float, seed: int
) -> list[str]:
    """The nested subset of shards for one fraction.

    The shuffle is reseeded identically for every fraction, so the subsets grow by
    accretion rather than being four independent samples.
    """
    shuffled = list(shard_paths)
    random.Random(seed).shuffle(shuffled)
    count = max(1, int(len(shuffled) * fraction))
    return sorted(shuffled[:count])


def train_fraction(
    fraction: float,
    shard_paths: Sequence[str],
    runs_dir: Path,
    device: str,
    seed: int,
    model_name: str,
    config: TrainConfig = TRAIN,
) -> Path:
    """Fine-tune CFG5 on one fraction of the shards and return the checkpoint path."""
    tag = fraction_tag(fraction)
    run_dir = runs_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = count_rows(shard_paths)
    steps_per_epoch = rows // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    print(
        f"\n=== {tag} ({fraction * 100:.0f}%): {len(shard_paths)} shards, {rows} rows, "
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

    dataset = ShardTextDataset(
        shard_paths,
        sources=[ORIGINAL, PARAGRAPH],
        requires=[Requirement.ORIGINAL, Requirement.PARAGRAPH],
        shuffle_shards=True,
        seed=seed,
    )
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

    history: dict[str, list[float]] = {"train_loss": []}
    step_count = 0

    for epoch in range(config.epochs):
        dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        batches = 0

        progress = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"  {tag} epoch {epoch + 1}/{config.epochs}",
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
                # The caption and the paragraph are both positives for their image,
                # but they are tokenised at their own lengths.
                texts = encode_mixed_lengths(
                    model,
                    processor,
                    batch.positives,
                    batch.pos_is_paragraph,
                    device,
                    config.max_length_caption,
                    config.max_length_paragraph,
                )
                loss = multi_positive_infonce(
                    images,
                    texts,
                    batch.pos_index.to(device),
                    batch.text_owner.to(device),
                    config.temperature,
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
            batches += 1

            if step % 50 == 0:
                progress.set_postfix(
                    loss=f"{running_loss / max(1, batches):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        history["train_loss"].append(running_loss / max(1, batches))
        print(f"  epoch {epoch + 1}: loss={history['train_loss'][-1]:.6f}")

    checkpoint = run_dir / CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "fraction": fraction,
            "num_shards": len(shard_paths),
            "train_rows": rows,
        },
        checkpoint,
    )
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
    """Load the four benchmarks once; every fraction is scored against the same items."""
    specs: list[EvalSpec] = []
    for name in args.benchmarks:
        if name == "flickr30k":
            benchmark = load_flickr30k(args.flickr_images_dir, args.flickr_captions)
            specs.append(EvalSpec("Flickr30k", benchmark, 77, CAPTION_KS))
        elif name == "coco":
            specs.append(EvalSpec("COCO", load_coco(args.coco_root), 77, CAPTION_KS))
        elif name == "sharegpt4v":
            benchmark = load_sharegpt4v(
                args.sharegpt4v_samples_dir,
                args.sharegpt4v_json,
                args.coco_root,
                args.llava_images_dir,
            )
            specs.append(
                EvalSpec(
                    "ShareGPT4V",
                    benchmark,
                    args.sharegpt4v_max_length,
                    DESCRIPTION_KS,
                )
            )
        elif name == "docci":
            benchmark = load_docci(args.docci_jsonlines, args.docci_images_dir)
            specs.append(EvalSpec("DOCCI", benchmark, 128, DESCRIPTION_KS))
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


def plot(results: dict[str, dict[str, dict[str, float]]], out_dir: Path) -> None:
    """R@1 against training-data fraction, one line per benchmark, I2T and T2I."""
    use_paper_style(
        **{
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
        }
    )
    percentages = [int(fraction * 100) for fraction in FRACTIONS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = (
        ("I2T_R@1", axes[0], "(a) Image → Text R@1"),
        ("T2I_R@1", axes[1], "(b) Text → Image R@1"),
    )
    for metric, ax, title in panels:
        for dataset in DATASET_ORDER:
            points = [
                (
                    percentage,
                    results.get(fraction_tag(fraction), {})
                    .get(dataset, {})
                    .get(metric),
                )
                for percentage, fraction in zip(percentages, FRACTIONS, strict=True)
            ]
            points = [(x, y * 100) for x, y in points if y is not None]
            if not points:
                continue

            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                color=DATASET_STYLE[dataset]["color"],
                marker=DATASET_MARKERS[dataset],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=2.0,
                label=dataset,
            )

        ax.set_xlabel("Training data fraction (%)")
        ax.set_ylabel("R@1 (%)")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks(percentages)
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(DATASET_ORDER),
        framealpha=0.95,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
        handletextpad=0.4,
        columnspacing=1.0,
        markerscale=0.9,
    )
    fig.suptitle(
        "Effect of training data scale on paragraph-supervised retrieval "
        "(CFG5: orig+para)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{FIGURE_STEM}.{suffix}")
    plt.close(fig)
    print(f"Wrote {out_dir / FIGURE_STEM}.pdf/.png")


def build_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """The LaTeX table: one row per fraction, R@1 in both directions."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Data efficiency: CFG5 (orig+para) trained on varying fractions of CC3M paragraphs. R@1 (\%).}",
        r"\label{tab:data_efficiency}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "cc" * len(DATASET_ORDER) + "}",
        r"\toprule",
        r"\textbf{Fraction} & "
        + " & ".join(r"\multicolumn{2}{c}{" + name + "}" for name in DATASET_ORDER)
        + r" \\",
    ]
    lines += [
        r"\cmidrule(lr){" + f"{2 + 2 * i}-{3 + 2 * i}" + "}"
        for i in range(len(DATASET_ORDER))
    ]
    lines.append(
        " & " + " & ".join([r"I$\to$T", r"T$\to$I"] * len(DATASET_ORDER)) + r" \\"
    )
    lines.append(r"\midrule")

    for fraction in FRACTIONS:
        metrics_by_dataset = results.get(fraction_tag(fraction), {})
        cells = [f"{int(fraction * 100)}\\%"]
        for dataset in DATASET_ORDER:
            metrics = metrics_by_dataset.get(dataset, {})
            for direction in ("I2T_R@1", "T2I_R@1"):
                value = metrics.get(direction)
                cells.append("---" if value is None else f"{value * 100:.1f}")
        lines.append(" & ".join(cells) + r" \\")

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
        help="Stages to run. Training resumes from any fraction already trained.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        choices=FRACTIONS,
        default=list(FRACTIONS),
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
        default=TRAIN_RUNS_DIR / "ablations" / "data_efficiency",
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

        for fraction in args.fractions:
            tag = fraction_tag(fraction)
            checkpoint = args.runs_dir / tag / CHECKPOINT_FILENAME

            if "train" in steps:
                if already_trained(checkpoint):
                    print(f"\n{tag}: already trained, reusing {checkpoint}")
                else:
                    checkpoint = train_fraction(
                        fraction,
                        select_shards(train_shards, fraction, args.seed),
                        args.runs_dir,
                        args.device,
                        args.seed,
                        args.blip_model,
                    )

            if "eval" in steps:
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"No checkpoint for {tag}: {checkpoint}. "
                        "Run with --steps train first."
                    )
                print(f"\n  evaluating {tag}")
                results[tag] = evaluate_checkpoint(checkpoint, specs, args)
                # Written after every fraction so a long run can be interrupted.
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
