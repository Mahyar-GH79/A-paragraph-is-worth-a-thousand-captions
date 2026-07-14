"""How much retrieval each configuration gains from longer text.

The same image-text pairs are re-encoded with the text truncated at 20, 40, ..., 128
tokens, and R@K is recomputed at each length. Paragraph-trained configurations keep
improving as the budget grows; caption-only configurations plateau early. CLIP is
included as a baseline up to its 77-token positional limit.

Per dataset:
    ``fig_text_truncation_<dataset>``        -- R@1 vs truncation length (I->T and T->I)
    ``table_text_truncation_<dataset>.tex``  -- I->T R@1 at each truncation length
    ``text_truncation_results_<dataset>.json``

Usage:
    python -m capara.analysis.text_length \
        --checkpoints baseline=none clip=clip cfg5=/runs/cfg5/final_model.pt \
        --datasets sharegpt4v docci --num_samples 10000
"""

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import ImageFile

from capara.analysis.models import (
    CLIP_MAX_LENGTH,
    checkpoint_path,
    clip_encode_image_paths,
    clip_encode_texts,
    free_cuda,
    load_clip,
    parse_checkpoints,
)
from capara.analysis.samples import (
    DOCCI_IMAGE_DIR,
    DOCCI_JSONL,
    SHAREGPT4V_IMAGE_ROOT,
    SHAREGPT4V_SAMPLE_JSON,
    Sample,
    load_docci_samples,
    load_sharegpt4v_samples,
)
from capara.analysis.style import (
    COLORS,
    MARKERS,
    PRETRAINED_LABELS,
    cfg_labels,
    dataset_slug,
    use_paper_style,
)
from capara.common.blip import encode_image_paths, encode_texts_batched, load_blip
from capara.common.metrics import owner_one_to_one, recall_at_k
from capara.common.paths import CLIP_MODEL, FIGURES_DIR, RESULTS_DIR, TABLES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS: dict[str, str] = {**PRETRAINED_LABELS, **cfg_labels()}

#: The pretrained baselines are drawn as reference curves, not as competing configs.
LINESTYLES: dict[str, str] = {"baseline": "--", "clip": ":"}

#: Token budgets to evaluate. 128 is the training-time maximum; 77 is CLIP's limit.
TRUNC_LENGTHS: list[int] = [20, 40, 60, 80, 100, 128]

#: ``all_results[tag][truncation_length] -> {"I2T_R@1": ..., "T2I_R@1": ..., ...}``
TruncationResults = "OrderedDict[str, OrderedDict[int, Dict[str, float]]]"


def evaluate_truncations(
    img_embs: torch.Tensor,
    encode: callable,
    lengths: Sequence[int],
) -> "OrderedDict[int, dict[str, float]]":
    """Recall at every truncation length, re-encoding the texts each time."""
    owner = owner_one_to_one(img_embs.size(0))
    results: OrderedDict[int, dict[str, float]] = OrderedDict()
    for length in lengths:
        txt_embs = encode(length)
        metrics = recall_at_k(img_embs, txt_embs, owner)
        results[length] = metrics
        print(
            f"  max_len={length:>3d}: I2T R@1={metrics['I2T_R@1'] * 100:.1f}%  "
            f"T2I R@1={metrics['T2I_R@1'] * 100:.1f}%"
        )
    return results


def run_dataset(
    dataset: str,
    samples: Sequence[Sample],
    checkpoints: Mapping[str, str],
    with_clip: bool,
    args: argparse.Namespace,
) -> TruncationResults:
    """Sweep the truncation length for every BLIP config, and for CLIP if requested."""
    print(f"\n{'#' * 70}\n# Dataset: {dataset}\n{'#' * 70}")

    texts = [sample["text"] for sample in samples]
    image_paths = [sample["image_path"] for sample in samples]

    print("\nEncoding BLIP image embeddings once (the vision tower is frozen)...")
    model, processor = load_blip(device=args.device)
    blip_img_embs = encode_image_paths(model, processor, image_paths, args.device, args.batch_size)
    print(f"  BLIP image embeddings: {tuple(blip_img_embs.shape)}")
    del model
    free_cuda()

    all_results: TruncationResults = OrderedDict()

    for tag, checkpoint in checkpoints.items():
        print(f"\n{'=' * 60}\nConfig: {tag} (BLIP)\n  Checkpoint: {checkpoint}\n{'=' * 60}")
        model, processor = load_blip(checkpoint=checkpoint_path(checkpoint), device=args.device)
        all_results[tag] = evaluate_truncations(
            blip_img_embs,
            lambda length, m=model, p=processor: encode_texts_batched(
                m, p, texts, args.device, length, args.batch_size
            ),
            TRUNC_LENGTHS,
        )
        del model
        free_cuda()

    del blip_img_embs
    free_cuda()

    if with_clip:
        print(f"\n{'=' * 60}\nConfig: clip (CLIP baseline)\n{'=' * 60}")
        model, processor = load_clip(CLIP_MODEL, args.device)
        clip_img_embs = clip_encode_image_paths(
            model, processor, image_paths, args.device, args.batch_size
        )
        clip_lengths = [length for length in TRUNC_LENGTHS if length <= CLIP_MAX_LENGTH]
        print(f"  CLIP max_position_embeddings={CLIP_MAX_LENGTH}, testing lengths: {clip_lengths}")
        all_results["clip"] = evaluate_truncations(
            clip_img_embs,
            lambda length, m=model, p=processor: clip_encode_texts(
                m, p, texts, args.device, length, args.batch_size
            ),
            clip_lengths,
        )
        del model, clip_img_embs
        free_cuda()

    return all_results


def plot_truncation_curves(results: TruncationResults, fig_dir: Path, dataset: str) -> None:
    """R@1 against truncation length, one line per config, both retrieval directions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    panels = [("I2T_R@1", "(a) Image → Text R@1"), ("T2I_R@1", "(b) Text → Image R@1")]
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for tag, per_length in results.items():
            is_pretrained = tag in ("baseline", "clip")
            points: list[tuple[int, float]] = [
                (length, per_length[length][metric] * 100)
                for length in TRUNC_LENGTHS
                if length in per_length and metric in per_length[length]
            ]
            if not points:
                continue

            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                color=COLORS.get(tag, "#333333"),
                marker=MARKERS.get(tag, "o"),
                markersize=10 if is_pretrained else 7,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=LABELS.get(tag, tag.upper()),
                linestyle=LINESTYLES.get(tag, "-"),
                linewidth=1.8,
                zorder=3 if is_pretrained else 5,
            )

        ax.set_xlabel("Max token length (truncation)")
        ax.set_ylabel("R@1 (%)")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks(TRUNC_LENGTHS)

        ax.axvline(CLIP_MAX_LENGTH, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(
            CLIP_MAX_LENGTH + 1,
            ax.get_ylim()[0] + 1,
            f"{CLIP_MAX_LENGTH} (default)",
            fontsize=6.5,
            color="#999999",
            rotation=90,
            va="bottom",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3, len(handles)),
        framealpha=0.95,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
        handletextpad=0.4,
        columnspacing=1.0,
        markerscale=0.9,
    )

    fig.suptitle(f"Effect of text truncation on retrieval ({dataset})", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    name = f"fig_text_truncation_{dataset_slug(dataset)}"
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


def generate_latex_table(results: TruncationResults, out_path: Path, dataset: str) -> None:
    """I->T R@1 at each truncation length; rows are configs, columns token budgets."""
    trained = [tag for tag in results if tag.lower() != "baseline"]
    best: dict[int, str] = {}
    for length in TRUNC_LENGTHS:
        best_val, best_tag = -1.0, None
        for tag in trained:
            value = results[tag].get(length, {}).get("I2T_R@1")
            if value is not None and value > best_val:
                best_val, best_tag = value, tag
        if best_tag:
            best[length] = best_tag

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{I2T R@1 (\%) at different text truncation lengths on " + dataset + r". "
        r"Higher values at longer truncations indicate the model effectively uses "
        r"additional text tokens.}",
        r"\label{tab:truncation_" + dataset_slug(dataset) + "}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + "l" + "c" * len(TRUNC_LENGTHS) + "}",
        r"\toprule",
        " & ".join([r"\textbf{Config}"] + [f"{length} tok" for length in TRUNC_LENGTHS]) + r" \\",
        r"\midrule",
    ]

    for tag in results:
        label = LABELS.get(tag, tag.upper())
        cells = [label.split(":")[0] if ":" in label else label]
        for length in TRUNC_LENGTHS:
            value = results[tag].get(length, {}).get("I2T_R@1")
            if value is None:
                cells.append("---")
                continue
            text = f"{value * 100:.1f}"
            cells.append(r"\textbf{" + text + "}" if best.get(length) == tag else text)

        row = " & ".join(cells) + r" \\"
        if tag.lower() == "baseline":
            row += "\n" + r"\midrule"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def save_json(results: TruncationResults, out_path: Path) -> None:
    """Raw recall values, keyed by config then truncation length."""
    raw = {
        tag: {
            str(length): {metric: float(value) for metric, value in metrics.items()}
            for length, metrics in per_length.items()
        }
        for tag, per_length in results.items()
    }
    out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def load_datasets(args: argparse.Namespace) -> list[tuple[str, list[Sample]]]:
    """Load every requested dataset."""
    datasets: list[tuple[str, list[Sample]]] = []
    if "sharegpt4v" in args.datasets:
        print(f"\nLoading {args.num_samples} ShareGPT4V samples...")
        samples = load_sharegpt4v_samples(
            args.sharegpt4v_json, args.sharegpt4v_image_root, args.num_samples, args.seed
        )
        print(f"  Loaded {len(samples)} valid samples")
        datasets.append(("ShareGPT4V", samples))
    if "docci" in args.datasets:
        print(f"\nLoading {args.num_samples} DOCCI samples...")
        samples = load_docci_samples(
            args.docci_jsonl, args.docci_image_dir, args.num_samples, args.seed
        )
        print(f"  Loaded {len(samples)} valid samples")
        datasets.append(("DOCCI", samples))
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="tag=path pairs for the BLIP configs, e.g. baseline=none cfg5=/runs/cfg5/final_model.pt. "
        "Add 'clip=clip' to include the pretrained CLIP baseline.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sharegpt4v", "docci"],
        choices=["sharegpt4v", "docci"],
    )
    parser.add_argument("--sharegpt4v_json", type=Path, default=SHAREGPT4V_SAMPLE_JSON)
    parser.add_argument("--sharegpt4v_image_root", type=Path, default=SHAREGPT4V_IMAGE_ROOT)
    parser.add_argument("--docci_jsonl", type=Path, default=DOCCI_JSONL)
    parser.add_argument("--docci_image_dir", type=Path, default=DOCCI_IMAGE_DIR)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--json_dir", type=Path, default=RESULTS_DIR / "text_length")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_paper_style(
        **{
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    for directory in (args.fig_dir, args.table_dir, args.json_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoints(args.checkpoints)
    with_clip = checkpoints.pop("clip", None) is not None

    print(f"BLIP configs: {list(checkpoints)}")
    print(f"CLIP baseline: {with_clip}")
    print(f"Truncation lengths: {TRUNC_LENGTHS}")

    datasets = load_datasets(args)
    if not datasets:
        raise SystemExit("No datasets requested.")

    for dataset, samples in datasets:
        results = run_dataset(dataset, samples, checkpoints, with_clip, args)

        print(f"\nGenerating figures for {dataset}...")
        plot_truncation_curves(results, args.fig_dir, dataset)
        generate_latex_table(
            results, args.table_dir / f"table_text_truncation_{dataset_slug(dataset)}.tex", dataset
        )
        save_json(
            results, args.json_dir / f"text_truncation_results_{dataset_slug(dataset)}.json"
        )

    print(f"\nFigures in {args.fig_dir}, tables in {args.table_dir}, JSON in {args.json_dir}")


if __name__ == "__main__":
    main()
