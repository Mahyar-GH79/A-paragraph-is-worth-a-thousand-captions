"""Grouped bar chart of image-text alignment across all four benchmarks.

Reads the ``alignment_<dataset>.json`` summaries written by
:mod:`capara.analysis.embedding_space` and renders ``fig_alignment_cross_dataset``:
mean paired cosine similarity per configuration, one bar group per config, one hatched
bar per dataset.

Usage:
    python -m capara.analysis.cross_dataset_alignment [--jsons DS=PATH ...] [--fig_dir DIR]
"""

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capara.analysis.style import (
    CFG_ORDER_WITH_BASELINE,
    DATASET_ORDER,
    DATASET_STYLE,
    DEFAULT_DATASET_STYLE,
    PRETRAINED_LABELS_COMPACT,
    cfg_labels,
    dataset_slug,
    use_paper_style,
)
from capara.common.paths import FIGURES_DIR, RESULTS_DIR

LABELS: dict[str, str] = {**PRETRAINED_LABELS_COMPACT, **cfg_labels(compact=True)}

#: Where embedding_space.py writes its per-dataset alignment summaries.
ALIGNMENT_DIR = RESULTS_DIR / "embedding_space"

FIGURE_NAME = "fig_alignment_cross_dataset"


def plot_cross_dataset(
    means: Mapping[str, Mapping[str, float]],
    fig_dir: Path,
    styles: Sequence[Mapping[str, str]] | None = None,
    baseline_line: bool = True,
) -> None:
    """Grouped bars of mean paired cosine similarity.

    Args:
        means: ``{dataset: {config_tag: mean_cosine}}``.
        styles: bar styles in dataset order; defaults to the per-dataset palette.
        baseline_line: draw a dotted line at the mean pretrained-BLIP alignment.
    """
    dataset_names = list(means)
    tags = [tag for tag in CFG_ORDER_WITH_BASELINE if any(tag in means[ds] for ds in dataset_names)]
    if not tags:
        print("No configurations to plot.")
        return

    n_datasets = len(dataset_names)
    width = 0.8 / n_datasets
    x = np.arange(len(tags))

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 1.3), 5.5))

    for index, dataset in enumerate(dataset_names):
        style = (
            styles[index % len(styles)]
            if styles
            else DATASET_STYLE.get(dataset, DEFAULT_DATASET_STYLE)
        )
        values = [means[dataset].get(tag, 0.0) for tag in tags]
        offset = (index - n_datasets / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=dataset,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=style["edgecolor"],
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, value in zip(bars, values, strict=True):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    fontweight="bold",
                    color=style["edgecolor"],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS.get(tag, tag.upper()) for tag in tags], rotation=30, ha="right", fontsize=9
    )
    ax.set_ylabel("Mean paired cosine similarity")
    ax.set_title("Image-text embedding alignment across datasets", fontsize=13, pad=10)
    ax.legend(fontsize=9, framealpha=0.95, loc="upper right")
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.2, axis="y")

    if baseline_line:
        baselines = [
            means[dataset]["baseline"] for dataset in dataset_names if "baseline" in means[dataset]
        ]
        if baselines:
            ax.axhline(
                float(np.mean(baselines)),
                color="#AAAAAA",
                linewidth=0.8,
                linestyle=":",
                alpha=0.5,
            )

    fig.tight_layout()
    fig.savefig(fig_dir / f"{FIGURE_NAME}.pdf")
    fig.savefig(fig_dir / f"{FIGURE_NAME}.png")
    plt.close(fig)
    print(f"Saved {FIGURE_NAME}.pdf/.png")


def default_alignment_jsons(alignment_dir: Path) -> list[str]:
    """``dataset=path`` pairs for whichever alignment summaries exist."""
    pairs = []
    for dataset in DATASET_ORDER:
        path = alignment_dir / f"alignment_{dataset_slug(dataset)}.json"
        if path.is_file():
            pairs.append(f"{dataset}={path}")
    return pairs


def load_alignment_jsons(pairs: Sequence[str]) -> "OrderedDict[str, dict[str, float]]":
    """Load ``dataset=path`` pairs into ``{dataset: {config: mean cosine}}``."""
    means: OrderedDict[str, dict[str, float]] = OrderedDict()
    for pair in pairs:
        dataset, path = pair.split("=", 1)
        with open(path) as handle:
            stats = json.load(handle)
        means[dataset] = {tag: values.get("mean", 0.0) for tag, values in stats.items()}
        print(f"Loaded {dataset}: {list(means[dataset])}")
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alignment_dir", type=Path, default=ALIGNMENT_DIR)
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=None,
        help="dataset=path pairs, e.g. DOCCI=results/embedding_space/alignment_docci.json",
    )
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    use_paper_style(**{"axes.labelsize": 12, "axes.titlesize": 13})
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    pairs = args.jsons or default_alignment_jsons(args.alignment_dir)
    if not pairs:
        raise SystemExit(
            f"No alignment JSONs in {args.alignment_dir}. "
            "Run capara.analysis.embedding_space first, or pass --jsons."
        )

    plot_cross_dataset(load_alignment_jsons(pairs), args.fig_dir)


if __name__ == "__main__":
    main()
