"""Pareto trade-off between short-caption and long-description retrieval.

Each configuration is a point: average R@1 over Flickr30k + COCO on the x-axis, average
R@1 over ShareGPT4V + DOCCI on the y-axis. The frontier is drawn over the fine-tuned
configs; the pretrained baselines are shown for reference.

Figures: ``pareto_i2t``, ``pareto_t2i``, ``pareto_panel`` (PDF and PNG).

Usage:
    python -m capara.analysis.pareto [--eval_dir DIR] [--fig_dir DIR]
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capara.analysis.results_tables import Results, load_results
from capara.analysis.style import COLORS, MARKERS, PRETRAINED_LABELS, cfg_labels, use_paper_style
from capara.common.paths import EVAL_RESULTS_DIR, FIGURES_DIR

LABELS: dict[str, str] = {**PRETRAINED_LABELS, **cfg_labels()}

#: Only the BLIP and CLIP baselines are plotted; the Long-CLIP rows stay in the tables.
PLOTTED_BASELINES = ("baseline_blip", "baseline_clip")

SHORT_CAPTION_BENCHMARKS = ("Flickr30k", "COCO")
LONG_DESCRIPTION_BENCHMARKS = ("ShareGPT4V", "DOCCI")

FRONTIER_COLOR = "#D55E00"
BASELINE_COLOR = "#888888"


def _mean_r1(
    cfg_data: dict[str, dict[str, float]],
    benchmarks: Sequence[str],
    metric: str,
) -> float | None:
    """Mean R@1 (in %) over the benchmarks a config was evaluated on."""
    values = [
        cfg_data.get(bench, {})[metric] * 100
        for bench in benchmarks
        if cfg_data.get(bench, {}).get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def short_score(cfg_data: dict, metric: str = "I2T_R@1") -> float | None:
    return _mean_r1(cfg_data, SHORT_CAPTION_BENCHMARKS, metric)


def long_score(cfg_data: dict, metric: str = "I2T_R@1") -> float | None:
    return _mean_r1(cfg_data, LONG_DESCRIPTION_BENCHMARKS, metric)


def pareto_frontier(xs: Sequence[float], ys: Sequence[float]) -> list[int]:
    """Indices of the points that maximise both axes, ordered along x."""
    ordered = sorted(range(len(xs)), key=lambda i: (-xs[i], -ys[i]))
    frontier: list[int] = []
    best_y = -float("inf")
    for i in ordered:
        if ys[i] > best_y:
            frontier.append(i)
            best_y = ys[i]
    return sorted(frontier, key=lambda i: xs[i])


def _plotted_configs(results: Results) -> list[str]:
    return [
        cfg
        for cfg in sorted(results)
        if not cfg.startswith("baseline") or cfg in PLOTTED_BASELINES
    ]


def _scatter_configs(
    ax: plt.Axes,
    results: Results,
    metric: str,
    marker_size: tuple[int, int],
    annotation_fontsize: float,
    annotation_offsets: dict[str, tuple[int, int]],
) -> tuple[list[float], list[float], list[str]]:
    """Draw one point per config; return the plotted coordinates and their tags."""
    xs: list[float] = []
    ys: list[float] = []
    tags: list[str] = []

    baseline_size, trained_size = marker_size
    for cfg in _plotted_configs(results):
        x = short_score(results[cfg], metric)
        y = long_score(results[cfg], metric)
        if x is None or y is None:
            print(f"  Skipping {cfg}: short={x}, long={y}")
            continue

        color = COLORS.get(cfg, "#333333")
        label = LABELS.get(cfg, cfg.upper())
        ax.scatter(
            x,
            y,
            c=color,
            marker=MARKERS.get(cfg, "o"),
            s=baseline_size if cfg.startswith("baseline") else trained_size,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
        ax.annotate(
            label.split(":")[0] if ":" in label else label,
            (x, y),
            textcoords="offset points",
            xytext=annotation_offsets.get(cfg, annotation_offsets["default"]),
            fontsize=annotation_fontsize,
            color=color,
            fontweight="bold",
        )

        xs.append(x)
        ys.append(y)
        tags.append(cfg)

    return xs, ys, tags


def _draw_frontier(
    ax: plt.Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    tags: Sequence[str],
    linewidth: float,
    alpha: float,
    label: str | None = None,
) -> None:
    """Frontier over the fine-tuned configs only."""
    trained = [i for i, tag in enumerate(tags) if not tag.startswith("baseline")]
    if len(trained) < 2:
        return
    txs = [xs[i] for i in trained]
    tys = [ys[i] for i in trained]
    frontier = pareto_frontier(txs, tys)
    if len(frontier) < 2:
        return
    ax.plot(
        [txs[i] for i in frontier],
        [tys[i] for i in frontier],
        color=FRONTIER_COLOR,
        linewidth=linewidth,
        linestyle="--",
        alpha=alpha,
        zorder=3,
        label=label,
    )


def _draw_baseline_guides(
    ax: plt.Axes,
    results: Results,
    metric: str,
    linewidth: float,
    alpha: float,
) -> tuple[float | None, float | None]:
    """Dotted lines through the pretrained BLIP baseline."""
    baseline = results.get("baseline_blip", {})
    x = short_score(baseline, metric)
    y = long_score(baseline, metric)
    if x is not None:
        ax.axvline(x, color=BASELINE_COLOR, linewidth=linewidth, linestyle=":", alpha=alpha)
    if y is not None:
        ax.axhline(y, color=BASELINE_COLOR, linewidth=linewidth, linestyle=":", alpha=alpha)
    return x, y


def plot_pareto(results: Results, fig_dir: Path, direction: str = "i2t") -> None:
    """Single-panel Pareto scatter for one retrieval direction."""
    if direction == "i2t":
        metric = "I2T_R@1"
        arrow = "I→T"
        fname = "pareto_i2t"
    else:
        metric = "T2I_R@1"
        arrow = "T→I"
        fname = "pareto_t2i"

    fig, ax = plt.subplots(figsize=(6.5, 5))

    offsets = {"default": (6, 6), "baseline_blip": (8, -12), "baseline_clip": (8, -12)}
    xs, ys, tags = _scatter_configs(
        ax, results, metric, marker_size=(120, 80), annotation_fontsize=7, annotation_offsets=offsets
    )
    _draw_frontier(ax, xs, ys, tags, linewidth=1.2, alpha=0.6, label="Pareto frontier")
    bl_x, bl_y = _draw_baseline_guides(ax, results, metric, linewidth=0.8, alpha=0.4)

    if bl_x is not None and bl_y is not None:
        ax.text(
            bl_x + 0.5,
            bl_y + 0.5,
            "better on both →",
            fontsize=7,
            color="#666666",
            style="italic",
            alpha=0.6,
        )

    ax.set_xlabel(f"Short-caption avg {arrow} R@1 (%)\n(Flickr30k + COCO)")
    ax.set_ylabel(f"Long-description avg {arrow} R@1 (%)\n(ShareGPT4V + DOCCI)")
    ax.set_title(
        f"Trade-off: short-caption vs. long-description retrieval ({arrow})", fontsize=12, pad=10
    )
    ax.legend(
        loc="lower left",
        framealpha=0.9,
        ncol=2,
        fontsize=7,
        markerscale=0.8,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    fig.tight_layout()
    fig.savefig(fig_dir / f"{fname}.pdf")
    fig.savefig(fig_dir / f"{fname}.png")
    plt.close(fig)
    print(f"Saved {fname}.pdf and {fname}.png")


def plot_pareto_panel(results: Results, fig_dir: Path) -> None:
    """Both directions side by side, with a shared legend."""
    fig, (ax_i2t, ax_t2i) = plt.subplots(1, 2, figsize=(12, 5))

    offsets = {"default": (5, 5)}
    for index, (ax, metric, title) in enumerate(
        [(ax_i2t, "I2T_R@1", "Image → Text"), (ax_t2i, "T2I_R@1", "Text → Image")]
    ):
        xs, ys, tags = _scatter_configs(
            ax, results, metric, marker_size=(100, 65), annotation_fontsize=6,
            annotation_offsets=offsets,
        )
        _draw_frontier(ax, xs, ys, tags, linewidth=1, alpha=0.5)
        _draw_baseline_guides(ax, results, metric, linewidth=0.6, alpha=0.3)

        ax.set_xlabel("Short-caption avg R@1 (%)", fontsize=10)
        ax.set_ylabel("Long-description avg R@1 (%)", fontsize=10)
        ax.set_title(f"({chr(97 + index)}) {title}", fontsize=11)

    handles, labels = ax_i2t.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(handles)),
        framealpha=0.9,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.04),
        markerscale=0.8,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(fig_dir / "pareto_panel.pdf")
    fig.savefig(fig_dir / "pareto_panel.png")
    plt.close(fig)
    print("Saved pareto_panel.pdf and pareto_panel.png")


def print_summary(results: Results) -> None:
    """Per-config R@1 on every benchmark, plus the two averages the figure plots."""
    header = (
        f"\n{'Config':<25s} {'Flickr I2T':>10s} {'COCO I2T':>10s} {'SGv4 I2T':>10s} "
        f"{'DOCCI I2T':>10s} {'Short avg':>10s} {'Long avg':>10s}"
    )
    print(header)
    print("-" * 95)

    def pct(value: float | None) -> str:
        return f"{value * 100:.1f}" if value is not None else "---"

    def avg(value: float | None) -> str:
        return f"{value:.1f}" if value is not None else "---"

    for cfg in sorted(results):
        data = results[cfg]
        cells = [pct(data.get(bench, {}).get("I2T_R@1")) for bench in
                 ("Flickr30k", "COCO", "ShareGPT4V", "DOCCI")]
        print(
            f"{cfg:<25s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s} {cells[3]:>10s} "
            f"{avg(short_score(data)):>10s} {avg(long_score(data)):>10s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval_dir", type=Path, default=EVAL_RESULTS_DIR)
    parser.add_argument(
        "--baseline_flickr_coco",
        type=str,
        default=None,
        help="Extra baseline JSON to fold in, if it lives outside --eval_dir",
    )
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    use_paper_style(
        **{
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    extra = [args.baseline_flickr_coco] if args.baseline_flickr_coco else []
    results = load_results(args.eval_dir, extra)

    print(f"Found {len(results)} configs:")
    for cfg in sorted(results):
        print(f"  {cfg}: {list(results[cfg])}")

    print_summary(results)

    plot_pareto(results, args.fig_dir, direction="i2t")
    plot_pareto(results, args.fig_dir, direction="t2i")
    plot_pareto_panel(results, args.fig_dir)

    print(f"\nAll figures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
