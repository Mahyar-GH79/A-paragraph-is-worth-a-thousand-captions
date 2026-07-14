"""Training dynamics across the ten configurations, from the saved ``history.json`` logs.

Figures:
    ``fig_training_dynamics`` -- 2x2 panel: train loss, val loss, val I->T R@1, val T->I R@1
    ``fig_training_grouped``  -- val I->T R@1, caption-only configs vs paragraph configs
Table:
    ``table_training_dynamics.tex`` -- final-epoch and best-epoch validation metrics

Reads training logs only: no GPU, no dataset.

Usage:
    python -m capara.analysis.training_dynamics [--train_root DIR] [--fig_dir DIR]
"""

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capara.analysis.style import CFG_ORDER, COLORS, MARKERS, cfg_labels, use_paper_style
from capara.common.paths import FIGURES_DIR, TABLES_DIR, TRAIN_RUNS_DIR

History = dict[str, list[float]]

LABELS: dict[str, str] = cfg_labels()

#: CFG7 (hard negatives) is dashed: its validation curve is the unstable one.
LINESTYLES: dict[str, str] = {"cfg7": "--"}

#: Panels of the 2x2 figure: (history key, title, y-label).
PANELS = [
    ("train_loss", "(a) Training loss", "Loss"),
    ("val_loss", "(b) Validation loss", "Loss"),
    ("val_r1_i2t", "(c) Val I→T R@1", "R@1"),
    ("val_r1_t2i", "(d) Val T→I R@1", "R@1"),
]

CAPTION_CFGS = ("cfg1", "cfg2", "cfg3", "cfg7", "cfg8")
PARAGRAPH_CFGS = ("cfg4", "cfg5", "cfg6", "cfg9", "cfg10")


def detect_cfg(dirname: str) -> str | None:
    """Config tag from a run directory, e.g. ``cfg5_cap_plus_para_20260227_181332`` -> ``cfg5``."""
    match = re.match(r"(cfg\d+)", dirname)
    return match.group(1) if match else None


def load_histories(train_root: Path) -> dict[str, History]:
    """Load one ``history.json`` per config; on duplicates keep the longer run."""
    histories: dict[str, History] = {}

    for run in sorted(Path(train_root).iterdir()):
        if not run.is_dir():
            continue
        cfg = detect_cfg(run.name)
        history_path = run / "history.json"
        if not cfg or not history_path.is_file():
            continue

        history = json.loads(history_path.read_text())
        if cfg in histories:
            if len(history.get("train_loss", [])) <= len(histories[cfg].get("train_loss", [])):
                continue

        histories[cfg] = history
        print(f"  Loaded {cfg} from {run.name}: {len(history.get('train_loss', []))} epochs")

    return histories


def _plot_curve(ax: plt.Axes, cfg: str, values: Sequence[float], markersize: float,
                linewidth: float, alpha: float = 1.0) -> None:
    ax.plot(
        range(1, len(values) + 1),
        values,
        color=COLORS.get(cfg, "#333333"),
        marker=MARKERS.get(cfg, "o"),
        markersize=markersize,
        markeredgecolor="white",
        markeredgewidth=0.4 if markersize < 5 else 0.5,
        label=LABELS.get(cfg, cfg.upper()),
        linestyle=LINESTYLES.get(cfg, "-"),
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_training_dynamics(histories: dict[str, History], fig_dir: Path) -> None:
    """Losses and validation recall against epoch, one line per config."""
    ordered = [cfg for cfg in CFG_ORDER if cfg in histories]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (key, title, ylabel) in zip(axes, PANELS, strict=True):
        for cfg in ordered:
            values = histories[cfg].get(key, [])
            if values:
                _plot_curve(ax, cfg, values, markersize=4, linewidth=1.5, alpha=0.9)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, pad=6)
        max_epochs = max(len(histories[cfg].get(key, [])) for cfg in ordered)
        ax.set_xticks(range(1, max_epochs + 1))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(handles)),
        framealpha=0.95,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.04),
        handletextpad=0.3,
        columnspacing=0.8,
        markerscale=0.9,
    )

    fig.suptitle("Training dynamics across configurations", fontsize=14, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fig_dir / "fig_training_dynamics.pdf")
    fig.savefig(fig_dir / "fig_training_dynamics.png")
    plt.close(fig)
    print("Saved fig_training_dynamics.pdf/.png")


def plot_grouped_dynamics(histories: dict[str, History], fig_dir: Path) -> None:
    """Validation I->T R@1, caption-only configs beside paragraph configs."""
    groups = [
        ([cfg for cfg in CFG_ORDER if cfg in histories and cfg in CAPTION_CFGS],
         "(a) Caption-only configs"),
        ([cfg for cfg in CFG_ORDER if cfg in histories and cfg in PARAGRAPH_CFGS],
         "(b) Paragraph configs"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (group, title) in zip(axes, groups, strict=True):
        for cfg in group:
            values = histories[cfg].get("val_r1_i2t", [])
            if values:
                _plot_curve(ax, cfg, values, markersize=5, linewidth=1.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val I→T R@1")
        ax.set_title(title, fontsize=11, pad=6)
        ax.legend(fontsize=8, framealpha=0.9)

        max_epochs = max(
            (len(histories[cfg].get("val_r1_i2t", [])) for cfg in group), default=10
        )
        ax.set_xticks(range(1, max_epochs + 1))

    fig.suptitle("Validation I→T R@1 convergence by config group", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(fig_dir / "fig_training_grouped.pdf")
    fig.savefig(fig_dir / "fig_training_grouped.png")
    plt.close(fig)
    print("Saved fig_training_grouped.pdf/.png")


def _best_epoch(history: History) -> tuple:
    """(epoch, best val I->T R@1, val T->I R@1 at that epoch); epoch is 1-based."""
    i2t = history.get("val_r1_i2t", [])
    t2i = history.get("val_r1_t2i", [])
    if not i2t:
        return None, None, None
    epoch = int(np.argmax(i2t)) + 1
    best_t2i = t2i[epoch - 1] if len(t2i) >= epoch else None
    return epoch, max(i2t), best_t2i


def print_summary(histories: dict[str, History]) -> None:
    """Final-epoch and best-epoch metrics, for a quick sanity check."""
    print(f"\n{'Config':<8s} {'Epochs':>6s} {'Train L':>8s} {'Val L':>8s} {'I2T R@1':>8s} {'T2I R@1':>8s}")
    print("-" * 50)

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "---"

    def last(history: History, key: str) -> float | None:
        values = history.get(key, [])
        return values[-1] if values else None

    for cfg in CFG_ORDER:
        if cfg not in histories:
            continue
        history = histories[cfg]
        n_epochs = len(history.get("train_loss", []))
        print(
            f"{cfg:<8s} {n_epochs:>6d} {fmt(last(history, 'train_loss')):>8s} "
            f"{fmt(last(history, 'val_loss')):>8s} {fmt(last(history, 'val_r1_i2t')):>8s} "
            f"{fmt(last(history, 'val_r1_t2i')):>8s}"
        )

    print(f"\n{'Config':<8s} {'Best ep':>7s} {'Best I2T':>8s} {'Best T2I':>8s}")
    print("-" * 35)
    for cfg in CFG_ORDER:
        if cfg not in histories:
            continue
        epoch, best_i2t, best_t2i = _best_epoch(histories[cfg])
        if epoch is None:
            continue
        print(f"{cfg:<8s} {epoch:>7d} {best_i2t:>8.4f} {fmt(best_t2i):>8s}")


def generate_latex_table(histories: dict[str, History], out_path: Path) -> None:
    """Final-epoch losses and best-epoch validation recall, one row per config."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training summary: final epoch and best-epoch validation metrics.}",
        r"\label{tab:training_dynamics}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Epochs} & \textbf{Train loss} & \textbf{Val loss} & "
        r"\textbf{Best ep.} & \textbf{Best I$\to$T R@1} & \textbf{Best T$\to$I R@1} \\",
        r"\midrule",
    ]

    def fmt(value: float | None, pct: bool = False) -> str:
        if value is None:
            return "---"
        return f"{value * 100:.1f}" if pct else f"{value:.4f}"

    for cfg in CFG_ORDER:
        if cfg not in histories:
            continue
        history = histories[cfg]
        epoch, best_i2t, best_t2i = _best_epoch(history)
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        cells = [
            LABELS.get(cfg, cfg.upper()).split(":")[0],
            str(len(train_loss)),
            fmt(train_loss[-1] if train_loss else None),
            fmt(val_loss[-1] if val_loss else None),
            str(epoch) if epoch else "---",
            fmt(best_i2t, pct=True),
            fmt(best_t2i, pct=True),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_root", type=Path, default=TRAIN_RUNS_DIR)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    use_paper_style(
        **{
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for training histories...")
    histories = load_histories(args.train_root)
    print(f"\nFound {len(histories)} training runs")
    if not histories:
        raise SystemExit(f"No history.json found under {args.train_root}")

    print_summary(histories)

    print("\nGenerating figures...")
    plot_training_dynamics(histories, args.fig_dir)
    plot_grouped_dynamics(histories, args.fig_dir)

    generate_latex_table(histories, args.table_dir / "table_training_dynamics.tex")

    print(f"\nFigures in {args.fig_dir}, table in {args.table_dir}")


if __name__ == "__main__":
    main()
