"""Statistics of the CC3M-derived training set.

Reads the generated-annotation JSONL (original caption, five positive captions, hard
negatives, paragraph, Llama quality score) and reports how long each kind of text is and
how the quality scores are distributed.

Outputs:
    ``table_dataset_stats.tex``
    ``fig_para_length_dist``       -- paragraph word counts
    ``fig_llama_score_dist``       -- Llama 3.2 Vision quality scores
    ``fig_text_length_comparison`` -- caption vs positive vs paragraph lengths

Usage:
    python -m capara.analysis.dataset_stats [--jsonl PATH] [--fig_dir DIR] [--table_dir DIR]
"""

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from capara.analysis.style import use_paper_style
from capara.common.paths import ANNOTATIONS_DIR, FIGURES_DIR, TABLES_DIR

DEFAULT_JSONL = ANNOTATIONS_DIR / "cc3m_qwen_llama_500000.jsonl"

CAPTION_COLOR = "#4E79A7"
POSITIVE_COLOR = "#F28E2B"
PARAGRAPH_COLOR = "#E15759"
SCORE_COLOR = "#76B7B2"


def load_records(jsonl_path: Path) -> list[dict]:
    """Load the annotation JSONL, skipping any malformed line."""
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(paragraph: str) -> int:
    normalised = paragraph.replace("!", ".").replace("?", ".")
    return len([s for s in normalised.split(".") if s.strip()])


def collect_stats(records: Sequence[dict]) -> dict[str, list]:
    """Word counts, per-image counts and quality scores, one list per statistic."""
    stats: dict[str, list] = {
        "original_words": [],
        "positive_words": [],
        "negative_words": [],
        "paragraph_words": [],
        "paragraph_sentences": [],
        "positive_counts": [],
        "negative_counts": [],
        "llama_scores": [],
    }

    for record in records:
        caption = record.get("original_caption", "")
        if caption:
            stats["original_words"].append(_word_count(caption))

        positives = record.get("positive_captions", [])
        stats["positive_counts"].append(len(positives))
        stats["positive_words"].extend(
            _word_count(c) for c in positives if isinstance(c, str) and c.strip()
        )

        negatives = record.get("hard_negative_captions", [])
        stats["negative_counts"].append(len(negatives))
        stats["negative_words"].extend(
            _word_count(c) for c in negatives if isinstance(c, str) and c.strip()
        )

        paragraph = record.get("paragraph")
        if isinstance(paragraph, str) and paragraph.strip():
            stats["paragraph_words"].append(_word_count(paragraph))
            stats["paragraph_sentences"].append(_sentence_count(paragraph))

        score = record.get("llama_score")
        if isinstance(score, (int, float)):
            stats["llama_scores"].append(int(score))

    return stats


def print_summary(stats: dict[str, list], n_records: int) -> None:
    def describe(values: Sequence[float]) -> str:
        if not values:
            return "N/A"
        return (
            f"mean={statistics.mean(values):.1f}, median={statistics.median(values):.0f}, "
            f"std={statistics.stdev(values):.1f}, min={min(values)}, max={max(values)}"
        )

    n_paragraphs = len(stats["paragraph_words"])
    print("\n" + "=" * 60)
    print(f"Total samples: {n_records:,}")
    print(f"With paragraph: {n_paragraphs:,} ({100 * n_paragraphs / n_records:.1f}%)")
    print(f"\nOriginal caption words: {describe(stats['original_words'])}")
    print(f"Positive caption words: {describe(stats['positive_words'])}")
    print(f"Hard negative words:    {describe(stats['negative_words'])}")
    print(f"Paragraph words:        {describe(stats['paragraph_words'])}")
    print(f"Paragraph sentences:    {describe(stats['paragraph_sentences'])}")
    print(f"Pos captions per image: {describe(stats['positive_counts'])}")
    print(f"Neg captions per image: {describe(stats['negative_counts'])}")
    print(f"Llama scores:           {describe(stats['llama_scores'])}")
    print(f"Llama score distribution: {dict(sorted(Counter(stats['llama_scores']).items()))}")


def generate_latex_table(stats: dict[str, list], n_records: int, out_path: Path) -> None:
    """Word counts, paragraph structure, quality scores and per-image counts."""

    def summarise(values: Sequence[float]) -> tuple[str, str, str, str]:
        if not values:
            return "---", "---", "---", "---"
        return (
            f"{statistics.mean(values):.1f}",
            f"{statistics.median(values):.0f}",
            f"{min(values)}",
            f"{max(values)}",
        )

    n_paragraphs = len(stats["paragraph_words"])
    coverage = f"{100 * n_paragraphs / n_records:.1f}"

    rows = [
        ("Original caption", stats["original_words"]),
        ("Generated positive caption", stats["positive_words"]),
        ("Hard negative caption", stats["negative_words"]),
        ("Paragraph", stats["paragraph_words"]),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset statistics for our CC3M-derived training set ("
        + f"{n_records // 1000}K"
        + r" samples).}",
        r"\label{tab:supp_dataset_stats}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Text type} & \textbf{Mean} & \textbf{Median} & \textbf{Min} & \textbf{Max} \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Word count per text}} \\",
        r"\midrule",
    ]
    for label, values in rows:
        mean, median, minimum, maximum = summarise(values)
        lines.append(f"{label} & {mean} & {median} & {minimum} & {maximum} " + r"\\")

    sent_mean, sent_median, sent_min, sent_max = summarise(stats["paragraph_sentences"])
    score_mean, score_median, score_min, score_max = summarise(stats["llama_scores"])

    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Paragraph structure}} \\",
        r"\midrule",
        f"Sentences per paragraph & {sent_mean} & {sent_median} & {sent_min} & {sent_max} " + r"\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Quality score (Llama 3.2 Vision, 0--10)}} \\",
        r"\midrule",
        f"Llama score & {score_mean} & {score_median} & {score_min} & {score_max} " + r"\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Counts per image}} \\",
        r"\midrule",
        f"Positive captions & {statistics.mean(stats['positive_counts']):.1f} & --- & "
        f"{min(stats['positive_counts'])} & {max(stats['positive_counts'])} " + r"\\",
        f"Hard negative captions & {statistics.mean(stats['negative_counts']):.1f} & --- & "
        f"{min(stats['negative_counts'])} & {max(stats['negative_counts'])} " + r"\\",
        r"Paragraphs & 1.0 & --- & --- & --- \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Coverage: " + coverage + r"\% of samples have a paragraph}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved LaTeX table: {out_path}")


def plot_paragraph_lengths(paragraph_words: Sequence[int], fig_dir: Path) -> None:
    """Histogram of paragraph word counts."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(paragraph_words, bins=50, color=CAPTION_COLOR, edgecolor="white", linewidth=0.5,
            alpha=0.85)
    ax.axvline(
        statistics.mean(paragraph_words),
        color=PARAGRAPH_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {statistics.mean(paragraph_words):.1f}",
    )
    ax.axvline(
        statistics.median(paragraph_words),
        color=POSITIVE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {statistics.median(paragraph_words):.0f}",
    )
    ax.set_xlabel("Word count per paragraph")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of paragraph lengths")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_para_length_dist.pdf")
    fig.savefig(fig_dir / "fig_para_length_dist.png")
    plt.close(fig)
    print("Saved fig_para_length_dist.pdf/.png")


def plot_llama_scores(llama_scores: Sequence[int], fig_dir: Path) -> None:
    """Bar chart of the Llama 3.2 Vision quality scores."""
    counter = Counter(llama_scores)
    scores = sorted(counter)
    counts = [counter[score] for score in scores]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(scores, counts, color=SCORE_COLOR, edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, count in zip(bars, counts, strict=True):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.axvline(
        statistics.mean(llama_scores),
        color=PARAGRAPH_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {statistics.mean(llama_scores):.2f}",
    )
    ax.set_xlabel("Llama 3.2 Vision quality score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of paragraph quality scores")
    ax.set_xticks(range(0, 11))
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_llama_score_dist.pdf")
    fig.savefig(fig_dir / "fig_llama_score_dist.png")
    plt.close(fig)
    print("Saved fig_llama_score_dist.pdf/.png")


def plot_text_length_comparison(stats: dict[str, list], fig_dir: Path) -> None:
    """The three granularity levels on one axis: caption, positive caption, paragraph."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.arange(0, 200, 3)

    for values, color, label in (
        (stats["original_words"], CAPTION_COLOR, "Original caption"),
        (stats["positive_words"], POSITIVE_COLOR, "Generated positive"),
        (stats["paragraph_words"], PARAGRAPH_COLOR, "Paragraph"),
    ):
        ax.hist(
            values,
            bins=bins,
            alpha=0.6,
            color=color,
            label=f"{label} (mean={statistics.mean(values):.1f})",
        )

    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.set_title("Text length comparison across granularity levels")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 180)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_text_length_comparison.pdf")
    fig.savefig(fig_dir / "fig_text_length_comparison.png")
    plt.close(fig)
    print("Saved fig_text_length_comparison.pdf/.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    use_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JSONL...")
    records = load_records(args.jsonl)
    print(f"Loaded {len(records):,} records")
    if not records:
        raise SystemExit(f"No records in {args.jsonl}")

    stats = collect_stats(records)
    print_summary(stats, len(records))

    generate_latex_table(stats, len(records), args.table_dir / "table_dataset_stats.tex")
    plot_paragraph_lengths(stats["paragraph_words"], args.fig_dir)
    plot_llama_scores(stats["llama_scores"], args.fig_dir)
    plot_text_length_comparison(stats, args.fig_dir)

    print(f"\nFigures in {args.fig_dir}, table in {args.table_dir}")


if __name__ == "__main__":
    main()
