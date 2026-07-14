"""Dataset quality: GPT-4o as an LLM judge, with the image in the prompt.

A random sample of the CC3M-derived annotations is sent to GPT-4o through the OpenAI
Batch API. For every record the judge sees the image and scores, on a 1--10 scale:

1. each generated positive caption -- does it describe the image correctly?
2. each hard negative caption -- is it plausible but wrong for the image?
3. the paragraph -- is it detailed and faithful?

The Batch API is asynchronous, so the run has three steps::

    export OPENAI_API_KEY=sk-...
    python -m capara.analysis.dataset_quality --step prepare   # upload and submit
    python -m capara.analysis.dataset_quality --step check --batch_id batch_xxx
    python -m capara.analysis.dataset_quality --step collect --batch_id batch_xxx

``collect`` downloads the judgements and writes ``fig_dataset_quality_distributions``,
``fig_dataset_statistics``, ``table_dataset_quality.tex`` and ``quality_eval_results.json``.

Cost, for 1000 images at ``detail: low`` (85 image tokens): roughly 1000 x 385 input and
1000 x 80 output tokens, about $1 at Batch API prices.
"""

import argparse
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from capara.analysis.style import use_paper_style
from capara.common.paths import ANNOTATIONS_DIR, FIGURES_DIR, RESULTS_DIR, TABLES_DIR

LLM_MODEL = "gpt-4o"

DEFAULT_JSONL = ANNOTATIONS_DIR / "cc3m_qwen_llama_500000.jsonl"
DEFAULT_WORK_DIR = RESULTS_DIR / "dataset_quality"

#: (display name, key in the judge's JSON response). Paragraph scores are scalars.
CATEGORIES: list[tuple[str, str]] = [
    ("Positive captions", "positive_scores"),
    ("Hard negatives", "negative_scores"),
    ("Paragraphs", "paragraph_score"),
]

JUDGE_COLOR = "#4E79A7"
PARAGRAPH_LENGTH_COLOR = "#4E79A7"
CAPTION_LENGTH_COLOR = "#E15759"
LLAMA_SCORE_COLOR = "#76B7B2"

Judgement = dict[str, object]


def _client():
    """An OpenAI client, or a clear error if the key is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export your key before running this step:\n"
            "    export OPENAI_API_KEY=sk-..."
        )
    from openai import OpenAI

    return OpenAI()


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_records(jsonl_path: Path) -> list[dict]:
    """Every annotation record in the JSONL."""
    records = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_records(jsonl_path: Path, num_samples: int, seed: int = 42) -> list[dict]:
    """A reproducible random subset of the annotations."""
    records = load_records(jsonl_path)
    print(f"Total records in JSONL: {len(records)}")
    if len(records) > num_samples:
        records = random.Random(seed).sample(records, num_samples)
    print(f"Sampled {len(records)} records")
    return records


def category_values(judgements: Sequence[Judgement], key: str) -> list[float]:
    """Flatten one category's scores across records."""
    if key == "paragraph_score":
        return [float(j[key]) for j in judgements if j and key in j]
    return [float(s) for j in judgements if j and key in j for s in j[key]]


# --------------------------------------------------------------------------------------
# Batch API
# --------------------------------------------------------------------------------------


def build_prompt(record: dict) -> str:
    """The judging prompt. The image itself is attached as a separate message part."""
    positives = record.get("positive_captions", [])
    negatives = record.get("hard_negative_captions", [])
    positive_list = "\n".join(f'{i + 1}. "{c}"' for i, c in enumerate(positives))
    negative_list = "\n".join(f'{i + 1}. "{c}"' for i, c in enumerate(negatives))

    return f"""You are evaluating the quality of synthetic training data for vision-language models.

You are given an IMAGE and its original caption, along with several generated texts. Look at the image carefully and score each text on a scale of 1-10.

**Original caption:** "{record.get("original_caption", "")}"

**Task 1: Score each POSITIVE caption (1-10).**
A good positive caption (score 8-10) is a correct, diverse rephrasing that accurately describes what you see in the image. A poor one (1-3) is incorrect or does not match the image.

Positive captions:
{positive_list}

**Task 2: Score each HARD NEGATIVE caption (1-10).**
A good hard negative (score 8-10) sounds plausible but describes something DIFFERENT from the image — it changes important details while keeping the style. A poor one (1-3) is either obviously wrong or actually correct for the image.

Hard negative captions:
{negative_list}

**Task 3: Score the PARAGRAPH (1-10).**
A good paragraph (score 8-10) is detailed, faithful to what you see in the image, and describes objects, attributes, and relationships concretely. A poor one (1-3) is vague, incorrect, or hallucinated.

Paragraph: "{record.get("paragraph", "")}"

**Respond ONLY with valid JSON in this exact format (no other text):**
{{
  "positive_scores": [score1, score2, ...],
  "negative_scores": [score1, score2, ...],
  "paragraph_score": score
}}"""


def prepare_batch(records: Sequence[dict], work_dir: Path) -> Path:
    """Write one Batch API request per record; records without an image URL are skipped."""
    batch_path = work_dir / "batch_requests.jsonl"

    written = skipped = 0
    with open(batch_path, "w") as handle:
        for index, record in enumerate(records):
            image_url = record.get("image_url", "")
            if not image_url:
                skipped += 1
                continue

            request = {
                "custom_id": f"sample-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                # detail=low fixes the image at 85 tokens.
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url, "detail": "low"},
                                },
                                {"type": "text", "text": build_prompt(record)},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
            }
            handle.write(json.dumps(request) + "\n")
            written += 1

    print(f"Wrote {written} requests to {batch_path} (skipped {skipped} without an image URL)")
    return batch_path


def submit_batch(batch_path: Path) -> str:
    """Upload the request file and create the batch job."""
    client = _client()

    print("Uploading batch file...")
    with open(batch_path, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    print(f"Uploaded file: {uploaded.id}")

    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Dataset quality eval with images"},
    )
    print(f"Batch created: {batch.id}\nStatus: {batch.status}")
    print("\nCheck its status with:")
    print(f"  python -m capara.analysis.dataset_quality --step check --batch_id {batch.id}")
    return batch.id


def check_batch(batch_id: str):
    """Print the status of a batch job."""
    batch = _client().batches.retrieve(batch_id)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created: {batch.created_at}")
    if getattr(batch, "request_counts", None):
        counts = batch.request_counts
        print(f"Completed: {counts.completed}/{counts.total} (failed: {counts.failed})")
    if batch.status == "completed":
        print(f"Output file: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"Error file: {batch.error_file_id}")
    return batch


def _parse_judgement(text: str) -> Judgement:
    """Parse the judge's JSON, tolerating a ```json fence."""
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def download_results(batch_id: str, n_records: int, work_dir: Path) -> list[Judgement | None]:
    """Download the batch output and parse one judgement per record (``None`` on failure)."""
    client = _client()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise SystemExit(f"Batch not yet complete. Status: {batch.status}")

    print("Downloading results...")
    raw_path = work_dir / "batch_raw_output.jsonl"
    raw_path.write_bytes(client.files.content(batch.output_file_id).read())
    print(f"Saved raw output: {raw_path}")

    if batch.error_file_id:
        error_path = work_dir / "batch_errors.jsonl"
        error_path.write_bytes(client.files.content(batch.error_file_id).read())
        print(f"Saved errors: {error_path}")

    judgements: list[Judgement | None] = [None] * n_records
    parsed = failed = 0

    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            index = int(result.get("custom_id", "sample--1").replace("sample-", ""))
            choices = result.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                failed += 1
                continue
            try:
                judgements[index] = _parse_judgement(
                    choices[0].get("message", {}).get("content", "").strip()
                )
                parsed += 1
            except (ValueError, KeyError) as error:
                print(f"  [warn] Could not parse the response for sample-{index}: {error}")
                failed += 1

    print(f"Parsed {parsed} responses, {failed} failures")
    return judgements


# --------------------------------------------------------------------------------------
# Figures and tables
# --------------------------------------------------------------------------------------


def plot_distributions(judgements: Sequence[Judgement], fig_dir: Path) -> None:
    """Score distribution per category: histogram of the judgements plus a smoothed density."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    for ax, (title, key) in zip(axes, CATEGORIES, strict=True):
        values = np.array(category_values(judgements, key), dtype=float)
        if values.size == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.hist(
            values,
            bins=np.arange(0.5, 11.5, 1.0),
            density=True,
            color=JUDGE_COLOR,
            alpha=0.30,
            edgecolor="white",
            linewidth=0.6,
            label="GPT-4o (LLM-as-a-judge)",
        )

        if len(set(values.tolist())) > 1:
            x_range = np.linspace(0, 10.5, 200)
            density = stats.gaussian_kde(values, bw_method=0.4)(x_range)
            ax.plot(x_range, density, color=JUDGE_COLOR, linewidth=1.7)
            ax.fill_between(x_range, density, alpha=0.18, color=JUDGE_COLOR)

        # Headroom, so that the mean annotation clears the density curve.
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.25)

        mean, std = float(values.mean()), float(values.std())
        ax.axvline(mean, color=JUDGE_COLOR, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.text(
            mean,
            top * 1.08,
            f"μ={mean:.1f} ± {std:.1f}",
            color=JUDGE_COLOR,
            fontsize=8,
            ha="center",
            fontweight="bold",
        )

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} (n={values.size:,})", fontsize=12)
        ax.set_xlim(0, 10.5)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9, handlelength=1.8)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Dataset quality: GPT-4o as a judge (with image)", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_dir / "fig_dataset_quality_distributions.pdf")
    fig.savefig(fig_dir / "fig_dataset_quality_distributions.png")
    plt.close(fig)
    print("Saved fig_dataset_quality_distributions.pdf/.png")


def _density_panel(ax, values: Sequence[float], color: str, xlabel: str, title: str,
                   annotation: str, offset: float, bw_method: float, x_range) -> None:
    if values:
        density = stats.gaussian_kde(values, bw_method=bw_method)(x_range)
        ax.fill_between(x_range, density, alpha=0.35, color=color)
        ax.plot(x_range, density, color=color, linewidth=1.5)
        ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1)
        ax.text(
            np.mean(values) + offset,
            ax.get_ylim()[1] * 0.9,
            annotation.format(mean=np.mean(values)),
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.15)


def plot_dataset_statistics(records: Sequence[dict], fig_dir: Path) -> None:
    """Paragraph lengths, caption lengths and Llama scores over the whole dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    para_lens = [len(r.get("paragraph", "").split()) for r in records if r.get("paragraph")]
    _density_panel(
        axes[0], para_lens, PARAGRAPH_LENGTH_COLOR, "Word count",
        "(a) Paragraph length distribution", "μ={mean:.0f} words", 2, 0.3,
        np.linspace(min(para_lens) - 5, max(para_lens) + 5, 200) if para_lens else np.array([]),
    )

    cap_lens = [len(r.get("original_caption", "").split()) for r in records]
    _density_panel(
        axes[1], cap_lens, CAPTION_LENGTH_COLOR, "Word count",
        "(b) Original caption length distribution", "μ={mean:.0f} words", 1, 0.3,
        np.linspace(max(0, min(cap_lens) - 2), max(cap_lens) + 5, 200) if cap_lens else np.array([]),
    )

    llama_scores = [r["llama_score"] for r in records if r.get("llama_score", -1) >= 0]
    _density_panel(
        axes[2], llama_scores, LLAMA_SCORE_COLOR, "Score",
        "(c) Llama 3.2 paragraph score distribution", "μ={mean:.1f}", 0.3, 0.4,
        np.linspace(0, 10.5, 200),
    )
    axes[2].set_xlim(0, 10.5)

    fig.suptitle("CC3M synthetic dataset statistics", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_dir / "fig_dataset_statistics.pdf")
    fig.savefig(fig_dir / "fig_dataset_statistics.png")
    plt.close(fig)
    print("Saved fig_dataset_statistics.pdf/.png")


def generate_latex_table(judgements: Sequence[Judgement], out_path: Path) -> None:
    """Mean, std, median and count of the GPT-4o scores, per category."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset quality judged by GPT-4o (LLM as a judge, with the image shown). "
        r"Scores are on a 1--10 scale; $N$ is the number of texts scored.}",
        r"\label{tab:dataset_quality}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Mean} & \textbf{Std} & \textbf{Median} & \textbf{$N$} \\",
        r"\midrule",
    ]

    for label, key in CATEGORIES:
        values = np.array(category_values(judgements, key), dtype=float)
        if values.size == 0:
            lines.append(f"{label} & --- & --- & --- & 0 " + r"\\")
            continue
        lines.append(
            f"{label} & {values.mean():.2f} & {values.std():.2f} & "
            f"{np.median(values):.2f} & {values.size:,} " + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def print_summary(judgements: Sequence[Judgement]) -> None:
    print("\n" + "=" * 60)
    print(f"SUMMARY — {LLM_MODEL} (with image)")
    print("=" * 60)
    for label, key in CATEGORIES:
        values = category_values(judgements, key)
        if values:
            print(
                f"  {label:20s}: mean={np.mean(values):.2f} std={np.std(values):.2f} "
                f"median={np.median(values):.2f} n={len(values):,}"
            )


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


def step_prepare(args: argparse.Namespace) -> None:
    records = sample_records(args.jsonl, args.num_samples, args.seed)

    sampled_path = args.work_dir / "sampled_records.json"
    sampled_path.write_text(json.dumps(records), encoding="utf-8")
    print(f"Saved sampled records: {sampled_path}")

    submit_batch(prepare_batch(records, args.work_dir))


def step_collect(args: argparse.Namespace) -> None:
    sampled_path = args.work_dir / "sampled_records.json"
    if sampled_path.is_file():
        records = json.loads(sampled_path.read_text())
        print(f"Loaded {len(records)} sampled records from {sampled_path}")
    else:
        records = sample_records(args.jsonl, args.num_samples, args.seed)

    judgements = download_results(args.batch_id, len(records), args.work_dir)

    results_path = args.work_dir / "quality_eval_results.json"
    results_path.write_text(
        json.dumps(
            {
                "llm_scores": judgements,
                "model": LLM_MODEL,
                "num_samples": len(records),
                "batch_id": args.batch_id,
                "with_image": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved results: {results_path}")

    valid = [j for j in judgements if j is not None]

    print("\nGenerating distribution plots...")
    plot_distributions(valid, args.fig_dir)

    print("Generating dataset statistics plots...")
    all_records = load_records(args.jsonl)
    print(f"Loaded {len(all_records)} records for the dataset statistics")
    plot_dataset_statistics(all_records, args.fig_dir)

    generate_latex_table(valid, args.table_dir / "table_dataset_quality.tex")
    print_summary(valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--step", required=True, choices=["prepare", "check", "collect"], help="Which step to run"
    )
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_id", type=str, default=None, help="Required by check and collect")
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_paper_style(
        **{
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    for directory in (args.work_dir, args.fig_dir, args.table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.step == "prepare":
        step_prepare(args)
    elif args.step == "check":
        if not args.batch_id:
            raise SystemExit("--batch_id is required for the check step")
        check_batch(args.batch_id)
    else:
        if not args.batch_id:
            raise SystemExit("--batch_id is required for the collect step")
        step_collect(args)


if __name__ == "__main__":
    main()
