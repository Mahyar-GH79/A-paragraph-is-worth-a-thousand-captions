"""The paper's two retrieval tables, built from the evaluation JSONs.

* ``table_short_caption.tex``   -- Flickr30k + COCO
* ``table_long_description.tex`` -- ShareGPT4V + DOCCI
* ``results_two_tables.tex``    -- both, as a compilable standalone document

The evaluation runs write three shapes of JSON (baselines, Flickr/COCO, ShareGPT4V,
DOCCI); :func:`load_results` walks the evaluation tree and folds them into one
``{config: {benchmark: {metric: value}}}`` index that the Pareto analysis reuses.

Usage:
    python -m capara.analysis.results_tables [--eval_dir DIR] [--table_dir DIR]
"""

import argparse
import glob
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

from capara.analysis.style import CFG_TABLE_DISPLAY, TABLE_ROW_ORDER
from capara.common.paths import EVAL_RESULTS_DIR, TABLES_DIR

Results = dict[str, dict[str, dict[str, float]]]

METRIC_KEYS: list[str] = ["I2T_R@1", "I2T_R@5", "I2T_R@10", "T2I_R@1", "T2I_R@5", "T2I_R@10"]

METRIC_SHORT: dict[str, str] = {
    "I2T_R@1": "R@1",
    "I2T_R@5": "R@5",
    "I2T_R@10": "R@10",
    "T2I_R@1": "R@1",
    "T2I_R@5": "R@5",
    "T2I_R@10": "R@10",
}

#: Top-level keys of the baseline JSONs, and the config tag each becomes.
BASELINE_KEYS: dict[str, str] = {
    "BLIP": "baseline_blip",
    "CLIP": "baseline_clip",
    "LONGCLIP_B": "baseline_longclip_b",
    "LONGCLIP_L": "baseline_longclip_l",
}


def detect_cfg(text: str) -> str | None:
    """The ``cfgN`` tag mentioned in a path or filename, if any."""
    match = re.search(r"(cfg\d+)", text)
    return match.group(1) if match else None


def normalize_bench(key: str, filepath: str = "") -> str:
    """Canonical benchmark name from a JSON key (falling back to the file's location)."""
    key_lower = key.lower()
    path_lower = filepath.lower()
    if "flickr" in key_lower:
        return "Flickr30k"
    if "docci" in key_lower or "docci" in path_lower:
        return "DOCCI"
    if "sharegpt" in key_lower or "sharegpt" in path_lower:
        return "ShareGPT4V"
    if "coco" in key_lower:
        return "COCO"
    return key


def _record(results: Results, tag: str, bench: str, metrics: dict[str, float]) -> None:
    slot = results.setdefault(tag, {}).setdefault(bench, {})
    for key in METRIC_KEYS:
        if key in metrics:
            slot[key] = metrics[key]


def _ingest_baseline(path: str, results: Results) -> None:
    """A baseline JSON holds one block per pretrained model."""
    with open(path) as handle:
        data = json.load(handle)
    for json_key, tag in BASELINE_KEYS.items():
        for bench_key, metrics in data.get(json_key, {}).items():
            _record(results, tag, normalize_bench(bench_key, path), metrics)


def _ingest_trained(path: str, results: Results) -> None:
    """Flickr/COCO and DOCCI runs both store their metrics under ``TRAINED_BLIP``."""
    cfg = detect_cfg(path)
    if not cfg:
        return
    with open(path) as handle:
        data = json.load(handle)
    for bench_key, metrics in data.get("TRAINED_BLIP", {}).items():
        _record(results, cfg, normalize_bench(bench_key, path), metrics)


def _ingest_sharegpt4v(path: str, results: Results) -> None:
    """ShareGPT4V runs store a flat ``metrics`` block."""
    cfg = detect_cfg(path)
    if not cfg:
        return
    with open(path) as handle:
        data = json.load(handle)
    metrics = data.get("metrics", {})
    if metrics:
        _record(results, cfg, "ShareGPT4V", metrics)


def load_results(eval_dir: Path, extra_baselines: Sequence[str] = ()) -> Results:
    """Index every evaluation JSON under ``eval_dir`` by config and benchmark."""
    results: Results = {}

    for root, _dirs, files in os.walk(eval_dir):
        rel_root = os.path.relpath(root, eval_dir).lower()
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)

            if fname.startswith("baseline_"):
                _ingest_baseline(path, results)
            elif "coco" in rel_root or "flickr" in rel_root:
                _ingest_trained(path, results)
            elif "sharegpt4v" in rel_root and "sharegpt4v" in fname.lower():
                _ingest_sharegpt4v(path, results)
            elif "docci" in rel_root and fname.startswith("eval_docci_"):
                _ingest_trained(path, results)

    for pattern in extra_baselines:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                _ingest_baseline(path, results)

    return results


def build_cfg_order(results: Results) -> list[str]:
    """Known configs in their canonical order, then anything unexpected."""
    known = [cfg for cfg in TABLE_ROW_ORDER if cfg in results]
    extras = sorted(
        (cfg for cfg in results if cfg not in TABLE_ROW_ORDER),
        key=lambda cfg: (0 if cfg.startswith("baseline") else 1, cfg),
    )
    return known + extras


def _find_best(results: Results, benchmarks: Sequence[str], cfg_order: Sequence[str]) -> dict:
    """The fine-tuned config with the highest score, per (benchmark, metric)."""
    best: dict = {}
    trained = [cfg for cfg in cfg_order if not cfg.startswith("baseline_")]
    for bench in benchmarks:
        for metric in METRIC_KEYS:
            best_val, best_tag = -1.0, None
            for cfg in trained:
                value = results.get(cfg, {}).get(bench, {}).get(metric)
                if value is not None and value > best_val:
                    best_val, best_tag = value, cfg
            if best_tag:
                best[(bench, metric)] = best_tag
    return best


def _fmt(value: float | None, is_best: bool = False) -> str:
    if value is None:
        return "---"
    text = f"{value * 100:.1f}"
    return r"\textbf{" + text + "}" if is_best else text


def generate_table(
    results: Results,
    benchmarks: Sequence[str],
    cfg_order: Sequence[str],
    caption: str,
    label: str,
) -> str | None:
    """One LaTeX table: rows are configs, column groups are (benchmark x direction x K)."""
    active_cfgs = [cfg for cfg in cfg_order if cfg in results]
    active_bench = [
        bench for bench in benchmarks if any(bench in results.get(cfg, {}) for cfg in active_cfgs)
    ]
    if not active_bench:
        return None

    best = _find_best(results, active_bench, cfg_order)
    n_bench = len(active_bench)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{" + "ll" + "c" * (n_bench * 6) + "}",
        r"\toprule",
    ]

    header = ["", ""] + [
        r"\multicolumn{6}{c}{" + bench.replace("_", r"\_") + "}" for bench in active_bench
    ]
    lines.append(" & ".join(header) + r" \\")
    for i in range(n_bench):
        start = 3 + i * 6
        lines.append(r"\cmidrule(lr){" + f"{start}-{start + 5}" + "}")

    directions = ["", ""]
    for _ in active_bench:
        directions.append(r"\multicolumn{3}{c}{Image$\to$Text}")
        directions.append(r"\multicolumn{3}{c}{Text$\to$Image}")
    lines.append(" & ".join(directions) + r" \\")
    for i in range(n_bench):
        start = 3 + i * 6
        lines.append(r"\cmidrule(lr){" + f"{start}-{start + 2}" + "}")
        lines.append(r"\cmidrule(lr){" + f"{start + 3}-{start + 5}" + "}")

    metric_row = [r"\textbf{Model}", r"\textbf{Training text}"]
    for _ in active_bench:
        metric_row.extend(METRIC_SHORT[metric] for metric in METRIC_KEYS)
    lines.append(" & ".join(metric_row) + r" \\")
    lines.append(r"\midrule")

    has_longclip = "baseline_longclip_b" in results or "baseline_longclip_l" in results
    for cfg in active_cfgs:
        display_name, train_text = CFG_TABLE_DISPLAY.get(cfg, (cfg.upper(), "unknown"))
        cells = [display_name, train_text]
        for bench in active_bench:
            bench_data = results.get(cfg, {}).get(bench, {})
            for metric in METRIC_KEYS:
                cells.append(
                    _fmt(bench_data.get(metric), best.get((bench, metric)) == cfg)
                )

        row = " & ".join(cells) + r" \\"
        # Rule between the pretrained baselines and the fine-tuned configs.
        if cfg == "baseline_longclip_l" or (cfg == "baseline_clip" and not has_longclip):
            row += "\n" + r"\midrule"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def standalone_document(tables: Sequence[str]) -> str:
    """Wrap the tables in a minimal document so they can be compiled on their own."""
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=0.5in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{adjustbox}",
        r"\begin{document}",
        r"\pagestyle{empty}",
    ]
    for table in tables:
        lines.append("")
        lines.append(table)
    lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval_dir", type=Path, default=EVAL_RESULTS_DIR)
    parser.add_argument(
        "--baseline_flickr_coco",
        type=str,
        default=None,
        help="Extra baseline JSON to fold in, if it lives outside --eval_dir",
    )
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    args.table_dir.mkdir(parents=True, exist_ok=True)

    extra = [args.baseline_flickr_coco] if args.baseline_flickr_coco else []
    results = load_results(args.eval_dir, extra)
    cfg_order = build_cfg_order(results)

    print(f"Found {len(results)} configs:")
    for cfg in cfg_order:
        print(f"  {cfg}: {list(results[cfg])}")

    short_caption = generate_table(
        results,
        benchmarks=["Flickr30k", "COCO"],
        cfg_order=cfg_order,
        caption=(
            r"Short-caption retrieval (Recall \%) on Flickr30k and COCO. "
            r"\textbf{Bold} indicates best among fine-tuned configurations."
        ),
        label="tab:short_caption",
    )
    long_description = generate_table(
        results,
        benchmarks=["ShareGPT4V", "DOCCI"],
        cfg_order=cfg_order,
        caption=(
            r"Long-description retrieval (Recall \%) on ShareGPT4V and DOCCI. "
            r"\textbf{Bold} indicates best among fine-tuned configurations."
        ),
        label="tab:long_description",
    )

    tables = [table for table in (short_caption, long_description) if table]
    standalone_path = args.table_dir / "results_two_tables.tex"
    standalone_path.write_text(standalone_document(tables), encoding="utf-8")
    print(f"\nSaved {standalone_path}")

    for table, name in ((short_caption, "table_short_caption.tex"),
                        (long_description, "table_long_description.tex")):
        if table:
            path = args.table_dir / name
            path.write_text(table + "\n", encoding="utf-8")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
