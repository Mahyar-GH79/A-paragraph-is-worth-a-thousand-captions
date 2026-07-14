"""Qualitative text-to-image retrieval figures.

Two analyses share the same machinery (load pairs, embed once per model, rank the
ground-truth image for every text query, pick the queries the models disagree on):

``disagreements``
    ShareGPT4V. Compares the fine-tuned configurations against each other and finds the
    queries where paragraph-trained configs succeed and caption-trained configs fail (and
    vice versa). Writes ``fig_qualitative_*`` and ``qualitative_disagreements.json``.

``docci``
    DOCCI. Compares CFG5 against BLIP$_0$, CLIP, Long-CLIP-B and Long-CLIP-L, and keeps
    the queries where only CFG5 retrieves the ground truth at rank 1. Writes the top-5
    retrieved images per model, ``metadata.json``, and ``qualitative_figures.tex``.

Usage:
    python -m capara.analysis.qualitative disagreements --checkpoints baseline=none cfg5=...
    python -m capara.analysis.qualitative docci --cfg5_ckpt /runs/cfg5/final_model.pt
"""

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile

from capara.analysis.models import (
    CLIP_MODEL_PATCH16,
    LONGCLIP_B_CKPT,
    LONGCLIP_L_CKPT,
    LONGCLIP_REPO,
    checkpoint_path,
    clip_encode_image_paths,
    clip_encode_texts,
    free_cuda,
    ground_truth_ranks,
    load_clip,
    load_longclip,
    longclip_encode_image_paths,
    longclip_encode_texts,
    parse_checkpoints,
    top_k_images,
)
from capara.analysis.samples import (
    DOCCI_IMAGE_DIR,
    DOCCI_JSONL,
    SHAREGPT4V_IMAGE_ROOT,
    SHAREGPT4V_SAMPLE_JSON,
    Sample,
    load_docci_records,
    load_sharegpt4v_samples,
)
from capara.analysis.style import (
    MODEL_LABELS,
    MODEL_TAGS,
    PRETRAINED_LABELS_COMPACT,
    cfg_labels,
    escape_latex,
    use_paper_style,
)
from capara.common.blip import encode_image_paths, encode_texts_batched, load_blip
from capara.common.paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS: dict[str, str] = {**PRETRAINED_LABELS_COMPACT, **cfg_labels(compact=True, prefix="CFG")}

PARAGRAPH_CFGS = ("cfg4", "cfg5", "cfg6", "cfg9", "cfg10")
CAPTION_CFGS = ("baseline", "cfg1", "cfg2", "cfg3", "cfg7", "cfg8")

#: A query only counts as a disagreement when the losing group ranks the image this far down.
DISAGREEMENT_MARGIN = 5


# --------------------------------------------------------------------------------------
# ShareGPT4V: where the configurations disagree
# --------------------------------------------------------------------------------------


def find_disagreement_cases(
    ranks: Mapping[str, np.ndarray],
    tags: Sequence[str],
    num_display: int,
) -> tuple[list[int], list[int]]:
    """Queries the paragraph configs win, and queries the caption configs win.

    A win means the winning group retrieves the ground-truth image first while the losing
    group pushes it past :data:`DISAGREEMENT_MARGIN`. Both lists are ordered by how badly
    the losing group failed.
    """
    para_tags = [tag for tag in tags if tag in PARAGRAPH_CFGS]
    cap_tags = [tag for tag in tags if tag in CAPTION_CFGS]
    if not para_tags or not cap_tags:
        para_tags, cap_tags = [tags[-1]], [tags[0]]

    para_wins: list[tuple[int, int]] = []
    cap_wins: list[tuple[int, int]] = []

    for query in range(len(ranks[tags[0]])):
        best_para = min(ranks[tag][query] for tag in para_tags)
        best_cap = min(ranks[tag][query] for tag in cap_tags)
        if best_para == 0 and best_cap > DISAGREEMENT_MARGIN:
            para_wins.append((query, best_cap))
        if best_cap == 0 and best_para > DISAGREEMENT_MARGIN:
            cap_wins.append((query, best_para))

    para_wins.sort(key=lambda item: -item[1])
    cap_wins.sort(key=lambda item: -item[1])

    half = num_display // 2
    selected_para = [query for query, _ in para_wins[:half]]
    selected_cap = [query for query, _ in cap_wins[:half]]

    remaining = num_display - len(selected_para) - len(selected_cap)
    if remaining > 0 and len(para_wins) > half:
        selected_para.extend(query for query, _ in para_wins[half : half + remaining])
    elif remaining > 0 and len(cap_wins) > half:
        selected_cap.extend(query for query, _ in cap_wins[half : half + remaining])

    return selected_para, selected_cap


def _image_strip(
    image_paths: Sequence[str],
    retrieved: Sequence[int],
    ground_truth: int,
) -> np.ndarray:
    """Retrieved thumbnails side by side, green-bordered when correct, red when not."""
    tiles = []
    for index in retrieved:
        border = [0, 180, 0] if index == ground_truth else [200, 0, 0]
        tile = np.full((106, 106, 3), border, dtype=np.uint8)
        try:
            image = Image.open(image_paths[index]).convert("RGB").resize((100, 100))
            tile[3:103, 3:103] = np.array(image)
        except OSError:
            tile = np.zeros((106, 106, 3), dtype=np.uint8)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def make_qualitative_figure(
    queries: Sequence[int],
    case: str,
    samples: Sequence[Sample],
    img_embs: torch.Tensor,
    txt_embs: Mapping[str, torch.Tensor],
    ranks: Mapping[str, np.ndarray],
    tags: Sequence[str],
    image_paths: Sequence[str],
    fig_dir: Path,
    top_k: int = 3,
) -> None:
    """Rows are queries, columns are configs; each cell shows the top-k retrieved images."""
    if not queries:
        return

    fig, axes = plt.subplots(
        len(queries), len(tags) + 1, figsize=(3.5 * (len(tags) + 1), 3.5 * len(queries))
    )
    if len(queries) == 1:
        axes = axes[np.newaxis, :]

    for row, query in enumerate(queries):
        ax = axes[row, 0]
        text = samples[query]["text"]
        ax.text(
            0.05,
            0.95,
            text[:200] + "..." if len(text) > 200 else text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            wrap=True,
            fontfamily="serif",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if row == 0:
            ax.set_title("Query text", fontsize=10, fontweight="bold")

        for col, tag in enumerate(tags):
            ax = axes[row, col + 1]
            retrieved = top_k_images(img_embs, txt_embs[tag][query], k=top_k)
            ax.imshow(_image_strip(image_paths, retrieved, query))

            rank = int(ranks[tag][query])
            color = "#008800" if rank == 0 else "#CC0000" if rank > 10 else "#CC8800"
            ax.set_xlabel(f"GT rank: {rank}", fontsize=8, color=color, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LABELS.get(tag, tag.upper()), fontsize=9, fontweight="bold")

    fig.suptitle(f"Qualitative T→I retrieval: {case}", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    name = f"fig_qualitative_{case.lower().replace(' ', '_')}"
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


def run_disagreements(args: argparse.Namespace) -> None:
    """ShareGPT4V: which queries separate the paragraph configs from the caption configs."""
    checkpoints = parse_checkpoints(args.checkpoints)
    tags = list(checkpoints)
    print(f"Configs: {tags}")

    samples = load_sharegpt4v_samples(
        args.sharegpt4v_json, args.sharegpt4v_image_root, args.num_queries, args.seed
    )
    print(f"Loaded {len(samples)} samples")

    texts = [sample["text"] for sample in samples]
    image_paths = [sample["image_path"] for sample in samples]

    print("\nEncoding images once (the vision tower is frozen)...")
    model, processor = load_blip(device=args.device)
    img_embs = encode_image_paths(model, processor, image_paths, args.device, args.batch_size)
    del model
    free_cuda()

    txt_embs: OrderedDict[str, torch.Tensor] = OrderedDict()
    ranks: OrderedDict[str, np.ndarray] = OrderedDict()

    for tag, checkpoint in checkpoints.items():
        print(f"\n{'=' * 50}\nConfig: {tag}\n{'=' * 50}")
        model, processor = load_blip(checkpoint=checkpoint_path(checkpoint), device=args.device)
        txt_embs[tag] = encode_texts_batched(
            model, processor, texts, args.device, args.max_length, args.batch_size
        )
        ranks[tag] = ground_truth_ranks(img_embs, txt_embs[tag])
        print(
            f"  T2I R@1: {np.mean(ranks[tag] == 0) * 100:.1f}%, "
            f"R@5: {np.mean(ranks[tag] < 5) * 100:.1f}%"
        )
        del model
        free_cuda()

    print("\nFinding disagreement cases...")
    para_wins, cap_wins = find_disagreement_cases(ranks, tags, args.num_display)
    print(f"  Paragraph wins: {len(para_wins)} examples")
    print(f"  Caption wins: {len(cap_wins)} examples")

    for label, queries in (("Paragraph-trained wins", para_wins), ("Caption-trained wins", cap_wins)):
        print(f"\n  {label}:")
        for query in queries:
            detail = ", ".join(f"{tag}={ranks[tag][query]}" for tag in tags)
            print(f"    Query {query}: ranks=[{detail}]")
            print(f"      Text: {texts[query][:100]}...")

    print("\nGenerating qualitative figures...")
    figure_args = (samples, img_embs, txt_embs, ranks, tags, image_paths, args.fig_dir)
    make_qualitative_figure(para_wins, "paragraph wins", *figure_args)
    make_qualitative_figure(cap_wins, "caption wins", *figure_args)
    make_qualitative_figure(
        para_wins[:3] + cap_wins[:3], "retrieval disagreements", *figure_args
    )

    args.json_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "para_wins": [
            {
                "query_idx": int(query),
                "text": texts[query][:300],
                "ranks": {tag: int(ranks[tag][query]) for tag in tags},
            }
            for query in para_wins
        ],
        "cap_wins": [
            {
                "query_idx": int(query),
                "text": texts[query][:300],
                "ranks": {tag: int(ranks[tag][query]) for tag in tags},
            }
            for query in cap_wins
        ],
    }
    out_path = args.json_dir / "qualitative_disagreements.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


# --------------------------------------------------------------------------------------
# DOCCI: CFG5 against the pretrained long-text baselines
# --------------------------------------------------------------------------------------


def _encode_all_models(
    records: Sequence[Sample], args: argparse.Namespace
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Image and text embeddings for BLIP0, CFG5, CLIP, Long-CLIP-B and Long-CLIP-L."""
    image_paths = [record["image_path"] for record in records]
    texts = [record["text"] for record in records]

    img_embs: dict[str, torch.Tensor] = {}
    txt_embs: dict[str, torch.Tensor] = {}

    print("\n=== BLIP0 ===")
    model, processor = load_blip(device=args.device)
    img_embs["blip0"] = encode_image_paths(model, processor, image_paths, args.device)
    txt_embs["blip0"] = encode_texts_batched(
        model, processor, texts, args.device, args.max_length
    )
    del model
    free_cuda()

    print("\n=== CFG5 ===")
    model, processor = load_blip(checkpoint=args.cfg5_ckpt, device=args.device)
    # The vision tower is frozen, so CFG5's image embeddings are BLIP0's.
    img_embs["cfg5"] = img_embs["blip0"]
    txt_embs["cfg5"] = encode_texts_batched(
        model, processor, texts, args.device, args.max_length
    )
    del model
    free_cuda()

    print("\n=== CLIP ===")
    model, processor = load_clip(CLIP_MODEL_PATCH16, args.device)
    img_embs["clip"] = clip_encode_image_paths(model, processor, image_paths, args.device)
    txt_embs["clip"] = clip_encode_texts(model, processor, texts, args.device)
    del model
    free_cuda()

    for tag, checkpoint in (("longclip_b", args.longclip_b_ckpt), ("longclip_l", args.longclip_l_ckpt)):
        print(f"\n=== {MODEL_LABELS[tag]} ===")
        longclip, model, preprocess = load_longclip(checkpoint, args.longclip_repo, args.device)
        img_embs[tag] = longclip_encode_image_paths(model, preprocess, image_paths, args.device)
        txt_embs[tag] = longclip_encode_texts(longclip, model, texts, args.device)
        del model
        free_cuda()

    return img_embs, txt_embs


def _select_docci_samples(
    ranks: Mapping[str, Sequence[int]],
    texts: Sequence[str],
    num_display: int,
    top_k: int,
) -> list[tuple[int, int]]:
    """Queries where CFG5 ranks the image first and the baselines miss the top-k.

    Falls back to "at least three baselines miss" when the strict criterion is too rare,
    then greedily keeps the textually most diverse queries.
    """
    baselines = [tag for tag in MODEL_TAGS if tag != "cfg5"]

    def candidates(min_failures: int) -> list[tuple[int, int]]:
        found = []
        for query in range(len(texts)):
            if ranks["cfg5"][query] != 0:
                continue
            failures = sum(1 for tag in baselines if ranks[tag][query] >= top_k)
            if failures >= min_failures:
                found.append((query, sum(int(ranks[tag][query]) for tag in baselines)))
        found.sort(key=lambda item: -item[1])
        return found

    strict = candidates(len(baselines))
    print(f"  Found {len(strict)} candidates where CFG5 gets rank 0 and all others fail")

    chosen = strict
    if len(strict) < num_display:
        relaxed = candidates(3)
        print("  WARNING: relaxing to CFG5 rank=0 and at least 3 baselines failing")
        print(f"  Relaxed: {len(relaxed)} candidates")
        chosen = relaxed or strict

    selected: list[tuple[int, int]] = []
    for query, score in chosen:
        if len(selected) >= num_display:
            break
        words = set(texts[query].lower().split())
        overlaps = (
            len(words & set(texts[other].lower().split()))
            / max(1, len(words | set(texts[other].lower().split())))
            for other, _ in selected
        )
        if all(overlap <= 0.3 for overlap in overlaps):
            selected.append((query, score))
    return selected


def _save_docci_images(
    selected: Sequence[tuple[int, int]],
    records: Sequence[Sample],
    img_embs: Mapping[str, torch.Tensor],
    txt_embs: Mapping[str, torch.Tensor],
    ranks: Mapping[str, Sequence[int]],
    out_dir: Path,
    top_k: int,
) -> None:
    """Write the top-k retrieved images and the ground truth, per sample and per model."""
    image_paths = [record["image_path"] for record in records]

    for sample_idx, (query, _) in enumerate(selected):
        sample_dir = out_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "query_text.txt").write_text(records[query]["text"], encoding="utf-8")

        for tag in MODEL_TAGS:
            model_dir = sample_dir / tag
            model_dir.mkdir(parents=True, exist_ok=True)

            retrieved = top_k_images(img_embs[tag], txt_embs[tag][query], k=top_k)
            for rank, index in enumerate(retrieved):
                image = Image.open(image_paths[index]).convert("RGB").resize((224, 224))
                image.save(model_dir / f"rank{rank + 1}.jpg", "JPEG", quality=90)

            truth = Image.open(image_paths[query]).convert("RGB").resize((224, 224))
            truth.save(model_dir / "gt.jpg", "JPEG", quality=90)

            (model_dir / "info.txt").write_text(
                f"gt_in_top{top_k}: {query in retrieved.tolist()}\n"
                f"gt_rank: {int(ranks[tag][query])}\n",
                encoding="utf-8",
            )


def generate_docci_latex(
    selected: Sequence[tuple[int, int]],
    records: Sequence[Sample],
    out_path: Path,
    image_prefix: str,
    top_k: int,
) -> None:
    """One ``figure*`` per sample: the query, then top-k retrievals and the ground truth."""
    figures = []

    for sample_idx, (query, _) in enumerate(selected):
        text = escape_latex(records[query]["text"]).replace('"', "''")

        lines = [
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \setlength{\tabcolsep}{1pt}",
            r"    \begin{tabular}{lcccccc}",
            r"        \multicolumn{7}{l}{\parbox{0.95\textwidth}{\scriptsize \textbf{Query:} "
            + text
            + r"}} \\[4pt]",
            r"        \scriptsize \textbf{Model} &",
        ]
        lines += [rf"        \scriptsize \textbf{{Top-{rank}}} &" for rank in range(1, top_k + 1)]
        lines.append(r"        \scriptsize \textbf{GT} \\[3pt]")

        for tag in MODEL_TAGS:
            prefix = f"{image_prefix}/sample_{sample_idx}/{tag}"
            cells = [f"        \\scriptsize {MODEL_LABELS[tag]}"]
            cells += [
                f"\\includegraphics[width=0.13\\textwidth]{{{prefix}/rank{rank}.jpg}}"
                for rank in range(1, top_k + 1)
            ]
            cells.append(f"\\includegraphics[width=0.13\\textwidth]{{{prefix}/gt.jpg}}")
            row = " &\n        ".join(cells)
            lines.append(f"        {row} \\\\" if tag == MODEL_TAGS[-1] else f"        {row} \\\\[4pt]")

        lines += [
            r"    \end{tabular}",
            r"    \caption{Qualitative text-to-image retrieval on DOCCI (sample "
            f"{sample_idx + 1}). Top-{top_k} retrieved images per model. "
            r"Only CFG5 (ours) retrieves the ground truth at rank 1.}",
            r"    \label{fig:qual_docci_" + str(sample_idx) + "}",
            r"\end{figure*}",
            "",
        ]
        figures.append("\n".join(lines))

    out_path.write_text("\n\n".join(figures), encoding="utf-8")
    print(f"  Saved LaTeX: {out_path}")


def run_docci(args: argparse.Namespace) -> None:
    """DOCCI: the queries only the paragraph-trained CFG5 gets right."""
    records = load_docci_records(args.docci_jsonl, args.docci_image_dir)
    print(f"DOCCI: {len(records)} samples")
    texts = [record["text"] for record in records]

    img_embs, txt_embs = _encode_all_models(records, args)

    print("\n=== Computing ranks ===")
    ranks = {}
    for tag in MODEL_TAGS:
        print(f"  {tag}...")
        ranks[tag] = ground_truth_ranks(img_embs[tag], txt_embs[tag])

    print("\n=== Finding disagreement samples ===")
    selected = _select_docci_samples(ranks, texts, args.num_display, args.top_k)
    print(f"\n  Selected {len(selected)} diverse samples:")
    for sample_idx, (query, score) in enumerate(selected):
        print(f"    Sample {sample_idx}: query_idx={query}, others_rank_sum={score}")
        for tag in MODEL_TAGS:
            print(f"      {tag}: rank={ranks[tag][query]}")

    print("\n=== Saving images ===")
    args.image_dir.mkdir(parents=True, exist_ok=True)
    _save_docci_images(selected, records, img_embs, txt_embs, ranks, args.image_dir, args.top_k)

    args.json_dir.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            "sample_idx": sample_idx,
            "query_idx": int(query),
            "text": records[query]["text"],
            "image_file": records[query].get("image_file", ""),
            "ranks": {tag: int(ranks[tag][query]) for tag in MODEL_TAGS},
        }
        for sample_idx, (query, _) in enumerate(selected)
    ]
    metadata_path = args.json_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved {metadata_path}")

    print("\n=== Generating LaTeX ===")
    args.table_dir.mkdir(parents=True, exist_ok=True)
    generate_docci_latex(
        selected,
        records,
        args.table_dir / "qualitative_figures.tex",
        args.latex_image_prefix,
        args.top_k,
    )

    print(f"\nImages in {args.image_dir}, metadata in {args.json_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch_size", type=int, default=64)
    common.add_argument("--max_length", type=int, default=128)
    common.add_argument("--device", type=str, default="cuda")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    common.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    common.add_argument("--json_dir", type=Path, default=RESULTS_DIR / "qualitative")

    disagreements = subparsers.add_parser(
        "disagreements", parents=[common], help="ShareGPT4V: paragraph configs vs caption configs"
    )
    disagreements.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="tag=path pairs, e.g. baseline=none cfg1=/runs/cfg1/final_model.pt",
    )
    disagreements.add_argument("--sharegpt4v_json", type=Path, default=SHAREGPT4V_SAMPLE_JSON)
    disagreements.add_argument("--sharegpt4v_image_root", type=Path, default=SHAREGPT4V_IMAGE_ROOT)
    disagreements.add_argument("--num_queries", type=int, default=2000)
    disagreements.add_argument("--num_display", type=int, default=6)

    docci = subparsers.add_parser(
        "docci", parents=[common], help="DOCCI: CFG5 vs BLIP, CLIP and Long-CLIP"
    )
    docci.add_argument("--cfg5_ckpt", type=Path, required=True)
    docci.add_argument("--longclip_repo", type=Path, default=LONGCLIP_REPO)
    docci.add_argument("--longclip_b_ckpt", type=Path, default=LONGCLIP_B_CKPT)
    docci.add_argument("--longclip_l_ckpt", type=Path, default=LONGCLIP_L_CKPT)
    docci.add_argument("--docci_jsonl", type=Path, default=DOCCI_JSONL)
    docci.add_argument("--docci_image_dir", type=Path, default=DOCCI_IMAGE_DIR)
    docci.add_argument("--num_display", type=int, default=4)
    docci.add_argument("--top_k", type=int, default=5)
    docci.add_argument(
        "--image_dir",
        type=Path,
        default=FIGURES_DIR / "qualitative_docci",
        help="Where the retrieved thumbnails are written",
    )
    docci.add_argument(
        "--latex_image_prefix",
        type=str,
        default="figures/qualitative_docci",
        help="Path the generated LaTeX uses to include the thumbnails",
    )

    args = parser.parse_args()

    use_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "disagreements":
        run_disagreements(args)
    else:
        run_docci(args)


if __name__ == "__main__":
    main()
