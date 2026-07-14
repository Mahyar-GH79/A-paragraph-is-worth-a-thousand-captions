"""Image-text embedding alignment, per dataset and per configuration.

For every dataset the frozen vision tower is embedded once and shared across the
configurations; only the text tower changes. Each configuration then yields the paired
cosine similarity between an image and its own description, which drives:

* ``fig_tsne_<dataset>``       -- t-SNE of the image and text clouds, one panel per config
* ``fig_alignment_<dataset>``  -- violin + mean bar of the paired cosine similarities
* ``table_alignment_<dataset>.tex``
* ``alignment_<dataset>.json`` -- the mean/median/std the cross-dataset figure reads
* ``fig_alignment_cross_dataset`` -- rendered by :mod:`capara.analysis.cross_dataset_alignment`

Usage:
    python -m capara.analysis.embedding_space \
        --checkpoints baseline=none cfg5=/runs/cfg5/final_model.pt \
        --datasets docci sharegpt4v --num_samples 10000
"""

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile

from capara.analysis.cross_dataset_alignment import plot_cross_dataset
from capara.analysis.models import checkpoint_path, free_cuda, parse_checkpoints
from capara.analysis.samples import (
    COCO_VAL_ANN,
    COCO_VAL_IMAGE_DIR,
    DOCCI_IMAGE_DIR,
    DOCCI_JSONL,
    FLICKR_IMAGE_DIR,
    SHAREGPT4V_IMAGE_ROOT,
    SHAREGPT4V_SAMPLE_JSON,
    Sample,
    load_coco_samples,
    load_docci_samples,
    load_flickr_samples,
    load_sharegpt4v_samples,
)
from capara.analysis.style import (
    CFG_ORDER_WITH_BASELINE,
    COLORS,
    PRETRAINED_LABELS_COMPACT,
    cfg_labels,
    dataset_slug,
    use_paper_style,
)
from capara.common.blip import encode_image_paths, encode_texts_batched, load_blip
from capara.common.paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS: dict[str, str] = {**PRETRAINED_LABELS_COMPACT, **cfg_labels(compact=True)}

IMG_COLOR = "#2166AC"
TXT_COLOR = "#B2182B"

#: Per-config embeddings and their paired cosine similarities, for one dataset.
ConfigEmbeddings = dict[str, dict[str, np.ndarray]]


def paired_cosine(img_embs: torch.Tensor, txt_embs: torch.Tensor) -> np.ndarray:
    """Cosine similarity between each image and its own description (both normalised)."""
    return (img_embs * txt_embs).sum(dim=-1).numpy()


def plot_tsne_grid(
    data: ConfigEmbeddings,
    fig_dir: Path,
    dataset: str,
    tsne_samples: int = 1500,
    seed: int = 42,
) -> None:
    """One t-SNE panel per config: image cloud, text cloud, and a few paired links."""
    from sklearn.manifold import TSNE

    tags = [tag for tag in CFG_ORDER_WITH_BASELINE if tag in data]
    n_cols = min(3, len(tags))
    n_rows = (len(tags) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for index, tag in enumerate(tags):
        ax = axes[index]
        img_embs = data[tag]["img_embs"]
        txt_embs = data[tag]["txt_embs"]

        n_total = img_embs.shape[0]
        if n_total > tsne_samples:
            chosen = np.random.RandomState(seed).choice(n_total, tsne_samples, replace=False)
            img_embs, txt_embs = img_embs[chosen], txt_embs[chosen]

        n_points = img_embs.shape[0]
        print(f"  t-SNE {tag} ({2 * n_points} pts)...")
        projected = TSNE(
            n_components=2,
            perplexity=30,
            random_state=seed,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
        ).fit_transform(np.vstack([img_embs, txt_embs]))
        img_xy, txt_xy = projected[:n_points], projected[n_points:]

        ax.scatter(
            img_xy[:, 0], img_xy[:, 1], c=IMG_COLOR, s=6, alpha=0.35, label="Image",
            edgecolors="none",
        )
        ax.scatter(
            txt_xy[:, 0], txt_xy[:, 1], c=TXT_COLOR, s=6, alpha=0.35, label="Text",
            edgecolors="none",
        )

        n_links = min(40, n_points)
        for i in np.random.RandomState(seed).choice(n_points, n_links, replace=False):
            ax.plot(
                [img_xy[i, 0], txt_xy[i, 0]],
                [img_xy[i, 1], txt_xy[i, 1]],
                color="#888888",
                linewidth=0.3,
                alpha=0.25,
            )

        mean_cos = float(np.mean(data[tag]["paired_cos"]))
        ax.set_title(f"{LABELS.get(tag, tag)} (cos={mean_cos:.3f})", fontsize=10)
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.tick_params(labelsize=7)
        if index == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.8, markerscale=1.5)

    for index in range(len(tags), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(f"t-SNE: image vs text embeddings ({dataset})", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    name = f"fig_tsne_{dataset_slug(dataset)}"
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


def plot_alignment(data: ConfigEmbeddings, fig_dir: Path, dataset: str) -> None:
    """Distribution (violin) and mean (bar) of the paired cosine similarity per config."""
    tags = [tag for tag in CFG_ORDER_WITH_BASELINE if tag in data]
    cosines = [data[tag]["paired_cos"] for tag in tags]
    labels = [LABELS.get(tag, tag) for tag in tags]
    colors = [COLORS.get(tag, "#333") for tag in tags]
    positions = range(len(tags))

    fig, (ax_violin, ax_bar) = plt.subplots(1, 2, figsize=(12, 4.5))

    parts = ax_violin.violinplot(
        cosines, positions=positions, showmeans=True, showmedians=True, showextrema=False
    )
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor(color)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1)

    ax_violin.set_xticks(list(positions))
    ax_violin.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax_violin.set_ylabel("Paired cosine similarity")
    ax_violin.set_title(f"(a) Alignment — {dataset}", fontsize=10)
    ax_violin.set_ylim(0, 1.05)
    ax_violin.grid(True, alpha=0.2)

    means = [float(np.mean(c)) for c in cosines]
    stds = [float(np.std(c)) for c in cosines]
    ax_bar.bar(positions, means, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax_bar.errorbar(positions, means, yerr=stds, fmt="none", ecolor="#444", capsize=3, capthick=1)
    for index, (mean, std) in enumerate(zip(means, stds, strict=True)):
        ax_bar.text(
            index,
            mean + std + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color=colors[index],
        )

    ax_bar.set_xticks(list(positions))
    ax_bar.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax_bar.set_ylabel("Mean paired cosine similarity")
    ax_bar.set_title(f"(b) Mean alignment — {dataset}", fontsize=10)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.grid(True, alpha=0.2)

    fig.tight_layout()

    name = f"fig_alignment_{dataset_slug(dataset)}"
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


def generate_latex_table(data: ConfigEmbeddings, out_path: Path, dataset: str) -> None:
    """Mean / median / std of the paired cosine similarity, best fine-tuned config in bold."""
    tags = [tag for tag in CFG_ORDER_WITH_BASELINE if tag in data]
    trained = [tag for tag in tags if tag != "baseline"]
    best = max(trained, key=lambda tag: np.mean(data[tag]["paired_cos"])) if trained else None

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Image-text alignment on " + dataset + r". Paired cosine similarity.}",
        r"\label{tab:align_" + dataset_slug(dataset) + "}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Mean} & \textbf{Median} & \textbf{Std} \\",
        r"\midrule",
    ]

    for tag in tags:
        cosines = data[tag]["paired_cos"]
        mean = f"{np.mean(cosines):.3f}"
        if tag == best:
            mean = r"\textbf{" + mean + "}"
        row = " & ".join(
            [LABELS.get(tag, tag), mean, f"{np.median(cosines):.3f}", f"{np.std(cosines):.3f}"]
        ) + r" \\"
        if tag == "baseline":
            row += "\n" + r"\midrule"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def run_dataset(
    dataset: str,
    samples: Sequence[Sample],
    checkpoints: Mapping[str, str],
    args: argparse.Namespace,
) -> ConfigEmbeddings:
    """Embed one dataset under every configuration and write its figures and tables."""
    print(f"\n{'#' * 70}\n# {dataset} ({len(samples)} samples)\n{'#' * 70}")

    texts = [sample["text"] for sample in samples]
    image_paths = [sample["image_path"] for sample in samples]

    print("\nEncoding images once (the vision tower is frozen)...")
    model, processor = load_blip(device=args.device)
    img_embs = encode_image_paths(model, processor, image_paths, args.device, args.batch_size)
    print(f"  {tuple(img_embs.shape)}")
    del model
    free_cuda()

    img_embs_np = img_embs.numpy()
    data: ConfigEmbeddings = OrderedDict()

    for tag, checkpoint in checkpoints.items():
        print(f"\n  {tag}")
        model, processor = load_blip(checkpoint=checkpoint_path(checkpoint), device=args.device)
        txt_embs = encode_texts_batched(
            model, processor, texts, args.device, args.max_length, args.batch_size
        )
        cosines = paired_cosine(img_embs, txt_embs)
        data[tag] = {
            "img_embs": img_embs_np,
            "txt_embs": txt_embs.numpy(),
            "paired_cos": cosines,
        }
        print(f"    cos: mean={np.mean(cosines):.4f}")
        del model
        free_cuda()

    del img_embs
    free_cuda()

    plot_tsne_grid(data, args.fig_dir, dataset, args.tsne_samples, args.seed)
    plot_alignment(data, args.fig_dir, dataset)
    generate_latex_table(
        data, args.table_dir / f"table_alignment_{dataset_slug(dataset)}.tex", dataset
    )

    summary = {
        tag: {
            "mean": float(np.mean(values["paired_cos"])),
            "median": float(np.median(values["paired_cos"])),
            "std": float(np.std(values["paired_cos"])),
        }
        for tag, values in data.items()
    }
    summary_path = args.json_dir / f"alignment_{dataset_slug(dataset)}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")

    return data


def load_datasets(args: argparse.Namespace) -> list[tuple[str, list[Sample]]]:
    """Load every requested dataset, in the order the figures expect."""
    loaders = {
        "sharegpt4v": lambda: (
            "ShareGPT4V",
            load_sharegpt4v_samples(
                args.sharegpt4v_json, args.sharegpt4v_image_root, args.num_samples, args.seed
            ),
        ),
        "docci": lambda: (
            "DOCCI",
            load_docci_samples(args.docci_jsonl, args.docci_image_dir, args.num_samples, args.seed),
        ),
        "coco": lambda: (
            "COCO",
            load_coco_samples(args.coco_ann, args.coco_image_dir, args.num_samples, args.seed),
        ),
        "flickr30k": lambda: (
            "Flickr30k",
            load_flickr_samples(args.flickr_image_dir, args.num_samples, args.seed),
        ),
    }

    datasets: list[tuple[str, list[Sample]]] = []
    for key in args.datasets:
        name, samples = loaders[key]()
        if samples:
            datasets.append((name, samples))
        else:
            print(f"  No samples for {name}; skipping.")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="tag=path pairs, e.g. baseline=none cfg5=/runs/cfg5/final_model.pt",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sharegpt4v", "docci", "coco", "flickr30k"],
        choices=["sharegpt4v", "docci", "coco", "flickr30k"],
    )
    parser.add_argument("--sharegpt4v_json", type=Path, default=SHAREGPT4V_SAMPLE_JSON)
    parser.add_argument("--sharegpt4v_image_root", type=Path, default=SHAREGPT4V_IMAGE_ROOT)
    parser.add_argument("--docci_jsonl", type=Path, default=DOCCI_JSONL)
    parser.add_argument("--docci_image_dir", type=Path, default=DOCCI_IMAGE_DIR)
    parser.add_argument("--coco_ann", type=Path, default=COCO_VAL_ANN)
    parser.add_argument("--coco_image_dir", type=Path, default=COCO_VAL_IMAGE_DIR)
    parser.add_argument("--flickr_image_dir", type=Path, default=FLICKR_IMAGE_DIR)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--tsne_samples", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--json_dir", type=Path, default=RESULTS_DIR / "embedding_space")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_paper_style()
    for directory in (args.fig_dir, args.table_dir, args.json_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoints(args.checkpoints)
    print(f"Configs: {list(checkpoints)}")

    datasets = load_datasets(args)
    if not datasets:
        raise SystemExit("No datasets could be loaded.")
    print(f"\nDatasets: {[name for name, _ in datasets]}")

    means: OrderedDict[str, dict[str, float]] = OrderedDict()
    for dataset, samples in datasets:
        data = run_dataset(dataset, samples, checkpoints, args)
        means[dataset] = {
            tag: float(np.mean(values["paired_cos"])) for tag, values in data.items()
        }

    if len(means) > 1:
        print("\nCross-dataset comparison...")
        plot_cross_dataset(means, args.fig_dir)

    print(f"\nDone. Figures in {args.fig_dir}, tables in {args.table_dir}, JSON in {args.json_dir}")


if __name__ == "__main__":
    main()
