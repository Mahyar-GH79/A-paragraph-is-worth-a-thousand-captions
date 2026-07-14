"""4x3 t-SNE grid: four benchmarks against three models.

Rows are Flickr30k, COCO, ShareGPT4V and DOCCI; columns are pretrained BLIP, pretrained
CLIP and the paragraph fine-tuned BLIP (CFG5). Each panel shows the image and text clouds
of the same pairs, with a sample of the pairs joined, and reports the mean paired cosine
similarity. CFG5 reuses BLIP's image embeddings: its vision tower is frozen.

Figure: ``fig_tsne_full_grid`` (PDF and PNG).

Usage:
    python -m capara.analysis.tsne_grid --cfg5_ckpt /runs/cfg5/final_model.pt
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile
from sklearn.manifold import TSNE

from capara.analysis.models import (
    CLIP_MODEL_PATCH16,
    clip_encode_image_paths,
    clip_encode_texts,
    free_cuda,
    load_clip,
)
from capara.analysis.samples import (
    COCO_TRAIN_ANN,
    COCO_TRAIN_IMAGE_DIR,
    DOCCI_IMAGE_DIR,
    DOCCI_JSONL,
    FLICKR_CAPTIONS_CSV,
    FLICKR_IMAGE_DIR,
    SHAREGPT4V_IMAGE_ROOT,
    SHAREGPT4V_SAMPLE_JSON,
    Sample,
    load_coco_samples,
    load_docci_samples,
    load_flickr_samples,
    load_sharegpt4v_samples,
)
from capara.analysis.style import DATASET_ORDER, use_paper_style
from capara.common.blip import encode_image_paths, encode_texts_batched, load_blip
from capara.common.paths import FIGURES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_COLOR = "#4E79A7"
TXT_COLOR = "#E15759"
LINE_COLOR = "#AAAAAA"

MODEL_TAGS = ["blip0", "clip", "cfg5"]
MODEL_LABELS = ["BLIP$_0$", "CLIP$_0$", "Paragraph fine-tuned BLIP"]

#: (image projection, text projection, mean paired cosine similarity)
Projection = tuple[np.ndarray, np.ndarray, float]


def compute_tsne(
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    num_samples: int,
    seed: int = 42,
) -> Projection:
    """t-SNE of a common subsample of the image and text embeddings."""
    n_total = img_embs.size(0)
    chosen = np.sort(
        np.random.RandomState(seed).choice(n_total, min(num_samples, n_total), replace=False)
    )

    images = img_embs[chosen].numpy()
    texts = txt_embs[chosen].numpy()
    mean_cos = float((images * texts).sum(axis=1).mean())

    projected = TSNE(
        n_components=2,
        perplexity=30,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
    ).fit_transform(np.concatenate([images, texts], axis=0))

    return projected[: len(chosen)], projected[len(chosen) :], mean_cos


def plot_panel(
    ax: plt.Axes,
    img_proj: np.ndarray,
    txt_proj: np.ndarray,
    n_lines: int = 40,
    seed: int = 42,
) -> None:
    """One t-SNE panel: both clouds, plus a sample of the image-text links."""
    n_points = len(img_proj)
    for index in np.random.RandomState(seed).choice(n_points, min(n_lines, n_points), replace=False):
        ax.plot(
            [img_proj[index, 0], txt_proj[index, 0]],
            [img_proj[index, 1], txt_proj[index, 1]],
            color=LINE_COLOR,
            linewidth=0.4,
            alpha=0.35,
            zorder=1,
        )

    for points, color in ((img_proj, IMG_COLOR), (txt_proj, TXT_COLOR)):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=color,
            s=12,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.15,
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")


def load_datasets(args: argparse.Namespace) -> dict[str, list[Sample]]:
    """Load every benchmark, over-sampling to absorb any unreadable images."""
    load_n = args.num_samples * 2
    print("Loading datasets...")

    datasets = {
        "Flickr30k": load_flickr_samples(
            args.flickr_images_dir, load_n, args.seed, args.flickr_captions_path
        ),
        "COCO": load_coco_samples(args.coco_ann, args.coco_image_dir, load_n, args.seed),
        "ShareGPT4V": load_sharegpt4v_samples(
            args.sharegpt4v_json, args.sharegpt4v_image_root, load_n, args.seed
        ),
        "DOCCI": load_docci_samples(args.docci_jsonl, args.docci_image_dir, load_n, args.seed),
    }
    for name, samples in datasets.items():
        print(f"  {name}: {len(samples)} samples")
    return datasets


def embed_dataset(
    samples: Sequence[Sample],
    models: dict[str, tuple],
    args: argparse.Namespace,
) -> dict[str, Projection]:
    """t-SNE projections of one dataset under all three models."""
    paths = [sample["image_path"] for sample in samples]
    texts = [sample["text"] for sample in samples]

    blip, blip_processor = models["blip0"]
    clip, clip_processor = models["clip"]
    cfg5, cfg5_processor = models["cfg5"]

    print("  Encoding BLIP0...")
    blip_img = encode_image_paths(blip, blip_processor, paths, args.device)
    blip_txt = encode_texts_batched(blip, blip_processor, texts, args.device, args.max_length)

    print("  Encoding CLIP0...")
    clip_img = clip_encode_image_paths(clip, clip_processor, paths, args.device)
    clip_txt = clip_encode_texts(clip, clip_processor, texts, args.device)

    print("  Encoding CFG5...")
    # Frozen vision tower: CFG5's image embeddings are BLIP0's.
    cfg5_txt = encode_texts_batched(cfg5, cfg5_processor, texts, args.device, args.max_length)

    embeddings = {
        "blip0": (blip_img, blip_txt),
        "clip": (clip_img, clip_txt),
        "cfg5": (blip_img, cfg5_txt),
    }

    projections: dict[str, Projection] = {}
    for tag in MODEL_TAGS:
        print(f"  t-SNE for {tag}...")
        img_embs, txt_embs = embeddings[tag]
        projections[tag] = compute_tsne(img_embs, txt_embs, args.num_samples, args.seed)

    del blip_img, blip_txt, clip_img, clip_txt, cfg5_txt
    free_cuda()
    return projections


def plot_grid(projections: dict[str, dict[str, Projection]], fig_dir: Path, seed: int) -> None:
    """The 4x3 grid, with a shared legend and the mean cosine similarity per panel."""
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))

    for row, dataset in enumerate(DATASET_ORDER):
        for col, tag in enumerate(MODEL_TAGS):
            ax = axes[row, col]
            img_proj, txt_proj, mean_cos = projections[dataset][tag]
            plot_panel(ax, img_proj, txt_proj, n_lines=40, seed=seed)

            ax.set_title(f"cos={mean_cos:.3f}", fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=12, fontweight="bold", labelpad=10)

    # The top row carries the model names as well as its own cosine similarity.
    for col, (tag, label) in enumerate(zip(MODEL_TAGS, MODEL_LABELS, strict=True)):
        mean_cos = projections[DATASET_ORDER[0]][tag][2]
        axes[0, col].set_title(f"{label}\ncos={mean_cos:.3f}", fontsize=10, fontweight="bold", pad=6)

    handles = [
        mlines.Line2D([], [], color=IMG_COLOR, marker="o", linestyle="None", markersize=6,
                      label="Image embeddings", alpha=0.7),
        mlines.Line2D([], [], color=TXT_COLOR, marker="o", linestyle="None", markersize=6,
                      label="Text embeddings", alpha=0.7),
        mlines.Line2D([], [], color=LINE_COLOR, linewidth=1, alpha=0.5, label="Paired connections"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        framealpha=0.95,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.4,
        columnspacing=1.5,
    )

    fig.suptitle(
        "t-SNE of image and text embeddings across datasets and models",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout(rect=[0.02, 0.025, 1, 0.995])

    fig.savefig(fig_dir / "fig_tsne_full_grid.pdf")
    fig.savefig(fig_dir / "fig_tsne_full_grid.png")
    plt.close(fig)
    print(f"\nSaved {fig_dir / 'fig_tsne_full_grid.pdf'}")


def print_summary(projections: dict[str, dict[str, Projection]]) -> None:
    """Mean paired cosine similarity per dataset and model."""
    print("\nCosine similarity summary:")
    print(f"{'Dataset':<12s} {'BLIP0':>10s} {'CLIP0':>10s} {'CFG5':>10s}")
    print("-" * 45)
    for dataset in DATASET_ORDER:
        values = [f"{projections[dataset][tag][2]:.4f}" for tag in MODEL_TAGS]
        print(f"{dataset:<12s} {values[0]:>10s} {values[1]:>10s} {values[2]:>10s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cfg5_ckpt", type=Path, required=True)
    parser.add_argument("--flickr_images_dir", type=Path, default=FLICKR_IMAGE_DIR)
    parser.add_argument("--flickr_captions_path", type=Path, default=FLICKR_CAPTIONS_CSV)
    parser.add_argument("--coco_ann", type=Path, default=COCO_TRAIN_ANN)
    parser.add_argument("--coco_image_dir", type=Path, default=COCO_TRAIN_IMAGE_DIR)
    parser.add_argument("--sharegpt4v_json", type=Path, default=SHAREGPT4V_SAMPLE_JSON)
    parser.add_argument("--sharegpt4v_image_root", type=Path, default=SHAREGPT4V_IMAGE_ROOT)
    parser.add_argument("--docci_jsonl", type=Path, default=DOCCI_JSONL)
    parser.add_argument("--docci_image_dir", type=Path, default=DOCCI_IMAGE_DIR)
    parser.add_argument("--num_samples", type=int, default=1500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_paper_style(**{"figure.dpi": 600, "savefig.dpi": 600})
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args)

    print("\nLoading models...")
    models = {
        "blip0": load_blip(device=args.device),
        "clip": load_clip(CLIP_MODEL_PATCH16, args.device),
        "cfg5": load_blip(checkpoint=args.cfg5_ckpt, device=args.device),
    }

    projections: dict[str, dict[str, Projection]] = {}
    for dataset in DATASET_ORDER:
        print(f"\n{'=' * 50}\nDataset: {dataset}\n{'=' * 50}")
        projections[dataset] = embed_dataset(datasets[dataset], models, args)

    del models
    free_cuda()

    print("\n=== Plotting 4x3 grid ===")
    plot_grid(projections, args.fig_dir, args.seed)
    print_summary(projections)

    print(f"\nDone. Figure in {args.fig_dir}")


if __name__ == "__main__":
    main()
