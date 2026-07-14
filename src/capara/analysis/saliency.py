"""Gradient-based saliency maps: which pixels drive the image-text similarity?

For each DOCCI sample and each model, the gradient of the image-text cosine similarity
with respect to the input pixels is taken, reduced over the colour channels, smoothed and
overlaid on the image. The text tower differs per model, so the maps show what each
model's notion of the description attends to.

Outputs:
    ``saliency_maps/sample_<i>/{original,<model>_saliency}.jpg``
    ``fig_saliency_combined`` -- rows are samples, columns are original + the five models
    ``saliency_figures.tex``

Usage:
    python -m capara.analysis.saliency --cfg5_ckpt /runs/cfg5/final_model.pt [--num_samples 4]
"""

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from capara.analysis.models import (
    CLIP_MODEL_PATCH16,
    LONGCLIP_B_CKPT,
    LONGCLIP_L_CKPT,
    LONGCLIP_REPO,
    load_clip,
    load_longclip,
)
from capara.analysis.samples import DOCCI_IMAGE_DIR, DOCCI_JSONL, Sample, load_docci_records
from capara.analysis.style import (
    MODEL_LABELS,
    MODEL_TAGS,
    escape_latex,
    use_paper_style,
)
from capara.common.blip import load_blip
from capara.common.paths import FIGURES_DIR, TABLES_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

#: Descriptions this long are detailed enough to be interesting, short enough to display.
WORD_COUNT_RANGE = (40, 150)

DISPLAY_SIZE = (224, 224)


def _pixel_saliency(pixel_values: torch.Tensor, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``d(similarity)/d(pixels)`` to a 2-D map, and resize the image to match."""
    gradient = pixel_values.grad.detach().cpu()[0]  # [3, H, W]
    saliency = gradient.abs().max(dim=0).values.numpy()  # [H, W]
    resized = np.array(image.resize((saliency.shape[1], saliency.shape[0])))
    return saliency, resized


def saliency_blip(model, processor, image_path: str, text: str, device: str, max_length: int = 128):
    """Pixel saliency of the BLIP image-text similarity."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        processor(images=[image], return_tensors="pt")["pixel_values"].to(device).requires_grad_(True)
    )
    tokens = processor.tokenizer(
        [text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.enable_grad():
        vision_out = model.vision_model(pixel_values=pixel_values, return_dict=True)
        img_proj = F.normalize(model.vision_proj(vision_out.pooler_output), dim=-1)

        text_out = model.text_encoder(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True
        )
        pooled = getattr(text_out, "pooler_output", None)
        if pooled is None:
            pooled = text_out.last_hidden_state[:, 0, :]
        txt_proj = F.normalize(model.text_proj(pooled), dim=-1)

        (img_proj * txt_proj).sum().backward()

    return _pixel_saliency(pixel_values, image)


def saliency_clip(model, processor, image_path: str, text: str, device: str):
    """Pixel saliency of the CLIP image-text similarity."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        processor(images=[image], return_tensors="pt")["pixel_values"].to(device).requires_grad_(True)
    )
    tokens = processor.tokenizer(
        [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.enable_grad():
        vision_out = model.vision_model(pixel_values=pixel_values, return_dict=True)
        img_proj = F.normalize(model.visual_projection(vision_out.pooler_output), dim=-1)

        text_out = model.text_model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True
        )
        txt_proj = F.normalize(model.text_projection(text_out.pooler_output), dim=-1)

        (img_proj * txt_proj).sum().backward()

    return _pixel_saliency(pixel_values, image)


def saliency_longclip(model, preprocess, longclip, image_path: str, text: str, device: str):
    """Pixel saliency of the Long-CLIP image-text similarity."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = preprocess(image).unsqueeze(0).to(device).requires_grad_(True)
    tokens = longclip.tokenize([text], truncate=True).to(device)

    with torch.enable_grad():
        img_proj = F.normalize(model.encode_image(pixel_values).float(), dim=-1)
        txt_proj = F.normalize(model.encode_text(tokens).float(), dim=-1)
        (img_proj * txt_proj).sum().backward()

    return _pixel_saliency(pixel_values, image)


def heatmap_overlay(
    saliency: np.ndarray,
    image: np.ndarray,
    sigma: float = 11,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth, normalise and blend a saliency map over the image with a jet colormap."""
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(saliency, sigma=sigma)
    low, high = smoothed.min(), smoothed.max()
    normalised = (smoothed - low) / (high - low) if high - low > 1e-8 else np.zeros_like(smoothed)

    height, width = image.shape[:2]
    if normalised.shape != (height, width):
        resized = Image.fromarray((normalised * 255).astype(np.uint8)).resize(
            (width, height), Image.BILINEAR
        )
        normalised = np.array(resized).astype(np.float32) / 255.0

    heatmap = cm.jet(normalised)[:, :, :3]
    blended = image.astype(np.float32) / 255.0 * (1 - alpha) + heatmap * alpha
    return np.clip(blended * 255, 0, 255).astype(np.uint8), normalised


def select_samples(
    records: Sequence[Sample],
    num_samples: int,
    metadata_path: Path | None = None,
    seed: int = 42,
) -> list[int]:
    """Pick the samples to visualise.

    With ``metadata_path`` the qualitative figure's samples are reused, so the two figures
    show the same images. Otherwise, moderately long descriptions are sampled from across
    the dataset.
    """
    if metadata_path and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text())
        indices = [entry["query_idx"] for entry in metadata[:num_samples]]
        print(f"Using {len(indices)} samples from {metadata_path}")
        return indices

    low, high = WORD_COUNT_RANGE
    candidates = [
        index
        for index, record in enumerate(records)
        if low <= len(record["text"].split()) <= high
    ]
    random.Random(seed).shuffle(candidates)

    if len(candidates) > num_samples * 10:
        chunk = len(candidates) // num_samples
        selected = [candidates[i * chunk] for i in range(num_samples) if candidates[i * chunk :]]
        print(f"Selected {len(selected)} diverse samples from {len(candidates)} candidates")
        return selected[:num_samples]

    print(f"Selected {min(num_samples, len(candidates))} samples")
    return candidates[:num_samples]


def _load_models(args: argparse.Namespace) -> dict[str, tuple]:
    """``{tag: (kind, model, processor_or_preprocess)}`` for all five models."""
    models: dict[str, tuple] = {}

    print("\nLoading BLIP0...")
    blip0, blip0_processor = load_blip(device=args.device)
    models["blip0"] = ("blip", blip0, blip0_processor)

    print("Loading CFG5...")
    cfg5, cfg5_processor = load_blip(checkpoint=args.cfg5_ckpt, device=args.device)
    models["cfg5"] = ("blip", cfg5, cfg5_processor)

    print("Loading CLIP...")
    clip, clip_processor = load_clip(CLIP_MODEL_PATCH16, args.device)
    models["clip"] = ("clip", clip, clip_processor)

    for tag, checkpoint in (("longclip_b", args.longclip_b_ckpt), ("longclip_l", args.longclip_l_ckpt)):
        print(f"Loading {MODEL_LABELS[tag]}...")
        longclip, model, preprocess = load_longclip(checkpoint, args.longclip_repo, args.device)
        models[tag] = ("longclip", model, preprocess, longclip)

    return models


def _compute_saliency(models: dict[str, tuple], tag: str, image_path: str, text: str,
                      device: str, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    entry = models[tag]
    kind, model = entry[0], entry[1]
    model.eval()
    model.zero_grad()

    if kind == "blip":
        return saliency_blip(model, entry[2], image_path, text, device, max_length)
    if kind == "clip":
        return saliency_clip(model, entry[2], image_path, text, device)
    return saliency_longclip(model, entry[2], entry[3], image_path, text, device)


def plot_combined(
    overlays: dict[str, list[np.ndarray]],
    originals: Sequence[np.ndarray],
    texts: Sequence[str],
    fig_dir: Path,
) -> None:
    """One row per sample: the description, then the image and its five saliency overlays."""
    n_samples = len(originals)
    n_cols = len(MODEL_TAGS) + 1

    row_height = 2.8
    text_height = 1.2
    total_height = n_samples * (row_height + text_height) + 1.5
    col_width = 2.5
    fig_width = col_width * n_cols + 1.5

    fig = plt.figure(figsize=(fig_width, total_height))

    for row in range(n_samples):
        y_text = 1.0 - (row * (row_height + text_height) + 0.3) / total_height
        fig.text(0.08, y_text, f"Sample {row + 1}: ", fontsize=9, fontweight="bold", va="top",
                 transform=fig.transFigure)
        fig.text(
            0.17,
            y_text,
            texts[row],
            fontsize=7,
            va="top",
            wrap=True,
            transform=fig.transFigure,
            fontstyle="italic",
            bbox=dict(boxstyle="square,pad=0.1", facecolor="none", edgecolor="none"),
            clip_on=False,
        )

        for col in range(n_cols):
            ax = fig.add_axes(
                [
                    (col * col_width + 1.2) / fig_width,
                    (total_height - (row * (row_height + text_height) + text_height + row_height))
                    / total_height,
                    (col_width - 0.3) / fig_width,
                    (row_height - 0.4) / total_height,
                ]
            )

            if col == 0:
                ax.imshow(originals[row])
                title = "Original"
            else:
                tag = MODEL_TAGS[col - 1]
                ax.imshow(overlays[tag][row])
                title = MODEL_LABELS[tag]

            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(fig_dir / "fig_saliency_combined.pdf")
    fig.savefig(fig_dir / "fig_saliency_combined.png")
    plt.close(fig)
    print("Saved fig_saliency_combined.pdf/.png")


def generate_latex(texts: Sequence[str], out_path: Path, image_prefix: str) -> None:
    """A single ``figure*`` with the full description above each row of overlays."""
    lines = [r"\begin{figure*}[t]", r"    \centering"]

    for index, text in enumerate(texts):
        lines.append(
            r"    \parbox{0.97\textwidth}{\scriptsize \textbf{Sample "
            + str(index + 1)
            + r":} "
            + escape_latex(text)
            + r"}"
        )
        lines += [r"    \vspace{2pt}", "", r"    \setlength{\tabcolsep}{1pt}",
                  r"    \begin{tabular}{cccccc}"]

        if index == 0:
            header = r"        \scriptsize \textbf{Original} & " + " & ".join(
                r"\scriptsize \textbf{" + MODEL_LABELS[tag] + "}" for tag in MODEL_TAGS
            ) + r" \\[2pt]"
            lines.append(header)

        cells = [f"\\includegraphics[width=0.155\\textwidth]{{{image_prefix}/sample_{index}/original.jpg}}"]
        cells += [
            f"\\includegraphics[width=0.155\\textwidth]{{{image_prefix}/sample_{index}/{tag}_saliency.jpg}}"
            for tag in MODEL_TAGS
        ]
        lines.append("        " + " &\n        ".join(cells) + r" \\")
        lines.append(r"    \end{tabular}")
        if index < len(texts) - 1:
            lines.append(r"    \vspace{6pt}")
        lines.append("")

    lines += [
        r"    \caption{Gradient-based saliency maps on DOCCI. For each sample, the full description "
        r"query is shown above, followed by the original image and heatmap overlays for five models. "
        r"Warm regions (red/yellow) indicate pixels that most influence the text-image similarity score. "
        r"CFG5 (ours) distributes attention across more diverse image regions, consistent with its "
        r"paragraph-trained text encoder capturing fine-grained attributes beyond the dominant object.}",
        r"    \label{fig:saliency_combined}",
        r"\end{figure*}",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cfg5_ckpt", type=Path, required=True)
    parser.add_argument("--longclip_repo", type=Path, default=LONGCLIP_REPO)
    parser.add_argument("--longclip_b_ckpt", type=Path, default=LONGCLIP_B_CKPT)
    parser.add_argument("--longclip_l_ckpt", type=Path, default=LONGCLIP_L_CKPT)
    parser.add_argument("--docci_jsonl", type=Path, default=DOCCI_JSONL)
    parser.add_argument("--docci_image_dir", type=Path, default=DOCCI_IMAGE_DIR)
    parser.add_argument(
        "--qualitative_metadata",
        type=Path,
        default=None,
        help="metadata.json from `qualitative docci`; reuse its samples instead of sampling",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fig_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLES_DIR)
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=FIGURES_DIR / "saliency_maps",
        help="Where the per-sample overlays are written",
    )
    parser.add_argument(
        "--latex_image_prefix",
        type=str,
        default="figures/saliency_maps",
        help="Path the generated LaTeX uses to include the overlays",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_paper_style(**{"savefig.pad_inches": 0.02})
    for directory in (args.fig_dir, args.table_dir, args.image_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = load_docci_records(args.docci_jsonl, args.docci_image_dir)
    indices = select_samples(records, args.num_samples, args.qualitative_metadata, args.seed)

    models = _load_models(args)

    overlays: dict[str, list[np.ndarray]] = {tag: [] for tag in MODEL_TAGS}
    originals: list[np.ndarray] = []
    texts: list[str] = []

    for sample_idx, query in enumerate(indices):
        record = records[query]
        texts.append(record["text"])
        original = np.array(Image.open(record["image_path"]).convert("RGB").resize(DISPLAY_SIZE))
        originals.append(original)

        print(f"\nSample {sample_idx} (idx={query}): {record['text'][:80]}...")
        sample_dir = args.image_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original).save(sample_dir / "original.jpg", quality=95)
        (sample_dir / "query_text.txt").write_text(record["text"], encoding="utf-8")

        for tag in MODEL_TAGS:
            print(f"  {tag}...", end=" ", flush=True)
            saliency, image = _compute_saliency(
                models, tag, record["image_path"], record["text"], args.device, args.max_length
            )
            overlay, _ = heatmap_overlay(saliency, image)
            overlays[tag].append(overlay)
            Image.fromarray(overlay).save(sample_dir / f"{tag}_saliency.jpg", quality=95)
            print("done")

    print("\nGenerating combined figure...")
    plot_combined(overlays, originals, texts, args.fig_dir)
    generate_latex(texts, args.table_dir / "saliency_figures.tex", args.latex_image_prefix)

    print(f"\nOverlays in {args.image_dir}, figure in {args.fig_dir}")


if __name__ == "__main__":
    main()
