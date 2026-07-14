"""Model loading, embedding and ranking helpers shared by the analysis modules.

BLIP goes through :mod:`capara.common.blip`, so the analyses embed exactly the way
training and evaluation do. CLIP and Long-CLIP only appear as baselines in the figures,
so their (much thinner) wrappers live here.
"""

import sys
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from capara.common.paths import DATA_ROOT, TRAIN_RUNS_DIR

PathLike = str | Path

#: The Long-CLIP baselines are used from the authors' repository, which is not a package.
LONGCLIP_REPO = DATA_ROOT / "long_clip" / "repo"
LONGCLIP_B_CKPT = LONGCLIP_REPO / "checkpoints" / "longclip-B.pt"
LONGCLIP_L_CKPT = LONGCLIP_REPO / "checkpoints" / "longclip-L.pt"

#: The DOCCI comparison figures (qualitative, saliency, t-SNE grid) benchmark against
#: CLIP ViT-B/16; the truncation study uses the ViT-B/32 baseline from ``common.paths``.
CLIP_MODEL_PATCH16 = "openai/clip-vit-base-patch16"

#: Long-CLIP's tokenizer is fixed at 248 context tokens; CLIP's at 77.
CLIP_MAX_LENGTH = 77


def parse_checkpoints(pairs: Sequence[str]) -> "OrderedDict[str, str]":
    """Parse ``tag=path`` CLI arguments, e.g. ``cfg5=/runs/cfg5/final_model.pt``.

    ``tag=none`` denotes the pretrained model.
    """
    parsed: OrderedDict[str, str] = OrderedDict()
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected tag=path, got {pair!r}")
        tag, path = pair.split("=", 1)
        parsed[tag] = path
    return parsed


def checkpoint_path(value: str | None) -> str | None:
    """``None`` for the pretrained model, otherwise the checkpoint path."""
    if value is None or value.lower() == "none":
        return None
    return value


def find_run_checkpoint(cfg: str, train_root: PathLike = TRAIN_RUNS_DIR) -> Path | None:
    """Locate ``final_model.pt`` for a config tag, e.g. ``cfg5`` -> the newest cfg5 run."""
    runs = sorted(
        run
        for run in Path(train_root).glob(f"{cfg}_*")
        if run.is_dir() and (run / "final_model.pt").is_file()
    )
    return runs[-1] / "final_model.pt" if runs else None


# --------------------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------------------


def load_clip(model_name: str, device: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Load a pretrained CLIP retrieval baseline."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return model.to(device).eval(), processor


def _autocast(device: str):
    return torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16)


@torch.no_grad()
def clip_encode_image_paths(
    model: CLIPModel,
    processor: CLIPProcessor,
    paths: Sequence[PathLike],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed images from disk into CLIP's normalised joint space."""
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="  CLIP images", leave=False):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with _autocast(device):
            vision_out = model.vision_model(pixel_values=pixel_values, return_dict=True)
            projected = model.visual_projection(vision_out.pooler_output)
        chunks.append(F.normalize(projected.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def clip_encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: Sequence[str],
    device: str,
    max_length: int = CLIP_MAX_LENGTH,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed texts into CLIP's normalised joint space, truncating at ``max_length``."""
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  CLIP texts", leave=False):
        tokens = processor.tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with _autocast(device):
            text_out = model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                return_dict=True,
            )
            projected = model.text_projection(text_out.pooler_output)
        chunks.append(F.normalize(projected.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------------------
# Long-CLIP
# --------------------------------------------------------------------------------------


def load_longclip(checkpoint: PathLike, repo: PathLike = LONGCLIP_REPO, device: str = "cpu"):
    """Load a Long-CLIP checkpoint from the upstream repository.

    Returns ``(longclip_module, model, preprocess)``; the module is needed for its
    tokenizer.
    """
    repo = str(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model import longclip  # provided by the Long-CLIP repository

    model, preprocess = longclip.load(str(checkpoint), device=device)
    return longclip, model.eval(), preprocess


@torch.no_grad()
def longclip_encode_image_paths(
    model,
    preprocess,
    paths: Sequence[PathLike],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed images from disk into Long-CLIP's normalised joint space."""
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="  Long-CLIP images", leave=False):
        batch = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in paths[start : start + batch_size]]
        ).to(device)
        with _autocast(device):
            features = model.encode_image(batch)
        chunks.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def longclip_encode_texts(
    longclip_module,
    model,
    texts: Sequence[str],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed texts into Long-CLIP's normalised joint space."""
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  Long-CLIP texts", leave=False):
        tokens = longclip_module.tokenize(
            list(texts[start : start + batch_size]), truncate=True
        ).to(device)
        with _autocast(device):
            features = model.encode_text(tokens)
        chunks.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------------------


@torch.no_grad()
def ground_truth_ranks(
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    """Text-to-image rank of the paired image for every text query (0 = retrieved first)."""
    n_texts = txt_embs.size(0)
    ranks = np.zeros(n_texts, dtype=int)
    for start in tqdm(range(0, n_texts, batch_size), desc="  ranks", leave=False):
        end = min(start + batch_size, n_texts)
        sim = txt_embs[start:end] @ img_embs.t()
        order = sim.argsort(dim=1, descending=True)
        targets = torch.arange(start, end).unsqueeze(1)
        ranks[start:end] = (order == targets).int().argmax(dim=1).numpy()
    return ranks


@torch.no_grad()
def top_k_images(img_embs: torch.Tensor, txt_emb: torch.Tensor, k: int) -> np.ndarray:
    """Indices of the ``k`` images a single text query retrieves first."""
    sim = txt_emb @ img_embs.t()
    return sim.argsort(descending=True)[:k].numpy()


def free_cuda(*tensors_or_models: Iterable) -> None:
    """Drop references and empty the CUDA cache (no-op on CPU)."""
    del tensors_or_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
