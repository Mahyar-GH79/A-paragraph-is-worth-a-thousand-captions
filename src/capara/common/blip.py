"""BLIP model loading and embedding.

Both towers are projected into BLIP's 256-d ITC space by hand rather than through
``get_{image,text}_features``, so that training, embedding-shard construction and
evaluation all use provably the same code path.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor

from .paths import BLIP_MODEL


def load_blip(
    model_name: str = BLIP_MODEL,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> tuple[BlipForImageTextRetrieval, BlipProcessor]:
    """Load BLIP, optionally overlaying a fine-tuned text tower.

    Raises if ``checkpoint`` provides no parameter that the model recognises --
    ``load_state_dict(strict=False)`` would otherwise silently evaluate the
    pretrained baseline and report it as a fine-tuned result.
    """
    processor = BlipProcessor.from_pretrained(model_name, use_fast=False)
    model = BlipForImageTextRetrieval.from_pretrained(model_name)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("model_state", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"Unrecognised checkpoint format: {checkpoint}")

        incompatible = model.load_state_dict(state, strict=False)
        matched = len(state) - len(incompatible.unexpected_keys)
        if matched == 0:
            raise ValueError(
                f"No parameter in {checkpoint} matched the model. "
                "The checkpoint is for a different architecture."
            )

    return model.to(device).eval(), processor


def freeze_vision_tower(model: BlipForImageTextRetrieval) -> None:
    """Train the text tower only: freeze the vision encoder, its projection, and the ITM head."""
    for module_name in ("vision_model", "vision_proj", "itm_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False

    for module_name in ("text_encoder", "text_proj", "text_projection"):
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def _text_projection(model: BlipForImageTextRetrieval) -> torch.nn.Module:
    for name in ("text_proj", "text_projection"):
        proj = getattr(model, name, None)
        if proj is not None:
            return proj
    raise AttributeError("BLIP model exposes neither text_proj nor text_projection")


def encode_texts(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    texts: Sequence[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Embed texts into the normalised 256-d ITC space. Differentiable."""
    tokens = processor.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    out = model.text_encoder(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True,
    )

    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0, :]  # CLS

    return F.normalize(_text_projection(model)(pooled), dim=-1)


@torch.no_grad()
def encode_images(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    images: Iterable[Image.Image],
    device: str,
) -> torch.Tensor:
    """Embed PIL images into the normalised 256-d ITC space."""
    inputs = processor(images=list(images), return_tensors="pt").to(device)
    pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    return F.normalize(model.vision_proj(pooled), dim=-1)


@torch.no_grad()
def encode_image_paths(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    paths: Sequence[str | Path],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Embed images from disk in batches."""
    from tqdm import tqdm

    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="images", leave=False):
        batch = [
            Image.open(p).convert("RGB") for p in paths[start : start + batch_size]
        ]
        chunks.append(encode_images(model, processor, batch, device).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 256)


@torch.no_grad()
def encode_texts_batched(
    model: BlipForImageTextRetrieval,
    processor: BlipProcessor,
    texts: Sequence[str],
    device: str,
    max_length: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """Embed texts in batches, without gradients."""
    from tqdm import tqdm

    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="texts", leave=False):
        batch = texts[start : start + batch_size]
        chunks.append(encode_texts(model, processor, batch, device, max_length).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 256)
