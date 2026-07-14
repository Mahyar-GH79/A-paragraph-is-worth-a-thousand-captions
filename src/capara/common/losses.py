"""Contrastive objectives.

Both losses operate on L2-normalised embeddings in BLIP's 256-d ITC space and
use a fixed temperature (the pretrained logit scale is not learned here).

Positives are addressed through a ragged ``pos_index`` matrix rather than a fixed
``k``, so the single-caption, multi-positive and variable-positive cases are one
code path. With one positive per image the objective reduces exactly to the
symmetric CLIP InfoNCE.
"""


import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def _gather_masked(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``logits`` at ``index``, sending ``-1`` padding to ``-inf``."""
    mask = index >= 0
    gathered = logits.gather(1, index.clamp(min=0))
    return gathered.masked_fill(~mask, NEG_INF)


def _masked_logsumexp(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """log-sum-exp over each image's texts, ignoring ``-1`` padding."""
    return torch.logsumexp(_gather_masked(logits, index), dim=1)


def _masked_max(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Max over each image's texts, ignoring ``-1`` padding."""
    return _gather_masked(logits, index).max(dim=1).values


def multi_positive_infonce(
    img: torch.Tensor,
    txt: torch.Tensor,
    pos_index: torch.Tensor,
    text_owner: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE where an image may have several positive texts.

    Image-to-text sums the probability mass over all of an image's positives;
    text-to-image is a plain cross-entropy because each text has exactly one
    source image.

    Args:
        img: ``[B, D]`` normalised image embeddings.
        txt: ``[T, D]`` normalised text embeddings for the whole batch.
        pos_index: ``[B, P_max]`` columns of ``txt`` that are positive for each
            image, padded with ``-1``.
        text_owner: ``[T]`` image index each text belongs to.
    """
    logits_i2t = (img @ txt.t()) / temperature  # [B, T]
    lse_all = torch.logsumexp(logits_i2t, dim=1)
    lse_pos = _masked_logsumexp(logits_i2t, pos_index)
    loss_i2t = (lse_all - lse_pos).mean()

    logits_t2i = (txt @ img.t()) / temperature  # [T, B]
    loss_t2i = F.cross_entropy(logits_t2i, text_owner.to(logits_t2i.device))

    return 0.5 * (loss_i2t + loss_t2i)


def multi_positive_infonce_with_negatives(
    img: torch.Tensor,
    txt_pos: torch.Tensor,
    txt_neg: torch.Tensor,
    pos_index: torch.Tensor,
    neg_index: torch.Tensor,
    text_owner: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-positive InfoNCE in which hard negatives act only as distractors.

    Negatives extend the image-to-text denominator but are never a retrieval
    target, so text-to-image is computed over the positives alone.

    Args:
        txt_pos: ``[T_pos, D]`` positive text embeddings.
        txt_neg: ``[T_neg, D]`` hard-negative text embeddings.
        pos_index: ``[B, P_max]`` columns of ``txt_pos``, padded with ``-1``.
        neg_index: ``[B, N_max]`` columns of ``txt_neg``, padded with ``-1``.
        text_owner: ``[T_pos]`` image index each positive text belongs to.

    Returns:
        The loss, and the mean margin ``max(own positive) - max(own negative)``
        in logit units -- a diagnostic for whether the negatives are separable.
    """
    txt_all = torch.cat([txt_pos, txt_neg], dim=0)
    logits_all = (img @ txt_all.t()) / temperature  # [B, T_pos + T_neg]

    logits_pos = logits_all[:, : txt_pos.size(0)]
    logits_neg = logits_all[:, txt_pos.size(0) :]

    lse_all = torch.logsumexp(logits_all, dim=1)
    lse_pos = _masked_logsumexp(logits_pos, pos_index)
    loss_i2t = (lse_all - lse_pos).mean()

    logits_t2i = (txt_pos @ img.t()) / temperature  # [T_pos, B]
    loss_t2i = F.cross_entropy(logits_t2i, text_owner.to(logits_t2i.device))

    margin = (
        _masked_max(logits_pos, pos_index) - _masked_max(logits_neg, neg_index)
    ).mean()

    return 0.5 * (loss_i2t + loss_t2i), margin
