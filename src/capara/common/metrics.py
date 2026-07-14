"""Recall@K for image-text retrieval.

Every ground-truth regime in this project -- one text per image (DOCCI,
ShareGPT4V), several captions per image (COCO, Flickr30k), and a fixed bank of
k positives per image (multi-positive validation) -- is the same computation
over a text-to-image ownership map, so there is a single implementation.

Scoring is chunked over both axes: the full similarity matrix is never
materialised. COCO train2017 alone (118k images x 592k captions) would need
280 GB as a dense fp32 matrix, so only the running top-K per query is kept.
"""

from collections.abc import Sequence

import torch

DEFAULT_KS: Sequence[int] = (1, 5, 10)

#: Queries and gallery entries scored per block. Peak memory is
#: ``query_chunk * gallery_chunk`` scores.
DEFAULT_QUERY_CHUNK = 1024
DEFAULT_GALLERY_CHUNK = 4096


@torch.no_grad()
def _topk_indices(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k: int,
    device: torch.device,
    query_chunk: int,
    gallery_chunk: int,
) -> torch.Tensor:
    """Indices of the ``k`` highest-scoring gallery entries per query, best first.

    Returns ``[N_query, min(k, N_gallery)]`` on the CPU.
    """
    n_query, n_gallery = query.size(0), gallery.size(0)
    k = min(k, n_gallery)
    top = torch.empty((n_query, k), dtype=torch.long)

    for q_start in range(0, n_query, query_chunk):
        q_block = query[q_start : q_start + query_chunk].to(device)
        best_scores: torch.Tensor | None = None
        best_indices: torch.Tensor | None = None

        for g_start in range(0, n_gallery, gallery_chunk):
            g_block = gallery[g_start : g_start + gallery_chunk].to(device)
            scores = q_block @ g_block.t()

            block_scores, block_positions = scores.topk(
                min(k, scores.size(1)), dim=1
            )
            block_indices = block_positions + g_start

            if best_scores is None:
                best_scores, best_indices = block_scores, block_indices
                continue

            merged_scores = torch.cat([best_scores, block_scores], dim=1)
            merged_indices = torch.cat([best_indices, block_indices], dim=1)
            best_scores, kept = merged_scores.topk(k, dim=1)
            best_indices = torch.gather(merged_indices, 1, kept)

        assert best_indices is not None  # n_gallery > 0 guarantees one block
        top[q_start : q_start + q_block.size(0)] = best_indices.cpu()

    return top


@torch.no_grad()
def recall_at_k(
    img: torch.Tensor,
    txt: torch.Tensor,
    text_owner: torch.Tensor,
    ks: Sequence[int] = DEFAULT_KS,
    device: str | torch.device | None = None,
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    gallery_chunk: int = DEFAULT_GALLERY_CHUNK,
) -> dict[str, float]:
    """Recall@K in both directions.

    Args:
        img: ``[N_img, D]`` normalised image embeddings.
        txt: ``[N_txt, D]`` normalised text embeddings.
        text_owner: ``[N_txt]`` -- ``text_owner[j]`` is the image index that text
            ``j`` describes. Images may own any number of texts.
        device: where to score. Defaults to the device ``img`` is already on;
            embeddings may live on the CPU and be scored on a GPU.

    Image-to-text scores a hit when any text the image owns is in the top K;
    text-to-image scores a hit when the text's own image is in the top K.
    """
    score_device = torch.device(device) if device is not None else img.device
    owner = text_owner.cpu()
    k_max = max(ks)

    top_i2t = _topk_indices(img, txt, k_max, score_device, query_chunk, gallery_chunk)
    top_t2i = _topk_indices(txt, img, k_max, score_device, query_chunk, gallery_chunk)

    image_ids = torch.arange(img.size(0)).unsqueeze(1)

    def _rate(hits: torch.Tensor) -> float:
        # Integer hits divided in Python: a float32 mean would round the reported
        # metric by ~1e-8 and stop results diffing cleanly against past runs.
        return int(hits.sum().item()) / max(1, hits.numel())

    out: dict[str, float] = {}
    for k in ks:
        retrieved_owner = owner[top_i2t[:, : min(k, top_i2t.size(1))]]
        out[f"I2T_R@{k}"] = _rate((retrieved_owner == image_ids).any(dim=1))

        retrieved_image = top_t2i[:, : min(k, top_t2i.size(1))]
        out[f"T2I_R@{k}"] = _rate((retrieved_image == owner.unsqueeze(1)).any(dim=1))
    return out


def owner_one_to_one(n: int, device: str = "cpu") -> torch.Tensor:
    """Ownership map when text ``i`` belongs to image ``i``."""
    return torch.arange(n, device=device)


def owner_image_major(n_images: int, k: int, device: str = "cpu") -> torch.Tensor:
    """Ownership map for an image-major text bank of ``k`` texts per image."""
    return torch.arange(n_images, device=device).repeat_interleave(k)


def owner_from_lists(cap_to_img: list[int], device: str = "cpu") -> torch.Tensor:
    """Ownership map from an explicit caption-to-image list (COCO, Flickr30k)."""
    return torch.as_tensor(cap_to_img, dtype=torch.long, device=device)
