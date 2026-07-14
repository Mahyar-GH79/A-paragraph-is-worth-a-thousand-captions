"""The merged losses and metrics must be numerically identical to the ten original
per-config scripts, otherwise the unified trainer would not reproduce the paper.

Each test re-implements the original formulation verbatim and compares.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capara.common.losses import (  # noqa: E402
    multi_positive_infonce,
    multi_positive_infonce_with_negatives,
)
from capara.common.metrics import (  # noqa: E402
    owner_from_lists,
    owner_image_major,
    owner_one_to_one,
    recall_at_k,
)
from capara.common.shards import Example, build_text_batch  # noqa: E402

TEMPERATURE = 0.07
TOL = 1e-5


def _normed(*shape):
    return F.normalize(torch.randn(*shape, dtype=torch.float64), dim=-1)


# --- originals, copied from the train_cfg*.py scripts -------------------------


def original_clip_style_contrastive_loss(img, txt, temperature):
    """train_cfg1.py:235"""
    logits = (img @ txt.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def original_loss_multi_positive_K(img, txt_flat, temperature, K):
    """train_cfg8.py loss_multi_positive_K"""
    B = img.size(0)
    device = img.device
    logits_i2t = (img @ txt_flat.t()) / temperature
    lse_all = torch.logsumexp(logits_i2t, dim=1)
    idx = torch.arange(B, device=device)
    pos_indices = idx.unsqueeze(1) * K + torch.arange(K, device=device).unsqueeze(0)
    lse_pos = torch.logsumexp(logits_i2t.gather(1, pos_indices), dim=1)
    loss_i2t = (lse_all - lse_pos).mean()
    logits_t2i = (txt_flat @ img.t()) / temperature
    labels_t2i = torch.arange(B, device=device).repeat_interleave(K)
    return 0.5 * (loss_i2t + F.cross_entropy(logits_t2i, labels_t2i))


def original_loss_with_explicit_negs(img, txt_orig, txt_pos, txt_neg, temperature):
    """train_cfg7.py loss_multi_pos_with_explicit_negs"""
    b = img.size(0)
    device = img.device
    txt_all = torch.cat([txt_orig, txt_pos, txt_neg], dim=0)
    logits_i2t = (img @ txt_all.t()) / temperature
    idx = torch.arange(b, device=device)
    pos_logits = torch.stack([logits_i2t[idx, idx], logits_i2t[idx, idx + b]], dim=1)
    loss_i2t = -(
        torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(logits_i2t, dim=1)
    ).mean()
    txt_pos_all = torch.cat([txt_orig, txt_pos], dim=0)
    logits_t2i = (txt_pos_all @ img.t()) / temperature
    labels_t2i = torch.arange(b, device=device).repeat(2)
    loss_t2i = F.cross_entropy(logits_t2i, labels_t2i)

    s_orig = (img * txt_orig).sum(dim=1) / temperature
    s_pos = (img * txt_pos).sum(dim=1) / temperature
    s_neg = (img * txt_neg).sum(dim=1) / temperature
    margin = (torch.maximum(s_orig, s_pos) - s_neg).mean()
    return 0.5 * (loss_i2t + loss_t2i), margin


def original_retrieval_metrics(img, txt):
    """train_cfg1.py:244 -- 1-to-1 recall"""
    sim = img @ txt.t()
    b = sim.size(0)
    gt = torch.arange(b)
    ri = sim.argsort(dim=1, descending=True)
    rt = sim.t().argsort(dim=1, descending=True)

    def recall(ranks, k):
        k = min(k, ranks.size(1))
        return float((ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean())

    return {f"I2T_R@{k}": recall(ri, k) for k in (1, 5, 10)} | {
        f"T2I_R@{k}": recall(rt, k) for k in (1, 5, 10)
    }


# --- equivalence tests --------------------------------------------------------


def test_single_positive_reduces_to_clip_infonce():
    """cfg1, cfg2, cfg4: one text per image."""
    torch.manual_seed(0)
    img, txt = _normed(16, 256), _normed(16, 256)
    pos_index = torch.arange(16).unsqueeze(1)
    owner = owner_one_to_one(16)

    merged = multi_positive_infonce(img, txt, pos_index, owner, TEMPERATURE)
    original = original_clip_style_contrastive_loss(img, txt, TEMPERATURE)
    assert torch.allclose(merged, original, atol=TOL)


@pytest.mark.parametrize("k", [2, 3, 6, 7])
def test_multi_positive_matches_original(k):
    """cfg3, cfg5, cfg6, cfg8, cfg9, cfg10: k image-major texts per image."""
    torch.manual_seed(k)
    b = 12
    img, txt = _normed(b, 256), _normed(b * k, 256)
    pos_index = torch.arange(b).unsqueeze(1) * k + torch.arange(k).unsqueeze(0)
    owner = owner_image_major(b, k)

    merged = multi_positive_infonce(img, txt, pos_index, owner, TEMPERATURE)
    original = original_loss_multi_positive_K(img, txt, TEMPERATURE, k)
    assert torch.allclose(merged, original, atol=TOL)


def test_hard_negative_loss_and_margin_match_original():
    """cfg7: negatives extend the i2t denominator only."""
    torch.manual_seed(7)
    b = 12
    img = _normed(b, 256)
    txt_orig, txt_pos, txt_neg = _normed(b, 256), _normed(b, 256), _normed(b, 256)

    # The merged loss takes an image-major positive bank: [orig_0, pos_0, orig_1, ...]
    txt_pos_flat = torch.stack([txt_orig, txt_pos], dim=1).reshape(2 * b, 256)
    pos_index = torch.arange(b).unsqueeze(1) * 2 + torch.arange(2).unsqueeze(0)
    neg_index = torch.arange(b).unsqueeze(1)
    owner = owner_image_major(b, 2)

    merged_loss, merged_margin = multi_positive_infonce_with_negatives(
        img, txt_pos_flat, txt_neg, pos_index, neg_index, owner, TEMPERATURE
    )
    original_loss, original_margin = original_loss_with_explicit_negs(
        img, txt_orig, txt_pos, txt_neg, TEMPERATURE
    )
    assert torch.allclose(merged_loss, original_loss, atol=TOL)
    assert torch.allclose(merged_margin, original_margin, atol=TOL)


def test_ragged_positives_ignore_padding():
    """cfg6 validation: images may carry different numbers of positives."""
    torch.manual_seed(1)
    img, txt = _normed(3, 256), _normed(6, 256)
    # image 0 owns texts {0,1}, image 1 owns {2,3,4}, image 2 owns {5}
    pos_index = torch.tensor([[0, 1, -1], [2, 3, 4], [5, -1, -1]])
    owner = torch.tensor([0, 0, 1, 1, 1, 2])

    loss = multi_positive_infonce(img, txt, pos_index, owner, TEMPERATURE)
    assert torch.isfinite(loss)

    # Padding must not change the answer: widening the pad columns is a no-op.
    wider = torch.cat([pos_index, torch.full((3, 2), -1)], dim=1)
    assert torch.allclose(
        loss, multi_positive_infonce(img, txt, wider, owner, TEMPERATURE), atol=TOL
    )


def test_recall_one_to_one_matches_original():
    torch.manual_seed(2)
    img, txt = _normed(64, 256), _normed(64, 256)
    merged = recall_at_k(img, txt, owner_one_to_one(64))
    original = original_retrieval_metrics(img, txt)
    for key, value in original.items():
        assert abs(merged[key] - value) < TOL, key


def test_recall_multi_caption_matches_hand_computation():
    """COCO/Flickr: an i2t hit means ANY caption of the image is retrieved."""
    torch.manual_seed(3)
    img = _normed(4, 256)
    txt = _normed(8, 256)
    cap_to_img = [0, 0, 1, 1, 2, 2, 3, 3]

    scores = recall_at_k(img, txt, owner_from_lists(cap_to_img), ks=(1,))

    sim = img @ txt.t()
    top1 = sim.argmax(dim=1)
    expected_i2t = sum(
        1 for i, j in enumerate(top1.tolist()) if cap_to_img[j] == i
    ) / 4
    assert abs(scores["I2T_R@1"] - expected_i2t) < TOL


def test_build_text_batch_indexes_are_consistent():
    """The flattening must keep pos_index, text_owner and the text list aligned."""
    examples = [
        Example(
            image=torch.zeros(256),
            positives=["a", "b"],
            negatives=["n1"],
            positive_is_paragraph=[False, True],
        ),
        Example(
            image=torch.zeros(256),
            positives=["c"],
            negatives=[],
            positive_is_paragraph=[False],
        ),
    ]
    batch = build_text_batch(examples)

    assert batch.positives == ["a", "b", "c"]
    assert batch.negatives == ["n1"]
    assert batch.pos_is_paragraph == [False, True, False]
    assert batch.text_owner.tolist() == [0, 0, 1]
    assert batch.pos_index.tolist() == [[0, 1], [2, -1]]
    assert batch.neg_index.tolist() == [[0], [-1]]
