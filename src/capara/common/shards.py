"""Streaming access to the precomputed BLIP image-embedding shards.

Training never touches an image. Each shard is a ``torch.save`` dict with

* ``image_feats``: ``[N, 256]`` L2-normalised BLIP ITC image embeddings (fp16 on disk)
* ``records``:     ``[N]`` dicts carrying ``original_caption``, ``positive_captions``,
                   ``hard_negative_captions`` and ``paragraph``

so a config is fully described by *which texts* it pairs with each image. That
choice is expressed as a list of :class:`TextSource`.
"""

import glob
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .text import clean_str_list, clean_text

EMBED_DIM = 256


class SourceKind(str, Enum):
    """Where a text comes from in a CC3M annotation record."""

    ORIGINAL = "original"
    PARAGRAPH = "paragraph"
    RANDOM_POSITIVE = "random_positive"
    TOP_POSITIVES = "top_positives"
    RANDOM_NEGATIVE = "random_negative"
    TOP_NEGATIVES = "top_negatives"


@dataclass(frozen=True)
class TextSource:
    """One text slot in a training example.

    Attributes:
        kind: which field of the record to draw from.
        count: how many texts ``TOP_POSITIVES`` yields.
        pad_with_original: if a record has fewer than ``count`` positives, repeat
            the original caption to keep the slot count fixed. When false the
            example simply carries fewer texts and the loss handles the ragged row.
        negative: the text is a distractor, never a retrieval target.
    """

    kind: SourceKind
    count: int = 1
    pad_with_original: bool = False
    negative: bool = False

    @property
    def is_paragraph(self) -> bool:
        return self.kind is SourceKind.PARAGRAPH

    def resolve(self, record: dict[str, Any], rng: random.Random) -> list[str]:
        """Draw this slot's texts from ``record``. May return an empty list."""
        original = clean_text(record.get("original_caption"))

        if self.kind is SourceKind.ORIGINAL:
            return [original] if original else []

        if self.kind is SourceKind.PARAGRAPH:
            paragraph = clean_text(record.get("paragraph"))
            return [paragraph] if paragraph else []

        if self.kind is SourceKind.RANDOM_POSITIVE:
            positives = clean_str_list(record.get("positive_captions"))
            if positives:
                return [rng.choice(positives)]
            return [original] if original else []

        if self.kind is SourceKind.TOP_POSITIVES:
            positives = clean_str_list(record.get("positive_captions"))[: self.count]
            if self.pad_with_original and original:
                while len(positives) < self.count:
                    positives.append(original)
            return positives

        if self.kind is SourceKind.RANDOM_NEGATIVE:
            negatives = clean_str_list(record.get("hard_negative_captions"))
            return [rng.choice(negatives)] if negatives else []

        if self.kind is SourceKind.TOP_NEGATIVES:
            negatives = clean_str_list(record.get("hard_negative_captions"))
            if not negatives:
                return []
            # Cycle rather than drop, so every image contributes the same number
            # of negatives and the only variable across N1/N3/N5 is that number.
            return [negatives[i % len(negatives)] for i in range(self.count)]

        raise ValueError(f"Unhandled source kind: {self.kind}")


# Shorthand constructors for the configs.
ORIGINAL = TextSource(SourceKind.ORIGINAL)
PARAGRAPH = TextSource(SourceKind.PARAGRAPH)
RANDOM_POSITIVE = TextSource(SourceKind.RANDOM_POSITIVE)
RANDOM_NEGATIVE = TextSource(SourceKind.RANDOM_NEGATIVE, negative=True)


def top_positives(count: int, pad_with_original: bool = False) -> TextSource:
    """A record's first ``count`` positive captions."""
    return TextSource(
        SourceKind.TOP_POSITIVES, count=count, pad_with_original=pad_with_original
    )


def top_negatives(count: int) -> TextSource:
    """A record's first ``count`` hard negatives, pinned rather than resampled.

    Used by the negative-scaling ablation, where the number of negatives in the
    denominator is the only thing allowed to vary.
    """
    return TextSource(SourceKind.TOP_NEGATIVES, count=count, negative=True)


class Requirement(str, Enum):
    """A record is dropped unless the required fields are present and non-empty."""

    ORIGINAL = "original"
    PARAGRAPH = "paragraph"
    POSITIVES = "positives"
    NEGATIVES = "negatives"


def record_satisfies(record: dict[str, Any], requires: Sequence[Requirement]) -> bool:
    for requirement in requires:
        if requirement is Requirement.ORIGINAL:
            if not clean_text(record.get("original_caption")):
                return False
        elif requirement is Requirement.PARAGRAPH:
            if not clean_text(record.get("paragraph")):
                return False
        elif requirement is Requirement.POSITIVES:
            if not clean_str_list(record.get("positive_captions")):
                return False
        elif requirement is Requirement.NEGATIVES:
            if not clean_str_list(record.get("hard_negative_captions")):
                return False
    return True


@dataclass
class Example:
    """One image and the texts a config pairs with it."""

    image: torch.Tensor  # [256]
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    #: Parallel to ``positives``; True where the text is a paragraph, which is
    #: tokenised at a longer max_length than a caption.
    positive_is_paragraph: list[bool] = field(default_factory=list)


def list_shards(shards_dir: str) -> list[str]:
    """Shard files in a stable order, falling back to partial shards."""
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard_*.pt")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(shards_dir, "partial_shard_*.pt")))
    return shards


def count_rows(shard_paths: Sequence[str]) -> int:
    """Total rows across shards, used to size the progress bar and LR schedule."""
    total = 0
    for path in shard_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        feats = obj.get("image_feats")
        if not torch.is_tensor(feats) or feats.ndim != 2 or feats.shape[1] != EMBED_DIM:
            shape = None if feats is None else tuple(feats.shape)
            raise ValueError(
                f"Shard {path} has image_feats {shape}, expected [N, {EMBED_DIM}]"
            )
        total += int(feats.shape[0])
    return total


class ShardTextDataset(IterableDataset):
    """Streams :class:`Example` objects from shards for a given set of text sources.

    Shards are split across dataloader workers. Sources that sample randomly are
    re-seeded per epoch via :meth:`set_epoch`, so the drawn caption changes each
    epoch while staying reproducible.
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        sources: Sequence[TextSource],
        requires: Sequence[Requirement] = (),
        shuffle_shards: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.sources = list(sources)
        self.requires = list(requires)
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        if worker is None:
            shard_paths = list(self.shard_paths)
        else:
            shard_paths = self.shard_paths[worker.id :: worker.num_workers]

        rng = random.Random(self.seed + 1000 * self.epoch + 17 * worker_id)
        if self.shuffle_shards:
            rng.shuffle(shard_paths)

        for path in shard_paths:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            feats = obj["image_feats"]
            records = obj["records"]

            for row in range(int(feats.shape[0])):
                record = records[row]
                if not record_satisfies(record, self.requires):
                    continue

                example = Example(image=feats[row].to(torch.float32))
                for source in self.sources:
                    texts = source.resolve(record, rng)
                    if source.negative:
                        example.negatives.extend(texts)
                    else:
                        example.positives.extend(texts)
                        example.positive_is_paragraph.extend(
                            [source.is_paragraph] * len(texts)
                        )

                if not example.positives:
                    continue
                yield example


def collate_examples(batch: list[Example]) -> tuple[torch.Tensor, list[Example]]:
    """Stack image embeddings; texts stay ragged and are tokenised downstream."""
    images = torch.stack([example.image for example in batch], dim=0)
    return images, batch


@dataclass
class TextBatch:
    """A flattened batch of texts ready for the loss.

    ``positives``/``negatives`` are flat text lists. ``pos_index`` and ``neg_index``
    map each image to its own rows (``-1``-padded), and ``text_owner`` maps each
    positive text back to its image.
    """

    positives: list[str]
    negatives: list[str]
    pos_is_paragraph: list[bool]
    pos_index: torch.Tensor  # [B, P_max]
    neg_index: torch.Tensor  # [B, N_max]
    text_owner: torch.Tensor  # [T_pos]


def build_text_batch(examples: Sequence[Example]) -> TextBatch:
    """Flatten per-image text lists into the index structures the losses expect."""
    positives: list[str] = []
    negatives: list[str] = []
    pos_is_paragraph: list[bool] = []
    owner: list[int] = []
    pos_rows: list[list[int]] = []
    neg_rows: list[list[int]] = []

    for image_id, example in enumerate(examples):
        rows = []
        for text, is_para in zip(example.positives, example.positive_is_paragraph, strict=True):
            rows.append(len(positives))
            positives.append(text)
            pos_is_paragraph.append(is_para)
            owner.append(image_id)
        pos_rows.append(rows)

        rows = []
        for text in example.negatives:
            rows.append(len(negatives))
            negatives.append(text)
        neg_rows.append(rows)

    return TextBatch(
        positives=positives,
        negatives=negatives,
        pos_is_paragraph=pos_is_paragraph,
        pos_index=_pad_index(pos_rows),
        neg_index=_pad_index(neg_rows),
        text_owner=torch.tensor(owner, dtype=torch.long),
    )


def _pad_index(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return torch.zeros(len(rows), 0, dtype=torch.long)
    padded = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return padded
