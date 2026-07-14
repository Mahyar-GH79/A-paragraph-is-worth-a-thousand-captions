"""The ten training configurations reported in the paper.

Every config trains the same model with the same optimiser; they differ only in
*which texts* are paired with each image, which is the question the paper asks.
The configs are grouped as:

* caption-only        -- cfg1, cfg2, cfg3, cfg8
* paragraph-bearing   -- cfg4, cfg5, cfg6, cfg9, cfg10
* hard-negative       -- cfg7

Batch size drops from 256 to 128 where a config carries many texts per image, so
that the text bank per step (``batch_size x texts_per_image``) stays tractable.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from capara.common.shards import (
    ORIGINAL,
    PARAGRAPH,
    RANDOM_NEGATIVE,
    RANDOM_POSITIVE,
    Requirement,
    TextSource,
    top_positives,
)

ALL_POSITIVES = 5


@dataclass
class TrainConfig:
    """A training run.

    Attributes:
        train_sources: the texts paired with each image during training.
        val_sources: the texts used for validation recall.
        val_loss_sources: the texts used for the validation loss. Several configs
            deliberately score the loss on a fixed subset so the curve stays
            comparable across epochs even though training samples texts at random.
        requires: a record lacking any of these fields is skipped.
    """

    name: str
    description: str

    train_sources: list[TextSource]
    val_sources: list[TextSource]
    val_loss_sources: list[TextSource] | None = None
    requires: list[Requirement] = field(default_factory=list)

    model_name: str = "Salesforce/blip-itm-base-coco"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    temperature: float = 0.07

    max_length_caption: int = 77
    max_length_paragraph: int = 128

    val_shards: int = 10
    num_workers: int = 4
    use_fp16: bool = True

    #: Persistent dataloader workers hold their own copy of the dataset, so the
    #: per-epoch reseeding in ``ShardTextDataset.set_epoch`` never reaches them
    #: and configs that sample a random caption draw the SAME one every epoch.
    #: The published runs were trained this way. Set false to make the resampling
    #: take effect -- results will then differ from the paper.
    persistent_workers: bool = True

    log_every_steps: int = 50
    metrics_every_steps: int = 20
    save_every_epochs: int = 1
    keep_last_k_checkpoints: int = 3
    max_steps_per_epoch: int | None = None
    plot_dpi: int = 160

    def __post_init__(self) -> None:
        if self.val_loss_sources is None:
            self.val_loss_sources = list(self.val_sources)

    @property
    def uses_hard_negatives(self) -> bool:
        return any(source.negative for source in self.train_sources)


def _config(name: str, description: str, **kwargs) -> TrainConfig:
    return TrainConfig(name=name, description=description, **kwargs)


CONFIGS: dict[str, TrainConfig] = {
    "cfg1": _config(
        "cfg1_original",
        "Original caption",
        train_sources=[ORIGINAL],
        val_sources=[ORIGINAL],
        requires=[Requirement.ORIGINAL],
    ),
    "cfg2": _config(
        "cfg2_random_positive",
        "Random positive caption",
        train_sources=[RANDOM_POSITIVE],
        # Recall is scored against all five positives; the loss uses only the
        # first so that it does not move with the random draw.
        val_sources=[top_positives(ALL_POSITIVES, pad_with_original=True)],
        val_loss_sources=[top_positives(1, pad_with_original=True)],
    ),
    "cfg3": _config(
        "cfg3_original_plus_random_positive",
        "Original + random positive",
        train_sources=[ORIGINAL, RANDOM_POSITIVE],
        val_sources=[ORIGINAL, top_positives(ALL_POSITIVES, pad_with_original=True)],
        val_loss_sources=[ORIGINAL, top_positives(1, pad_with_original=True)],
        requires=[Requirement.ORIGINAL],
    ),
    "cfg4": _config(
        "cfg4_paragraph_only",
        "Paragraph only",
        train_sources=[PARAGRAPH],
        val_sources=[PARAGRAPH],
        requires=[Requirement.PARAGRAPH],
    ),
    "cfg5": _config(
        "cfg5_original_plus_paragraph",
        "Original caption + paragraph",
        train_sources=[ORIGINAL, PARAGRAPH],
        val_sources=[ORIGINAL, PARAGRAPH],
        requires=[Requirement.ORIGINAL, Requirement.PARAGRAPH],
    ),
    "cfg6": _config(
        "cfg6_random_positive_plus_paragraph",
        "Random positive + paragraph",
        train_sources=[RANDOM_POSITIVE, PARAGRAPH],
        val_sources=[top_positives(ALL_POSITIVES), PARAGRAPH],
        requires=[Requirement.POSITIVES, Requirement.PARAGRAPH],
    ),
    "cfg7": _config(
        "cfg7_original_plus_positive_plus_hard_negative",
        "Original + positive + hard negative",
        train_sources=[ORIGINAL, RANDOM_POSITIVE, RANDOM_NEGATIVE],
        # Validation is 1-to-1 on the original caption: the negatives are a
        # training-time distractor, not a retrieval target.
        val_sources=[ORIGINAL],
        requires=[
            Requirement.ORIGINAL,
            Requirement.POSITIVES,
            Requirement.NEGATIVES,
        ],
    ),
    "cfg8": _config(
        "cfg8_original_plus_all_positives",
        "Original + all 5 positives",
        train_sources=[ORIGINAL, top_positives(ALL_POSITIVES, pad_with_original=True)],
        val_sources=[ORIGINAL, top_positives(ALL_POSITIVES, pad_with_original=True)],
        requires=[Requirement.ORIGINAL],
        batch_size=128,
    ),
    "cfg9": _config(
        "cfg9_original_plus_all_positives_plus_paragraph",
        "Original + all 5 positives + paragraph",
        train_sources=[
            ORIGINAL,
            top_positives(ALL_POSITIVES, pad_with_original=True),
            PARAGRAPH,
        ],
        val_sources=[
            ORIGINAL,
            top_positives(ALL_POSITIVES, pad_with_original=True),
            PARAGRAPH,
        ],
        requires=[Requirement.ORIGINAL, Requirement.PARAGRAPH],
        batch_size=128,
    ),
    "cfg10": _config(
        "cfg10_original_plus_random_positive_plus_paragraph",
        "Original + random positive + paragraph",
        train_sources=[ORIGINAL, RANDOM_POSITIVE, PARAGRAPH],
        val_sources=[ORIGINAL, RANDOM_POSITIVE, PARAGRAPH],
        requires=[Requirement.ORIGINAL, Requirement.PARAGRAPH],
        batch_size=128,
    ),
}


def get_config(name: str) -> TrainConfig:
    """Look up a config by key (``cfg1`` .. ``cfg10``)."""
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown config {name!r}. Available: {', '.join(CONFIGS)}"
        ) from None


def config_names() -> Sequence[str]:
    return tuple(CONFIGS)
