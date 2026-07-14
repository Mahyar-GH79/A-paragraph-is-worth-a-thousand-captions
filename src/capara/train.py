"""Fine-tune BLIP's text tower against precomputed image embeddings.

The vision tower is frozen and never executed: images enter as the 256-d
embeddings built by ``capara.data.build_image_embeddings`` and projected by
``capara.data.project_embeddings``. Only ``text_encoder`` and ``text_proj`` are
trained, which is what makes ten full configurations affordable.

Usage:
    python -m capara.train --config cfg5
    python -m capara.train --config cfg5 --device cpu --epochs 1 --max-steps-per-epoch 2
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from capara.common.blip import encode_texts, freeze_vision_tower, load_blip
from capara.common.losses import (
    multi_positive_infonce,
    multi_positive_infonce_with_negatives,
)
from capara.common.metrics import recall_at_k
from capara.common.paths import SHARDS_256_DIR, TRAIN_RUNS_DIR
from capara.common.shards import (
    ShardTextDataset,
    TextBatch,
    TextSource,
    build_text_batch,
    collate_examples,
    count_rows,
    list_shards,
)
from capara.configs import TrainConfig, config_names, get_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_mixed_lengths(
    model,
    processor,
    texts: Sequence[str],
    is_paragraph: Sequence[bool],
    device: str,
    max_length_caption: int,
    max_length_paragraph: int,
) -> torch.Tensor:
    """Embed a mixed batch, tokenising paragraphs at a longer max_length than captions.

    Returns ``[len(texts), 256]`` in the original text order.
    """
    caption_rows = [i for i, para in enumerate(is_paragraph) if not para]
    paragraph_rows = [i for i, para in enumerate(is_paragraph) if para]

    chunks: list[torch.Tensor] = []
    order: list[int] = []

    for rows, max_length in (
        (caption_rows, max_length_caption),
        (paragraph_rows, max_length_paragraph),
    ):
        if not rows:
            continue
        chunks.append(
            encode_texts(model, processor, [texts[i] for i in rows], device, max_length)
        )
        order.extend(rows)

    embedded = torch.cat(chunks, dim=0)
    positions = torch.tensor(order, device=embedded.device)
    inverse = torch.empty_like(positions)
    inverse[positions] = torch.arange(len(order), device=embedded.device)
    return embedded[inverse]


def compute_loss(
    model,
    processor,
    images: torch.Tensor,
    batch: TextBatch,
    cfg: TrainConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return ``(loss, margin_or_None, positive_text_embeddings)``."""
    txt_pos = encode_mixed_lengths(
        model,
        processor,
        batch.positives,
        batch.pos_is_paragraph,
        device,
        cfg.max_length_caption,
        cfg.max_length_paragraph,
    )
    pos_index = batch.pos_index.to(device)
    text_owner = batch.text_owner.to(device)

    if batch.negatives:
        txt_neg = encode_texts(
            model, processor, batch.negatives, device, cfg.max_length_caption
        )
        loss, margin = multi_positive_infonce_with_negatives(
            images,
            txt_pos,
            txt_neg,
            pos_index,
            batch.neg_index.to(device),
            text_owner,
            cfg.temperature,
        )
        return loss, margin, txt_pos

    loss = multi_positive_infonce(images, txt_pos, pos_index, text_owner, cfg.temperature)
    return loss, None, txt_pos


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup, then cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def build_loader(
    shard_paths: Sequence[str],
    sources: Sequence[TextSource],
    cfg: TrainConfig,
    device: str,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    dataset = ShardTextDataset(
        shard_paths,
        sources=sources,
        requires=cfg.requires,
        shuffle_shards=shuffle,
        seed=cfg.seed,
    )
    persistent = cfg.persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=drop_last,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_examples,
    )


@torch.no_grad()
def run_validation(
    model,
    processor,
    loss_loader: DataLoader,
    metric_loader: DataLoader | None,
    cfg: TrainConfig,
    device: str,
    total_steps: int,
) -> dict[str, float]:
    """Validation loss and recall.

    Most configs score both on the same texts and ``metric_loader`` is ``None``,
    so the shards are read once. Configs whose loss is deliberately restricted to
    a fixed subset of the validation texts (cfg2, cfg3) pass a second loader.
    """
    model.eval()

    loss_sum = 0.0
    n_loss = 0
    recall_sums: dict[str, float] = {}
    n_recall = 0

    single_pass = metric_loader is None

    for images, examples in tqdm(
        loss_loader, total=total_steps, desc="val", leave=False
    ):
        images = F.normalize(images.to(device), dim=-1)
        batch = build_text_batch(examples)
        loss, _, txt = compute_loss(model, processor, images, batch, cfg, device)

        loss_sum += float(loss.item())
        n_loss += 1

        if single_pass:
            for key, value in recall_at_k(
                images, txt, batch.text_owner.to(device)
            ).items():
                recall_sums[key] = recall_sums.get(key, 0.0) + value
            n_recall += 1

    if not single_pass:
        for images, examples in tqdm(
            metric_loader, total=total_steps, desc="val recall", leave=False
        ):
            images = F.normalize(images.to(device), dim=-1)
            batch = build_text_batch(examples)
            txt = encode_mixed_lengths(
                model,
                processor,
                batch.positives,
                batch.pos_is_paragraph,
                device,
                cfg.max_length_caption,
                cfg.max_length_paragraph,
            )
            for key, value in recall_at_k(
                images, txt, batch.text_owner.to(device)
            ).items():
                recall_sums[key] = recall_sums.get(key, 0.0) + value
            n_recall += 1

    stats = {"val_loss": loss_sum / max(1, n_loss)}
    for key, total in recall_sums.items():
        stats[f"val_{key}"] = total / max(1, n_recall)
    return stats


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    keys = sorted(history)
    n_epochs = max((len(history[key]) for key in keys), default=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for epoch in range(n_epochs):
            writer.writerow(
                [epoch]
                + [history[k][epoch] if epoch < len(history[k]) else "" for k in keys]
            )


def save_plots(run_dir: Path, history: dict[str, list[float]], dpi: int) -> None:
    def plot(keys: Sequence[str], title: str, ylabel: str, filename: str) -> None:
        present = [key for key in keys if history.get(key)]
        if not present:
            return
        plt.figure()
        for key in present:
            plt.plot(range(len(history[key])), history[key], label=key)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=dpi)
        plt.close()

    plot(["train_loss", "val_loss"], "Loss curves", "loss", "loss_curves.png")
    plot(
        ["val_I2T_R@1", "val_I2T_R@5", "val_I2T_R@10"],
        "Validation recall, image to text",
        "recall",
        "val_recall_i2t.png",
    )
    plot(
        ["val_T2I_R@1", "val_T2I_R@5", "val_T2I_R@10"],
        "Validation recall, text to image",
        "recall",
        "val_recall_t2i.png",
    )
    plot(["train_margin"], "Positive-negative margin", "logits", "train_margin.png")


def trim_checkpoints(ckpt_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    for path in checkpoints[: max(0, len(checkpoints) - keep_last_k)]:
        path.unlink(missing_ok=True)


def train(cfg: TrainConfig, shards_dir: Path, out_root: Path, device: str) -> Path:
    set_seed(cfg.seed)

    shards = list_shards(str(shards_dir))
    if not shards:
        raise SystemExit(f"No shards found in {shards_dir}")
    if cfg.val_shards >= len(shards):
        raise SystemExit(f"val_shards={cfg.val_shards} but only {len(shards)} shards exist")

    train_shards = shards[: -cfg.val_shards]
    val_shards = shards[-cfg.val_shards :]

    run_dir = out_root / f"{cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config     : {cfg.name} -- {cfg.description}")
    print(f"Run dir    : {run_dir}")
    print(f"Device     : {device}")
    print(f"Shards     : {len(train_shards)} train, {len(val_shards)} val")
    print(f"Batch size : {cfg.batch_size}, epochs: {cfg.epochs}, lr: {cfg.lr}")

    # Row counts drive the progress bar and the cosine schedule. They count every
    # row in a shard, including records a config later drops for a missing
    # paragraph, so paragraph configs run slightly fewer steps than planned and
    # their learning rate never fully anneals. The published runs behaved this
    # way; changing it would change the reported numbers.
    train_rows = count_rows(train_shards)
    val_rows = count_rows(val_shards)

    steps_per_epoch = train_rows // cfg.batch_size
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    val_steps = math.ceil(val_rows / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    print(f"Steps/epoch: {steps_per_epoch} (total {total_steps})")

    save_json(run_dir / "train_config.json", asdict(cfg))

    model, processor = load_blip(cfg.model_name, device=device)
    freeze_vision_tower(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.use_fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = build_loader(
        train_shards, cfg.train_sources, cfg, device,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
    )
    val_loss_loader = build_loader(
        val_shards, cfg.val_loss_sources, cfg, device,
        shuffle=False, num_workers=0, drop_last=False,
    )
    needs_second_pass = cfg.val_loss_sources != cfg.val_sources
    val_metric_loader = (
        build_loader(
            val_shards, cfg.val_sources, cfg, device,
            shuffle=False, num_workers=0, drop_last=False,
        )
        if needs_second_pass
        else None
    )

    history: dict[str, list[float]] = {}

    def record(key: str, value: float) -> None:
        history.setdefault(key, []).append(value)

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_loader.dataset.set_epoch(epoch)

        running_loss = 0.0
        running_margin = 0.0
        n_batches = 0
        started = time.time()

        pbar = tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
            dynamic_ncols=True,
        )
        for step, (images, examples) in enumerate(pbar):
            if step >= steps_per_epoch:
                break

            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                images = F.normalize(images, dim=-1)

            batch = build_text_batch(examples)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss, margin, txt = compute_loss(model, processor, images, batch, cfg, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_multiplier(global_step, total_steps, cfg.warmup_steps)

            running_loss += float(loss.item())
            if margin is not None:
                running_margin += float(margin.item())
            n_batches += 1

            if step % cfg.metrics_every_steps == 0:
                with torch.no_grad():
                    scores = recall_at_k(images, txt, batch.text_owner.to(device), ks=(1, 5))
                pbar.set_postfix(
                    loss=f"{running_loss / n_batches:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    r1i=f"{scores['I2T_R@1']:.2f}",
                    r1t=f"{scores['T2I_R@1']:.2f}",
                )

        record("train_loss", running_loss / max(1, n_batches))
        if cfg.uses_hard_negatives:
            record("train_margin", running_margin / max(1, n_batches))

        print(
            f"epoch {epoch + 1}: train_loss={history['train_loss'][-1]:.6f} "
            f"({time.time() - started:.0f}s)"
        )

        if (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": asdict(cfg),
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )
            trim_checkpoints(ckpt_dir, cfg.keep_last_k_checkpoints)

        stats = run_validation(
            model, processor, val_loss_loader, val_metric_loader, cfg, device, val_steps
        )
        for key, value in stats.items():
            record(key, value)

        print(
            f"  val_loss={stats['val_loss']:.6f} "
            f"I2T_R@1={stats.get('val_I2T_R@1', 0.0):.3f} "
            f"T2I_R@1={stats.get('val_T2I_R@1', 0.0):.3f}"
        )

        save_json(run_dir / "history.json", history)
        save_history_csv(run_dir / "history.csv", history)
        save_plots(run_dir, history, cfg.plot_dpi)

    final_model = run_dir / "final_model.pt"
    torch.save(
        {"model_state": model.state_dict(), "cfg": asdict(cfg), "history": history},
        final_model,
    )
    save_json(
        run_dir / "final_report.json",
        {
            "run_dir": str(run_dir),
            "final_model": str(final_model),
            "final_epoch": cfg.epochs,
            "final_stats": {key: (values[-1] if values else None) for key, values in history.items()},
        },
    )

    print(f"\nDone. Final model: {final_model}")
    return final_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BLIP's text tower on precomputed image embeddings."
    )
    parser.add_argument("--config", required=True, choices=config_names())
    parser.add_argument("--shards-dir", type=Path, default=SHARDS_256_DIR)
    parser.add_argument("--out-root", type=Path, default=TRAIN_RUNS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, help="override the config's epoch count")
    parser.add_argument("--batch-size", type=int, help="override the config's batch size")
    parser.add_argument("--num-workers", type=int, help="dataloader worker processes")
    parser.add_argument(
        "--val-shards", type=int, help="how many trailing shards to hold out for validation"
    )
    parser.add_argument(
        "--max-steps-per-epoch", type=int, help="cap optimiser steps per epoch (smoke tests)"
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help=(
            "disable persistent dataloader workers so that per-epoch caption "
            "resampling actually takes effect (diverges from the published runs)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args.config)

    for name in ("epochs", "batch_size", "num_workers", "max_steps_per_epoch", "val_shards"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_persistent_workers:
        cfg.persistent_workers = False

    train(cfg, args.shards_dir, args.out_root, args.device)


if __name__ == "__main__":
    main()
