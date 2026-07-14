"""Embed the CC3M images referenced by the annotation JSONL with BLIP's vision tower.

Each surviving record contributes one row to a shard file. A shard is a dict
``{"image_feats": Tensor[N, 768], "records": list[dict]}`` where row ``i`` of
``image_feats`` belongs to ``records[i]``. ``index.jsonl`` maps every ``sample_id`` to
its ``(shard_id, row_in_shard)``.

Two properties of the shard format are load-bearing:

* The stored vectors are the L2-normalised **pooled vision output (768-d), before
  ``vision_proj``**, not BLIP's 256-d ITC embeddings. Keeping the pre-projection
  features means a retrained/altered projection can be applied later without
  re-downloading and re-encoding half a million images; ``project_embeddings.py``
  performs that 768 -> 256 projection. This is why the shared ``capara.common.blip``
  helpers cannot be used to encode here -- ``encode_images`` deliberately returns the
  projected 256-d ITC vectors.
* Features are stored as fp16. At ~500k images, fp32 would roughly double the ~1.5 GB
  footprint for no measurable retrieval difference.

``sample_id`` is the sha1 of the image URL. CC3M ships URLs rather than image IDs, and a
content-independent stable key lets a run resume (and de-duplicate) by reading back
``index.jsonl`` alone, without touching the images.
"""

import argparse
import hashlib
import json
import os
import time
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from capara.common.blip import load_blip
from capara.common.paths import ANNOTATIONS_DIR, BLIP_MODEL, EMBED_DATASET_DIR
from capara.common.text import clean_str_list, clean_text

try:
    import requests
except ImportError:
    requests = None

DEFAULT_JSONL = ANNOTATIONS_DIR / "cc3m_qwen_llama_500000.jsonl"
DEFAULT_SHARD_SIZE = 5000
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 2
DEFAULT_MIN_IMAGE_SIDE = 32

DEFAULT_PRINT_EVERY_SCANNED = 200
DEFAULT_PRINT_EVERY_KEPT = 100
DEFAULT_SNAPSHOT_EVERY_KEPT = 200
DEFAULT_INDEX_FLUSH_EVERY = 500

STORE_DTYPE = torch.float16


def stable_id_from_url(url: str) -> str:
    """Stable per-sample key: sha1 of the image URL (CC3M has no image IDs)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def download_image(
    url: str,
    timeout: float,
    retries: int,
    min_image_side: int,
) -> Image.Image | None:
    """Fetch an image as RGB, or ``None`` if it fails, is too small, or is undecodable."""
    if requests is None:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True, headers=headers)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            img = Image.open(r.raw).convert("RGB")
            width, height = img.size
            if min(width, height) < min_image_side:
                return None
            return img
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
                continue
            return None
    return None


def load_resume_state(index_jsonl: str) -> tuple[int, set[str]]:
    """Read ``index.jsonl`` and return the number of indexed rows and their sample_ids."""
    if not os.path.exists(index_jsonl):
        return 0, set()

    total = 0
    seen: set[str] = set()
    with open(index_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("sample_id", None)
            if isinstance(sample_id, str) and sample_id:
                seen.add(sample_id)
                total += 1
    return total, seen


def write_shard(
    shard_path: str,
    image_feats: torch.Tensor,
    records: list[dict[str, Any]],
) -> None:
    """Write a shard atomically so an interrupted run never leaves a half-written .pt."""
    tmp = shard_path + ".tmp"
    torch.save({"image_feats": image_feats, "records": records}, tmp)
    os.replace(tmp, shard_path)


def append_index_rows(index_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(index_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def encode_image_pooled(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    use_amp: bool,
) -> torch.Tensor:
    """Return the L2-normalised 768-d pooled vision output for one image.

    This is BLIP's vision tower *without* ``vision_proj``; see the module docstring for
    why the shards keep the pre-projection features.
    """
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            feat = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    else:
        feat = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().to(STORE_DTYPE).squeeze(0)


def build_record(obj: dict[str, Any], sample_id: str, url: str, original_caption: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_url": url,
        "original_caption": original_caption,
        "positive_captions": clean_str_list(obj.get("positive_captions", [])),
        "hard_negative_captions": clean_str_list(obj.get("hard_negative_captions", [])),
        "paragraph": clean_text(obj.get("paragraph", None)),
        "llama_score": (
            int(obj["llama_score"]) if isinstance(obj.get("llama_score", None), int) else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(DEFAULT_JSONL),
        help="Annotation JSONL produced by generate_annotations.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(EMBED_DATASET_DIR),
        help="Directory for shards/, index.jsonl and build_stats.json.",
    )
    parser.add_argument("--model", type=str, default=BLIP_MODEL)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Rows per shard file.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--min-image-side",
        type=int,
        default=DEFAULT_MIN_IMAGE_SIDE,
        help="Drop images whose shorter side is below this many pixels.",
    )
    parser.add_argument("--print-every-scanned", type=int, default=DEFAULT_PRINT_EVERY_SCANNED)
    parser.add_argument("--print-every-kept", type=int, default=DEFAULT_PRINT_EVERY_KEPT)
    parser.add_argument(
        "--snapshot-every-kept",
        type=int,
        default=DEFAULT_SNAPSHOT_EVERY_KEPT,
        help="Write a partial_shard_*.pt snapshot every N kept images; 0 disables snapshots.",
    )
    parser.add_argument("--index-flush-every", type=int, default=DEFAULT_INDEX_FLUSH_EVERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if requests is None:
        raise RuntimeError("requests is not available. Install it with: pip install requests")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    shard_size = args.shard_size

    out_dir = args.out_dir
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    index_jsonl = os.path.join(out_dir, "index.jsonl")
    stats_json = os.path.join(out_dir, "build_stats.json")

    print("Device:", device)
    print("JSONL:", args.jsonl)
    print("OUT_DIR:", out_dir)
    print("Shard size:", shard_size)
    print("Partial snapshots every kept:", args.snapshot_every_kept)
    print("Index:", index_jsonl)

    already, seen_ids = load_resume_state(index_jsonl)
    print("Resume kept (from index):", already)
    print("Resume unique ids:", len(seen_ids))

    shard_id = already // shard_size
    shard_offset = already % shard_size
    if shard_offset != 0:
        # A partial shard means the previous run died mid-shard; starting a fresh shard
        # keeps every index row's (shard_id, row_in_shard) pointing at a real row.
        print(
            "\nWarning: resume state indicates a partial shard. "
            "Starting a NEW shard to keep row mapping consistent."
        )
        shard_id += 1
        shard_offset = 0

    print("Starting shard id:", shard_id, "starting row_in_shard:", shard_offset)

    model, processor = load_blip(args.model, device=device)

    buf_feats: list[torch.Tensor] = []
    buf_records: list[dict[str, Any]] = []
    buf_index_rows: list[dict[str, Any]] = []

    kept = already
    scanned = 0

    skipped_bad_json = 0
    skipped_missing = 0
    skipped_download = 0
    skipped_embed = 0
    skipped_dup = 0
    download_ok = 0
    embed_ok = 0

    use_amp = device == "cuda"

    pbar = tqdm(total=None, desc="Scanning JSONL", dynamic_ncols=True)

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            pbar.update(1)

            if scanned % args.print_every_scanned == 0:
                print(
                    f"\n[progress] scanned={scanned:,} kept={kept:,} shard_id={shard_id} "
                    f"shard_buf={len(buf_records)} download_ok={download_ok:,} embed_ok={embed_ok:,} "
                    f"skip_dup={skipped_dup:,} skip_dl={skipped_download:,} "
                    f"skip_emb={skipped_embed:,} skip_miss={skipped_missing:,}"
                )

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_json += 1
                continue

            url = obj.get("image_url", None)
            if not isinstance(url, str) or not url.startswith("http"):
                skipped_missing += 1
                continue

            sample_id = stable_id_from_url(url)
            if sample_id in seen_ids:
                skipped_dup += 1
                continue

            original_caption = clean_text(obj.get("original_caption", ""))
            if original_caption is None:
                skipped_missing += 1
                continue

            img = download_image(
                url,
                timeout=args.timeout,
                retries=args.retries,
                min_image_side=args.min_image_side,
            )
            if img is None:
                skipped_download += 1
                continue
            download_ok += 1

            try:
                feat = encode_image_pooled(model, processor, img, device, use_amp)
            except Exception:
                skipped_embed += 1
                continue
            embed_ok += 1

            buf_feats.append(feat)
            buf_records.append(build_record(obj, sample_id, url, original_caption))

            row_in_shard = shard_offset + (len(buf_records) - 1)
            buf_index_rows.append(
                {"sample_id": sample_id, "shard_id": shard_id, "row_in_shard": row_in_shard}
            )

            seen_ids.add(sample_id)
            kept += 1

            if kept % args.print_every_kept == 0:
                print(
                    f"\n[kept] kept_embeddings={kept:,}  current_shard={shard_id}  "
                    f"shard_buf={len(buf_records)}"
                )

            if len(buf_index_rows) >= args.index_flush_every:
                append_index_rows(index_jsonl, buf_index_rows)
                buf_index_rows = []

            if (
                args.snapshot_every_kept > 0
                and kept % args.snapshot_every_kept == 0
                and len(buf_records) > 0
            ):
                snap_path = os.path.join(shards_dir, f"partial_shard_{shard_id:05d}.pt")
                feats_tensor = torch.stack(buf_feats, dim=0)
                write_shard(snap_path, feats_tensor, buf_records)
                print(
                    f"[snapshot] wrote {snap_path} rows={feats_tensor.size(0)} total_kept={kept:,}"
                )

            if shard_offset + len(buf_records) >= shard_size:
                shard_path = os.path.join(shards_dir, f"shard_{shard_id:05d}.pt")
                feats_tensor = torch.stack(buf_feats, dim=0)
                write_shard(shard_path, feats_tensor, buf_records)

                append_index_rows(index_jsonl, buf_index_rows)
                buf_index_rows = []

                print(
                    f"\n[shard] SAVED {shard_path} rows={feats_tensor.size(0)} total_kept={kept:,}"
                )

                shard_id += 1
                shard_offset = 0
                buf_feats, buf_records = [], []

            pbar.set_postfix(
                kept=kept,
                shard=shard_id,
                buf=len(buf_records),
                dl_ok=download_ok,
                emb_ok=embed_ok,
            )

    pbar.close()

    if len(buf_records) > 0:
        shard_path = os.path.join(shards_dir, f"shard_{shard_id:05d}.pt")
        feats_tensor = torch.stack(buf_feats, dim=0)
        write_shard(shard_path, feats_tensor, buf_records)
        append_index_rows(index_jsonl, buf_index_rows)
        print(f"\n[final] SAVED {shard_path} rows={feats_tensor.size(0)} total_kept={kept:,}")

    stats = {
        "jsonl_path": args.jsonl,
        "kept": kept,
        "scanned": scanned,
        "download_ok": download_ok,
        "embed_ok": embed_ok,
        "skipped_bad_json": skipped_bad_json,
        "skipped_missing_fields": skipped_missing,
        "skipped_download_fail": skipped_download,
        "skipped_embed_fail": skipped_embed,
        "skipped_duplicate": skipped_dup,
        "shard_size": shard_size,
        "store_dtype": str(STORE_DTYPE),
        "model": args.model,
        "device": device,
        "shards_dir": shards_dir,
        "index_jsonl": index_jsonl,
    }
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
