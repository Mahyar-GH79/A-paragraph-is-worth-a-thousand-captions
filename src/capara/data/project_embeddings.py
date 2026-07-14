"""Project the 768-d image shards into BLIP's 256-d ITC space.

Reads the pooled vision features written by ``build_image_embeddings.py``
(``shards/``, ``[N, 768]``), applies BLIP's ``vision_proj``, L2-normalises, and writes
``shards_256/`` (``[N, 256]``). The ``records`` list is copied through unchanged, so row
``i`` still refers to the same sample in both directories.

Shards already present in the output directory are skipped, so the run resumes. The
projection is computed in fp32 for numerical stability and stored in fp16 to match the
input shard format.
"""

import argparse
import glob
import json
import os
from typing import Any

import torch
from tqdm import tqdm

from capara.common.blip import load_blip
from capara.common.paths import BLIP_MODEL, SHARDS_256_DIR, SHARDS_768_DIR

OUT_DTYPE = torch.float16
INPUT_DIM = 768
REPORT_NAME = "convert_768_to_256_report.json"


def load_shard(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected shard format: {path}")
    if "image_feats" not in obj or "records" not in obj:
        raise ValueError(f"Missing keys in shard: {path}")
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in-dir",
        type=str,
        default=str(SHARDS_768_DIR),
        help="Directory of 768-d input shards (shard_*.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SHARDS_256_DIR),
        help="Directory for the projected 256-d shards.",
    )
    parser.add_argument("--model", type=str, default=BLIP_MODEL)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help=f"Report JSON path (default: <parent of out-dir>/{REPORT_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dir = args.in_dir
    out_dir = args.out_dir
    report_path = args.report or os.path.join(os.path.dirname(out_dir), REPORT_NAME)

    os.makedirs(out_dir, exist_ok=True)

    print("Device:", device)
    print("IN :", in_dir)
    print("OUT:", out_dir)

    shard_paths = sorted(glob.glob(os.path.join(in_dir, "shard_*.pt")))
    if not shard_paths:
        raise RuntimeError(f"No shard_*.pt files found in {in_dir}")

    print("Found shards:", len(shard_paths))

    model, _ = load_blip(args.model, device=device)
    if not hasattr(model, "vision_proj"):
        raise AttributeError(
            "Model has no vision_proj. This script assumes a BLIP ITM model with vision_proj."
        )

    total_rows = 0
    report: dict[str, Any] = {
        "model": args.model,
        "in_shards_dir": in_dir,
        "out_shards_dir": out_dir,
        "device": device,
        "converted_shards": 0,
        "total_rows": 0,
        "notes": (
            "Converts stored pooled image feats (768) to BLIP ITC space (256) "
            "using vision_proj, then L2-normalizes."
        ),
    }

    pbar = tqdm(shard_paths, desc="Converting shards", dynamic_ncols=True)

    for in_path in pbar:
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        if os.path.exists(out_path):
            continue

        shard = load_shard(in_path)
        feats = shard["image_feats"]
        records = shard["records"]

        if not torch.is_tensor(feats):
            raise ValueError(f"image_feats is not a tensor in {in_path}")
        if feats.ndim != 2 or feats.size(1) != INPUT_DIM:
            raise ValueError(
                f"Expected [N,{INPUT_DIM}] feats, got {tuple(feats.shape)} in {in_path}"
            )

        feats = feats.to(torch.float32)

        with torch.no_grad():
            proj = model.vision_proj(feats.to(device))
            proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            proj = proj.detach().to("cpu").to(OUT_DTYPE)

        tmp = out_path + ".tmp"
        torch.save({"image_feats": proj, "records": records}, tmp)
        os.replace(tmp, out_path)

        total_rows += proj.size(0)
        report["converted_shards"] += 1
        report["total_rows"] = total_rows

        pbar.set_postfix(shards_done=report["converted_shards"], rows=total_rows)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print("Converted shards:", report["converted_shards"])
    print("Total rows:", report["total_rows"])
    print("Report:", report_path)
    print("Example output file:", os.path.join(out_dir, os.path.basename(shard_paths[0])))


if __name__ == "__main__":
    main()
