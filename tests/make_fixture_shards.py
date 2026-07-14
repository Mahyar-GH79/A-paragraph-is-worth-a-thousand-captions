"""Build tiny synthetic embedding shards so the pipeline can be exercised on CPU.

The shards mimic the real ones from ``capara.data.project_embeddings``: 256-d
L2-normalised fp16 image features plus the CC3M annotation records.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

EMBED_DIM = 256


def build(out_dir: Path, n_shards: int, rows_per_shard: int, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    for shard_id in range(n_shards):
        feats = F.normalize(
            torch.randn(rows_per_shard, EMBED_DIM, generator=generator), dim=-1
        ).to(torch.float16)

        records = []
        for row in range(rows_per_shard):
            uid = shard_id * rows_per_shard + row
            records.append(
                {
                    "sample_id": f"fixture{uid:06d}",
                    "original_caption": f"a photo of object number {uid}",
                    "positive_captions": [
                        f"a picture showing object {uid}, view {v}" for v in range(5)
                    ],
                    "hard_negative_captions": [
                        f"a picture of a completely different object {uid + 1000}, view {v}"
                        for v in range(5)
                    ],
                    "paragraph": (
                        f"This image depicts object number {uid} resting on a plain "
                        f"surface. The object dominates the frame and is lit from the "
                        f"left, casting a soft shadow to the right. Behind it the "
                        f"background falls away into an even, neutral tone."
                    ),
                    "llama_score": 8,
                }
            )

        torch.save(
            {"image_feats": feats, "records": records},
            out_dir / f"shard_{shard_id:05d}.pt",
        )

    print(f"Wrote {n_shards} shards x {rows_per_shard} rows to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--rows-per-shard", type=int, default=32)
    args = parser.parse_args()
    build(args.out_dir, args.shards, args.rows_per_shard)


if __name__ == "__main__":
    main()
