"""Download and extract the LLaVA-Pretrain (LAION-CC-SBU) image archive.

Produces ``<out-root>/images/<00000..>/...``. The download resumes into a ``.part`` file
that is only moved into place once complete, and the whole step is skipped when the
extraction directory already holds data.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from capara.common.paths import LLAVA_PRETRAIN_ROOT

LLAVA_PRETRAIN_IMAGES_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip"
)


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "python"})
    with urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}")
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

        tmp.replace(out_path)


def safe_extract_zip(zip_path: Path, dst_dir: Path) -> None:
    """Extract ``zip_path``, rejecting members that would escape ``dst_dir`` (zip-slip)."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        for member in members:
            extracted = (dst_dir / member.filename).resolve()
            if not str(extracted).startswith(str(dst_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")

        pbar = tqdm(total=len(members), desc=f"Extracting to {dst_dir}", dynamic_ncols=True)
        for member in members:
            zf.extract(member, dst_dir)
            pbar.update(1)
        pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=LLAVA_PRETRAIN_ROOT,
        help="Destination directory; images land in <out-root>/images.",
    )
    parser.add_argument("--url", type=str, default=LLAVA_PRETRAIN_IMAGES_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root: Path = args.out_root
    zip_path = out_root / "images.zip"
    extract_dir = out_root / "images"

    print("Download URL:", args.url)
    print("Zip path:", zip_path)
    print("Extract dir:", extract_dir)

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"[OK] {extract_dir} already exists and is non-empty; nothing to do.")
        return

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    if not zip_path.exists():
        download(args.url, zip_path)
    else:
        print(f"[OK] Zip already exists: {zip_path}")

    safe_extract_zip(zip_path, extract_dir)

    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    print(f"[DONE] Extracted. Top-level subdirs: {len(subdirs)}")
    if len(subdirs) == 0:
        print("[WARN] No subdirs found; extraction may not have worked as expected.")
        sys.exit(1)

    print("[DONE] LLaVA-Pretrain images ready at:", extract_dir)


if __name__ == "__main__":
    main()
