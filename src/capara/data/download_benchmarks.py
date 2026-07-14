"""Download and unpack the COCO 2017 train split (images + caption annotations).

Produces::

    <coco-root>/images/train2017/*.jpg
    <coco-root>/annotations/captions_train2017.json

Downloads and extractions are skipped when the target already exists, so the script is
safe to re-run.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from capara.common.paths import COCO_ROOT

COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path, timeout: float = 30.0) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()


def unzip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def download_coco2017(coco_root: Path, timeout: float = 30.0) -> None:
    """Download the COCO 2017 *train* split and keep only the caption annotations."""
    img_dir = coco_root / "images"
    ann_dir = coco_root / "annotations"
    ensure_dir(img_dir)
    ensure_dir(ann_dir)

    train_zip = coco_root / "downloads" / "train2017.zip"
    ann_zip = coco_root / "downloads" / "annotations_trainval2017.zip"

    download_file(COCO_TRAIN2017_URL, train_zip, timeout=timeout)
    download_file(COCO_ANN_URL, ann_zip, timeout=timeout)

    if not (img_dir / "train2017").exists():
        print("Extracting train2017.zip")
        unzip(train_zip, img_dir)

    if not (ann_dir / "captions_train2017.json").exists():
        print("Extracting annotations_trainval2017.zip")
        tmp_extract = coco_root / "tmp_extract"
        unzip(ann_zip, tmp_extract)

        src = tmp_extract / "annotations" / "captions_train2017.json"
        if not src.exists():
            raise RuntimeError("Could not find captions_train2017.json inside extracted annotations")

        (ann_dir / "captions_train2017.json").write_bytes(src.read_bytes())
        shutil.rmtree(tmp_extract, ignore_errors=True)

    print("COCO 2017 (train) ready at:", coco_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=COCO_ROOT,
        help="Destination directory for the COCO 2017 train split.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.coco_root)
    download_coco2017(args.coco_root, timeout=args.timeout)


if __name__ == "__main__":
    main()
