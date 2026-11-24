"""Extract the HuggingFace parquet shards into plain image/mask folders.

Usage:
    python extract.py --src data --dst data/cfd
"""
import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

SHARDS = {
    "train": ["train0.parquet", "train1.parquet", "train2.parquet"],
    "test": ["test.parquet"],
}


def dump(split: str, files, dst: Path) -> int:
    img_dir, msk_dir = dst / split / "images", dst / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        table = pq.read_table(f)
        for row in tqdm(table.to_pylist(), desc=f"{split}/{f.name}", ncols=80):
            name = Path(row["pixel_values"]["path"]).stem
            img = Image.open(io.BytesIO(row["pixel_values"]["bytes"])).convert("RGB")
            msk = Image.open(io.BytesIO(row["label"]["bytes"])).convert("L")
            img.save(img_dir / f"{name}.jpg", quality=95)
            msk.save(msk_dir / f"{name}.png")  # PNG keeps the mask lossless
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="data/cfd")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    for split, names in SHARDS.items():
        n = dump(split, [src / n for n in names], dst)
        print(f"{split}: {n} pairs -> {dst / split}")


if __name__ == "__main__":
    main()
