from pathlib import Path
from datetime import datetime
import shutil
import os

SRC = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2")
DST = Path("/home/tahiti/MarineAdapter_git/experiments/2026-09-full-suite")

START = datetime(2026, 9, 1, 0, 0, 0).timestamp()
END   = datetime(2026, 10, 1, 0, 0, 0).timestamp()

MAX_BYTES = 50 * 1024 * 1024

# Anything that is clearly model/data weight or temporary bulk.
EXCLUDED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".avi",
    ".mp4",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
}

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".cache",
    "checkpoints",
    "checkpoint",
    "weights",
    "models",
    "model",
    "images",
    "raw",
    "data",
    "datasets",
}

# Explicitly useful file classes.
KEEP_SUFFIXES = {
    ".py",
    ".sh",
    ".txt",
    ".log",
    ".csv",
    ".json",
    ".md",
    ".tex",
    ".bib",
    ".yaml",
    ".yml",
    ".toml",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".tsv",
}

copied = []
skipped_large = []
skipped_type = []
skipped_excluded = []

DST.mkdir(parents=True, exist_ok=True)

for src in SRC.rglob("*"):
    if not src.is_file():
        continue

    rel = src.relative_to(SRC)

    parts_lower = {x.lower() for x in rel.parts[:-1]}

    if parts_lower & EXCLUDED_DIR_NAMES:
        skipped_excluded.append(rel)
        continue

    suffix = src.suffix.lower()

    if suffix in EXCLUDED_SUFFIXES:
        skipped_excluded.append(rel)
        continue

    # Only September 2026 material.
    mt = src.stat().st_mtime

    if not (START <= mt < END):
        continue

    size = src.stat().st_size

    if size > MAX_BYTES:
        skipped_large.append((rel, size))
        continue

    if suffix not in KEEP_SUFFIXES:
        skipped_type.append(rel)
        continue

    dst = DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)

    copied.append((rel, size))

print("=" * 90)
print("SEPTEMBER EXPORT COMPLETE")
print("=" * 90)

print("Copied files:", len(copied))
print("Total copied: %.2f MB" % (
    sum(x[1] for x in copied) / 1024 / 1024
))

print("\nLargest copied files:")
for rel, size in sorted(
    copied,
    key=lambda x: x[1],
    reverse=True
)[:30]:
    print(f"{size/1024/1024:9.2f} MB  {rel}")

print("\nSkipped >50 MB:", len(skipped_large))
for rel, size in sorted(
    skipped_large,
    key=lambda x: x[1],
    reverse=True
)[:30]:
    print(f"{size/1024/1024:9.2f} MB  {rel}")

print("\nSkipped excluded data/model files:", len(skipped_excluded))
print("Skipped unknown extensions:", len(skipped_type))

manifest = DST / "EXPORT_MANIFEST.txt"

with manifest.open("w") as f:
    f.write("MarineAdapter September 2026 export\n")
    f.write("Source: /home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2\n")
    f.write("Window: 2026-09-01 <= mtime < 2026-10-01\n\n")

    f.write("COPIED FILES\n")
    f.write("=" * 80 + "\n")

    for rel, size in sorted(copied):
        f.write(f"{size:12d}  {rel}\n")

    f.write("\nSKIPPED LARGE FILES\n")
    f.write("=" * 80 + "\n")

    for rel, size in sorted(skipped_large):
        f.write(f"{size:12d}  {rel}\n")

print("\nManifest:")
print(manifest)
