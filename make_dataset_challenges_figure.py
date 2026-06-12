from pathlib import Path
import random
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageStat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/tahiti/MARINE_DATASETS")
IMAGE_ROOT = ROOT / "FathomNet" / "images"
OUT_DIR = ROOT / "MARINE_RESULTS_DYNAMIC" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "dataset_challenges_fathomnet.png"
OUT_PDF = OUT_DIR / "dataset_challenges_fathomnet.pdf"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
random.seed(42)
np.random.seed(42)


# ============================================================
# HELPERS
# ============================================================

def list_classes():
    rows = []
    for d in sorted(IMAGE_ROOT.iterdir()):
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if files:
            rows.append({"class": d.name, "count": len(files), "files": files})
    rows = sorted(rows, key=lambda r: r["count"], reverse=True)
    return rows


def safe_open(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def crop_resize(img, size=160):
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def image_stats(path):
    img = safe_open(path)
    if img is None:
        return None

    small = img.resize((128, 128), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0

    gray = np.asarray(ImageOps.grayscale(small)).astype(np.float32) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # edge variance as a no-opencv blur proxy
    edges = ImageOps.grayscale(small).filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges).astype(np.float32) / 255.0
    edge_var = float(edge_arr.var())

    # rough channel imbalance / color cast proxy
    means = arr.reshape(-1, 3).mean(axis=0)
    color_cast = float(np.max(means) - np.min(means))

    return {
        "path": path,
        "brightness": brightness,
        "contrast": contrast,
        "edge_var": edge_var,
        "color_cast": color_cast,
    }


def label_text(s, max_len=22):
    s = s.replace("_", " ")
    if len(s) <= max_len:
        return s
    return s[:max_len - 1] + "…"


def draw_image_tile(ax, img, title, subtitle=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_edgecolor("0.25")
    ax.set_title(title, fontsize=8.5, pad=3)
    if subtitle:
        ax.text(
            0.5, -0.08, subtitle,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7.5
        )


def choose_one(files):
    files = list(files)
    random.shuffle(files)
    for p in files:
        img = safe_open(p)
        if img is not None:
            return p, img
    return None, None


# ============================================================
# COLLECT DATA
# ============================================================

classes = list_classes()

if not classes:
    raise RuntimeError(f"No images found in {IMAGE_ROOT}")

counts_df = pd.DataFrame([{"class": r["class"], "count": r["count"]} for r in classes])
counts_df.to_csv(OUT_DIR / "dataset_class_counts.csv", index=False)

all_files = []
for r in classes:
    for p in r["files"]:
        all_files.append((r["class"], p))

# sample subset for visual-quality diagnostics to keep script fast
sampled_for_stats = random.sample(all_files, min(len(all_files), 900))

stats = []
for cls, p in sampled_for_stats:
    st = image_stats(p)
    if st is not None:
        st["class"] = cls
        stats.append(st)

stats_df = pd.DataFrame(stats)

# ============================================================
# PANEL A: frequent + rare class examples
# ============================================================

frequent = classes[:4]
rare = sorted(classes, key=lambda r: r["count"])[:4]
class_panel = frequent + rare

class_examples = []
for r in class_panel:
    p, img = choose_one(r["files"])
    if img is not None:
        class_examples.append((r["class"], r["count"], crop_resize(img, 160)))

# ============================================================
# PANEL C: visual challenges
# ============================================================

challenge_examples = []

if not stats_df.empty:
    dark_row = stats_df.sort_values("brightness", ascending=True).iloc[0]
    low_contrast_row = stats_df.sort_values("contrast", ascending=True).iloc[0]
    blur_row = stats_df.sort_values("edge_var", ascending=True).iloc[0]
    color_row = stats_df.sort_values("color_cast", ascending=False).iloc[0]

    challenges = [
        ("dark image", dark_row),
        ("low contrast", low_contrast_row),
        ("blur / weak edges", blur_row),
        ("color cast", color_row),
    ]

    used = set()
    for name, row in challenges:
        p = Path(row["path"])
        if str(p) in used:
            continue
        img = safe_open(p)
        if img is not None:
            used.add(str(p))
            challenge_examples.append((name, row["class"], crop_resize(img, 160)))

# fallback
while len(challenge_examples) < 4:
    cls, p = random.choice(all_files)
    img = safe_open(p)
    if img is not None:
        challenge_examples.append(("visual variability", cls, crop_resize(img, 160)))

# ============================================================
# PANEL D: fine-grained similar groups
# ============================================================

similar_keywords = ["coral", "corallimorph", "coralliidae"]
similar_classes = [
    r for r in classes
    if any(k.lower() in r["class"].lower() for k in similar_keywords)
]
similar_classes = similar_classes[:6]

similar_examples = []
for r in similar_classes:
    p, img = choose_one(r["files"])
    if img is not None:
        similar_examples.append((r["class"], r["count"], crop_resize(img, 160)))

while len(similar_examples) < 6:
    r = random.choice(classes)
    p, img = choose_one(r["files"])
    if img is not None:
        similar_examples.append((r["class"], r["count"], crop_resize(img, 160)))

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(18, 11), dpi=180)
gs = fig.add_gridspec(
    nrows=3,
    ncols=12,
    height_ratios=[1.1, 1.05, 1.05],
    hspace=0.55,
    wspace=0.25
)

fig.suptitle(
    "FathomNet subset: image-level classes, class imbalance, and underwater visual variability",
    fontsize=17,
    fontweight="bold",
    y=0.985
)

# ------------------------------------------------------------
# A. class examples
# ------------------------------------------------------------

ax_title_a = fig.add_subplot(gs[0, :])
ax_title_a.axis("off")
ax_title_a.text(
    0.0, 1.05,
    "A. Examples from frequent and rare classes",
    fontsize=13,
    fontweight="bold",
    transform=ax_title_a.transAxes
)

for i, (cls, cnt, img) in enumerate(class_examples[:8]):
    ax = fig.add_subplot(gs[0, i + 2])
    draw_image_tile(
        ax,
        img,
        label_text(cls),
        f"n={cnt}"
    )

# ------------------------------------------------------------
# B. long-tail distribution
# ------------------------------------------------------------

ax_bar = fig.add_subplot(gs[1, :5])
plot_df = counts_df.sort_values("count", ascending=False).copy()
x = np.arange(len(plot_df))
ax_bar.bar(x, plot_df["count"].values)
ax_bar.set_yscale("log")
ax_bar.set_title("B. Long-tailed class distribution", fontsize=13, fontweight="bold")
ax_bar.set_ylabel("Images per class, log scale")
ax_bar.set_xlabel("Classes sorted by frequency")
ax_bar.grid(axis="y", alpha=0.25)
ax_bar.set_xticks([])

top_txt = "\n".join([
    f"{row['class'].replace('_',' ')}: {row['count']}"
    for _, row in plot_df.head(5).iterrows()
])
tail_txt = "\n".join([
    f"{row['class'].replace('_',' ')}: {row['count']}"
    for _, row in plot_df.tail(5).iterrows()
])

ax_bar.text(
    0.02, 0.95,
    "Largest classes:\n" + top_txt,
    transform=ax_bar.transAxes,
    ha="left", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6")
)

ax_bar.text(
    0.98, 0.95,
    "Smallest classes:\n" + tail_txt,
    transform=ax_bar.transAxes,
    ha="right", va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6")
)

# ------------------------------------------------------------
# C. visual challenges
# ------------------------------------------------------------

ax_c_title = fig.add_subplot(gs[1, 5:])
ax_c_title.axis("off")
ax_c_title.text(
    0.0, 1.03,
    "C. Examples selected by simple image statistics",
    fontsize=13,
    fontweight="bold",
    transform=ax_c_title.transAxes
)

for i, (problem, cls, img) in enumerate(challenge_examples[:4]):
    ax = fig.add_subplot(gs[1, 5 + i])
    draw_image_tile(
        ax,
        img,
        problem,
        label_text(cls, 18)
    )

# ------------------------------------------------------------
# D. fine-grained / visually close categories
# ------------------------------------------------------------

ax_d_title = fig.add_subplot(gs[2, :])
ax_d_title.axis("off")
ax_d_title.text(
    0.0, 1.05,
    "D. Visually close marine concepts in the coral / corallimorph group",
    fontsize=13,
    fontweight="bold",
    transform=ax_d_title.transAxes
)

start_col = 3
for i, (cls, cnt, img) in enumerate(similar_examples[:6]):
    ax = fig.add_subplot(gs[2, start_col + i])
    draw_image_tile(
        ax,
        img,
        label_text(cls, 20),
        f"n={cnt}"
    )

# ------------------------------------------------------------
# Footer note
# ------------------------------------------------------------

fig.text(
    0.5,
    0.015,
    "The figure illustrates why the task is treated as label-efficient underwater image recognition: "
    "classes are imbalanced, several concepts are visually similar, and image quality varies across underwater conditions.",
    ha="center",
    fontsize=9
)

fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
print("Class counts saved:")
print(OUT_DIR / "dataset_class_counts.csv")
