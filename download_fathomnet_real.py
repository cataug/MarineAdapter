import json
import os
import time
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm
from fathomnet.api import images, boundingboxes


ROOT = Path("/home/tahiti/MARINE_DATASETS/FathomNet").resolve()
IMG_ROOT = ROOT / "images"
META_ROOT = ROOT / "metadata"

IMG_ROOT.mkdir(parents=True, exist_ok=True)
META_ROOT.mkdir(parents=True, exist_ok=True)


KEYWORDS = [
    "fish",
    "shark",
    "ray",
    "crab",
    "squid",
    "octopus",
    "jelly",
    "coral",
    "sponge",
    "urchin",
    "star",
]

MAX_CONCEPTS = 20
MAX_IMAGES_PER_CONCEPT = 100


def safe_name(s):
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


print("ROOT:", ROOT)
print("Checking FathomNet API...")

concepts = boundingboxes.find_concepts()
print("Total concepts:", len(concepts))

selected = []
for c in concepts:
    cl = str(c).lower()
    if any(k in cl for k in KEYWORDS):
        selected.append(c)

selected = selected[:MAX_CONCEPTS]

print("\nSelected concepts:")
for c in selected:
    print(" -", c)

if not selected:
    raise RuntimeError("No matching concepts found.")

all_records = []

for concept in selected:
    print("\n=== concept:", concept, "===")

    try:
        recs = images.find_by_concept(concept)
    except Exception as e:
        print("[SKIP] API error:", concept, e)
        continue

    print("found images:", len(recs))
    recs = recs[:MAX_IMAGES_PER_CONCEPT]

    concept_dir = IMG_ROOT / safe_name(concept)
    concept_dir.mkdir(parents=True, exist_ok=True)

    ok = 0

    for r in tqdm(recs):
        try:
            url = getattr(r, "url", None)
            uuid = getattr(r, "uuid", None)

            if not url or not uuid:
                continue

            ext = os.path.splitext(url.split("?")[0])[-1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                ext = ".jpg"

            out_path = concept_dir / f"{uuid}{ext}"

            if not out_path.exists() or out_path.stat().st_size == 0:
                urlretrieve(url, out_path)

            boxes = getattr(r, "boundingBoxes", [])

            all_records.append(
                {
                    "concept_query": str(concept),
                    "uuid": str(uuid),
                    "url": str(url),
                    "image_path": str(out_path),
                    "width": getattr(r, "width", None),
                    "height": getattr(r, "height", None),
                    "n_boxes": len(boxes) if boxes is not None else 0,
                }
            )

            ok += 1
            time.sleep(0.03)

        except Exception as e:
            print("[FAIL]", concept, getattr(r, "uuid", "no_uuid"), e)

    print("downloaded:", ok)

meta_path = META_ROOT / "downloaded_records.json"

with open(meta_path, "w") as f:
    json.dump(all_records, f, indent=2)

print("\nDONE")
print("Downloaded records:", len(all_records))
print("Images dir:", IMG_ROOT)
print("Metadata:", meta_path)