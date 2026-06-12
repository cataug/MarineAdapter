import os

# IMPORTANT: fixes RuntimeError: could not create a primitive on old/virtual CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import random
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch

# IMPORTANT: disable MKLDNN / oneDNN
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ============================================================
# CONFIG
# ============================================================

BASE = Path("/home/tahiti/MARINE_DATASETS")
DATA_ROOT = BASE / "FathomNet" / "images"
MODEL_ROOT = BASE / "MARINE_MODELS"
OUT_ROOT = BASE / "MARINE_EXPERIMENTS" / "marineov_clip_v002"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cpu"

MIN_IMAGES_PER_CLASS = 10
MAX_IMAGES_PER_CLASS = 100

# Use only stable small CLIP first
MODELS = {
    "clip_vit_base": MODEL_ROOT / "clip_vit_base",
}

PROMPT_TEMPLATES = {
    "plain": "{label}",
    "photo": "a photo of a {label}",
    "underwater": "an underwater photo of a {label}",
    "marine": "a marine image of a {label}",
    "rov": "an underwater ROV image of a {label}",
    "object": "a photo of the marine object {label}",
}

MAIN_PROMPT = "underwater"

SHOTS = [1, 2, 5, 10]
FS_SEEDS = [0, 1, 2]

ADAPTER_EPOCHS = 60
ADAPTER_LR = 1e-3
ADAPTER_WEIGHT_DECAY = 1e-4
ADAPTER_BATCH = 64
ADAPTER_HIDDENS = [64, 256]
ADAPTER_PATIENCE = 10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# UTILS
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)

def clean_label(x):
    return x.replace("_", " ").strip()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def save_report(y_true, y_pred, labels, out_path):
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(rep).T.to_csv(out_path)


# ============================================================
# BUILD DATASET
# ============================================================

print("=" * 100)
print("MarineOV CLIP experiment v002")
print("=" * 100)
print("DATA_ROOT :", DATA_ROOT)
print("MODEL_ROOT:", MODEL_ROOT)
print("OUT_ROOT  :", OUT_ROOT)
print("DEVICE    :", DEVICE)
print("MKLDNN    :", torch.backends.mkldnn.enabled)
print("=" * 100)

rows = []

for class_dir in sorted(DATA_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    label = clean_label(class_dir.name)

    files = []
    for ext in IMAGE_EXTS:
        files.extend(class_dir.glob(f"*{ext}"))
        files.extend(class_dir.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    if MAX_IMAGES_PER_CLASS is not None:
        files = files[:MAX_IMAGES_PER_CLASS]

    for p in files:
        rows.append({
            "path": str(p),
            "label": label,
            "folder": class_dir.name,
        })

df = pd.DataFrame(rows)

if len(df) == 0:
    raise RuntimeError("No images found.")

counts = df["label"].value_counts()
keep = counts[counts >= MIN_IMAGES_PER_CLASS].index.tolist()
df = df[df["label"].isin(keep)].copy().reset_index(drop=True)

labels = sorted(df["label"].unique())
label_to_id = {l: i for i, l in enumerate(labels)}
id_to_label = {i: l for l, i in label_to_id.items()}
df["y"] = df["label"].map(label_to_id).astype(int)

print("\nDataset:")
print("images :", len(df))
print("classes:", len(labels))
print(df["label"].value_counts().to_string())

df.to_csv(OUT_ROOT / "all_images.csv", index=False)

train_df, tmp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["y"],
    random_state=SEED,
)

val_df, test_df = train_test_split(
    tmp_df,
    test_size=0.50,
    stratify=tmp_df["y"],
    random_state=SEED,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(OUT_ROOT / "split_train.csv", index=False)
val_df.to_csv(OUT_ROOT / "split_val.csv", index=False)
test_df.to_csv(OUT_ROOT / "split_test.csv", index=False)

print("\nSplits:")
print("train:", len(train_df))
print("val  :", len(val_df))
print("test :", len(test_df))

with open(OUT_ROOT / "dataset_info.json", "w") as f:
    json.dump({
        "n_images": int(len(df)),
        "n_classes": int(len(labels)),
        "labels": labels,
        "counts": df["label"].value_counts().to_dict(),
    }, f, indent=2)


# ============================================================
# ENCODING
# ============================================================

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))

@torch.no_grad()
def encode_images_one_by_one(model, processor, split_df, model_name, split_name):
    cache = OUT_ROOT / f"emb_{model_name}_{split_name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["paths"].tolist()

    X = []
    y = []
    paths = split_df["path"].tolist()
    ys = split_df["y"].astype(int).tolist()

    model.eval()

    print(f"\nEncoding {model_name}/{split_name}: {len(paths)} images")

    for i, (p, yy) in enumerate(zip(paths, ys), 1):
        img = load_image(p)

        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        feats = model.get_image_features(pixel_values=pixel_values)
        feats = F.normalize(feats, dim=-1)

        X.append(feats.cpu().numpy()[0])
        y.append(yy)

        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.savez_compressed(cache, X=X, y=y, paths=np.asarray(paths, dtype=object))
    return X, y, paths

@torch.no_grad()
def encode_text(model, processor, template):
    texts = [template.format(label=l) for l in labels]
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32), texts

def zero_shot_predict(X, T):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    logits = Xn @ Tn.T
    pred = logits.argmax(axis=1)
    return pred, logits


# ============================================================
# BASELINES
# ============================================================

def linear_probe(X_train, y_train, X_test):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def sample_kshot(y, k, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idxs = []
    for c in sorted(np.unique(y)):
        cidx = np.where(y == c)[0]
        take = min(k, len(cidx))
        idxs.extend(rng.choice(cidx, size=take, replace=False).tolist())
    return np.asarray(sorted(idxs), dtype=np.int64)


# ============================================================
# OUR METHOD: MARINE ADAPTER
# ============================================================

class MarineAdapter(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        z = self.norm(x)
        z = F.gelu(self.fc1(z))
        z = self.fc2(z)
        out = x + self.scale * z
        return F.normalize(out, dim=-1)

def train_adapter(Xtr, ytr, Xval, yval, Xte, yte, T, hidden, seed):
    seed_everything(seed)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xval = torch.tensor(Xval, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)
    T = F.normalize(T, dim=-1)

    dim = Xtr.shape[1]
    net = MarineAdapter(dim, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=ADAPTER_LR, weight_decay=ADAPTER_WEIGHT_DECAY)

    best_f1 = -1
    best_state = None
    bad = 0

    n = Xtr.shape[0]

    for ep in range(1, ADAPTER_EPOCHS + 1):
        net.train()
        perm = torch.randperm(n)

        for st in range(0, n, ADAPTER_BATCH):
            idx = perm[st:st + ADAPTER_BATCH]
            xb = Xtr[idx]
            yb = ytr[idx]

            z = net(xb)
            logits = z @ T.T * 20.0
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            pred_val = (net(Xval) @ T.T * 20.0).argmax(1).numpy()

        val_f1 = f1_score(yval, pred_val, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= ADAPTER_PATIENCE:
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        pred_test = (net(Xte) @ T.T * 20.0).argmax(1).numpy()

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return pred_test, {
        "best_val_macro_f1": float(best_f1),
        "epochs": int(ep),
        "trainable_params": int(trainable),
    }


# ============================================================
# RUN
# ============================================================

all_results = []
t0 = time.time()

for model_name, model_path in MODELS.items():
    if not model_path.exists():
        print("SKIP missing model:", model_path)
        continue

    print("\n" + "=" * 100)
    print("MODEL:", model_name)
    print("PATH :", model_path)
    print("=" * 100)

    processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(DEVICE)
    model.eval()

    X_train, y_train, _ = encode_images_one_by_one(model, processor, train_df, model_name, "train")
    X_val, y_val, _ = encode_images_one_by_one(model, processor, val_df, model_name, "val")
    X_test, y_test, test_paths = encode_images_one_by_one(model, processor, test_df, model_name, "test")

    print("\nEmbeddings:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    text_cache = {}

    # zero-shot + prompt ablation
    for pname, template in PROMPT_TEMPLATES.items():
        T, texts = encode_text(model, processor, template)
        text_cache[pname] = T

        pred, logits = zero_shot_predict(X_test, T)
        m = metrics(y_test, pred)

        all_results.append({
            "model": model_name,
            "method": "zero_shot_clip",
            "setting": "prompt_ablation",
            "prompt": pname,
            "shots": 0,
            "seed": -1,
            "hidden": -1,
            "trainable_params": 0,
            **m,
        })

        save_report(y_test, pred, labels, OUT_ROOT / f"report_{model_name}_zero_{pname}.csv")

        pd.DataFrame({
            "path": test_paths,
            "true": [id_to_label[int(i)] for i in y_test],
            "pred": [id_to_label[int(i)] for i in pred],
        }).to_csv(OUT_ROOT / f"pred_{model_name}_zero_{pname}.csv", index=False)

        print(f"ZERO {pname:12s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    T_main = text_cache[MAIN_PROMPT]

    # full linear probe
    pred = linear_probe(X_train, y_train, X_test)
    m = metrics(y_test, pred)

    all_results.append({
        "model": model_name,
        "method": "linear_probe",
        "setting": "full_data",
        "prompt": "none",
        "shots": -1,
        "seed": -1,
        "hidden": -1,
        "trainable_params": -1,
        **m,
    })

    save_report(y_test, pred, labels, OUT_ROOT / f"report_{model_name}_linear_full.csv")
    print(f"LINEAR FULL acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    # full adapter
    for hidden in ADAPTER_HIDDENS:
        pred, extra = train_adapter(X_train, y_train, X_val, y_val, X_test, y_test, T_main, hidden, SEED)
        m = metrics(y_test, pred)

        all_results.append({
            "model": model_name,
            "method": "marine_adapter",
            "setting": "full_data",
            "prompt": MAIN_PROMPT,
            "shots": -1,
            "seed": SEED,
            "hidden": hidden,
            "trainable_params": extra["trainable_params"],
            "best_val_macro_f1": extra["best_val_macro_f1"],
            "epochs": extra["epochs"],
            **m,
        })

        save_report(y_test, pred, labels, OUT_ROOT / f"report_{model_name}_adapter_full_h{hidden}.csv")
        print(f"ADAPTER FULL h={hidden} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

    # few-shot
    for shot in SHOTS:
        for fs_seed in FS_SEEDS:
            idx = sample_kshot(y_train, shot, fs_seed)
            X_fs = X_train[idx]
            y_fs = y_train[idx]

            # few-shot linear probe
            try:
                pred = linear_probe(X_fs, y_fs, X_test)
                m = metrics(y_test, pred)

                all_results.append({
                    "model": model_name,
                    "method": "linear_probe",
                    "setting": "few_shot",
                    "prompt": "none",
                    "shots": shot,
                    "seed": fs_seed,
                    "hidden": -1,
                    "trainable_params": -1,
                    **m,
                })

                print(f"LINEAR {shot}-shot seed={fs_seed} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
            except Exception as e:
                print("LINEAR FAILED:", shot, fs_seed, repr(e))

            # few-shot adapter
            for hidden in ADAPTER_HIDDENS:
                try:
                    pred, extra = train_adapter(X_fs, y_fs, X_val, y_val, X_test, y_test, T_main, hidden, fs_seed)
                    m = metrics(y_test, pred)

                    all_results.append({
                        "model": model_name,
                        "method": "marine_adapter",
                        "setting": "few_shot",
                        "prompt": MAIN_PROMPT,
                        "shots": shot,
                        "seed": fs_seed,
                        "hidden": hidden,
                        "trainable_params": extra["trainable_params"],
                        "best_val_macro_f1": extra["best_val_macro_f1"],
                        "epochs": extra["epochs"],
                        **m,
                    })

                    print(f"ADAPTER {shot}-shot seed={fs_seed} h={hidden} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
                except Exception as e:
                    print("ADAPTER FAILED:", shot, fs_seed, hidden, repr(e))

    del model
    del processor


# ============================================================
# SAVE RESULTS
# ============================================================

res = pd.DataFrame(all_results)
res.to_csv(OUT_ROOT / "all_results.csv", index=False)

group_cols = ["model", "method", "setting", "prompt", "shots", "hidden"]
metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]

agg = res.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
agg.columns = ["_".join(x).strip("_") for x in agg.columns]
agg = agg.reset_index()
agg.to_csv(OUT_ROOT / "summary_results.csv", index=False)

best = agg.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False)
best.to_csv(OUT_ROOT / "best_results_sorted.csv", index=False)

show_cols = [
    "model", "method", "setting", "prompt", "shots", "hidden",
    "accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "weighted_f1_mean"
]
paper = best[show_cols].copy()

for c in ["accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "weighted_f1_mean"]:
    paper[c] = paper[c].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

paper.to_csv(OUT_ROOT / "paper_results.csv", index=False)
paper.head(30).to_latex(OUT_ROOT / "paper_results.tex", index=False, escape=True)

with open(OUT_ROOT / "best_results_top40.md", "w") as f:
    f.write(paper.head(40).to_markdown(index=False))

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
print("Saved to:", OUT_ROOT)
print("Time min:", round((time.time() - t0) / 60, 2))
print("\nTOP RESULTS:")
print(paper.head(20).to_string(index=False))
