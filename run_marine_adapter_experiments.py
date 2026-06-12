import os
import csv
import json
import time
import math
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch

# GPU is available, but cuDNN may fail on this machine with:
# RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
# Keep CUDA, disable cuDNN only.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Keep CPU backends quiet/safe if CPU fallback happens.
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass


# Disable oneDNN/MKLDNN CPU kernels because they crash on this VM CPU.
# This fixes: RuntimeError('could not create a primitive')
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

try:
    import torchvision
    from torchvision import transforms
    from torchvision import models
    TORCHVISION_OK = True
except Exception as e:
    TORCHVISION_OK = False
    TORCHVISION_ERROR = repr(e)


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def human_time(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def macro_f1_from_lists(y_true, y_pred, num_classes):
    eps = 1e-12
    f1s = []
    for c in range(num_classes):
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        if sum(yt == c for yt in y_true) > 0:
            f1s.append(f1)

    if not f1s:
        return 0.0
    return float(sum(f1s) / len(f1s))


def balanced_accuracy_from_lists(y_true, y_pred, num_classes):
    eps = 1e-12
    recalls = []
    for c in range(num_classes):
        total = sum(yt == c for yt in y_true)
        if total == 0:
            continue
        correct = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        recalls.append(correct / (total + eps))
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


# ============================================================
# Dataset
# ============================================================

class FolderMarineDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, str(path)


def collect_samples(image_root, min_images_per_class=5, max_classes=0, max_images_per_class=0):
    image_root = Path(image_root)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_dirs = []
    for p in sorted(image_root.iterdir()):
        if not p.is_dir():
            continue
        imgs = []
        for x in p.rglob("*"):
            if x.is_file() and x.suffix.lower() in exts:
                imgs.append(x)
        if len(imgs) >= min_images_per_class:
            class_dirs.append((p.name, sorted(imgs)))

    class_dirs = sorted(class_dirs, key=lambda x: len(x[1]), reverse=True)

    if max_classes and max_classes > 0:
        class_dirs = class_dirs[:max_classes]

    class_names = sorted([c for c, _ in class_dirs])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples_by_class = defaultdict(list)

    for cname, imgs in class_dirs:
        if max_images_per_class and max_images_per_class > 0:
            imgs = imgs[:max_images_per_class]
        label = class_to_idx[cname]
        for img in imgs:
            samples_by_class[label].append(img)

    return class_names, class_to_idx, samples_by_class


def make_splits(samples_by_class, train_ratio=0.70, val_ratio=0.15, seed=42):
    rng = random.Random(seed)

    train_samples = []
    val_samples = []
    test_samples = []

    for label, paths in samples_by_class.items():
        paths = list(paths)
        rng.shuffle(paths)

        n = len(paths)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 3 else 0

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        if len(test_paths) == 0 and len(val_paths) > 1:
            test_paths = [val_paths.pop()]

        for p in train_paths:
            train_samples.append((p, label))
        for p in val_paths:
            val_samples.append((p, label))
        for p in test_paths:
            test_samples.append((p, label))

    return train_samples, val_samples, test_samples


def subsample_train_budget(train_samples, budget, num_classes, seed=42):
    """
    budget: float in (0,1], e.g. 0.01, 0.05, 0.10, 1.0
    class-preserving subsampling
    """
    if budget >= 0.999:
        return list(train_samples)

    rng = random.Random(seed)

    by_class = defaultdict(list)
    for p, y in train_samples:
        by_class[y].append((p, y))

    selected = []

    for c in range(num_classes):
        items = by_class.get(c, [])
        if not items:
            continue
        rng.shuffle(items)

        k = max(1, int(math.ceil(len(items) * budget)))
        selected.extend(items[:k])

    rng.shuffle(selected)
    return selected


# ============================================================
# Transforms
# ============================================================

def get_transforms(img_size=224):
    if not TORCHVISION_OK:
        raise RuntimeError(f"torchvision import failed: {TORCHVISION_ERROR}")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.04,
            )
        ], p=0.7),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, eval_tf


# ============================================================
# Models
# ============================================================

class SmallCNN(nn.Module):
    """
    Baseline: train from scratch.
    """
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.classifier(f)


def load_resnet18_backbone(pretrained=True):
    """
    Returns feature extractor and feature dimension.
    """
    if not TORCHVISION_OK:
        raise RuntimeError(f"torchvision import failed: {TORCHVISION_ERROR}")

    try:
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
    except Exception as e:
        print("[WARN] Could not load pretrained ResNet18:", repr(e))
        print("[WARN] Falling back to random ResNet18.")
        model = models.resnet18(weights=None)

    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class FrozenLinearProbe(nn.Module):
    """
    Baseline: frozen ResNet18 backbone + linear classifier.
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = load_resnet18_backbone(pretrained=pretrained)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
        return self.classifier(f)


class MarineAdapterModel(nn.Module):
    """
    Ours: frozen backbone + residual marine adapter + classifier.

    features = backbone(image)
    adapted = features + Adapter(features)
    logits = classifier(adapted)
    """
    def __init__(self, num_classes, adapter_dim=128, dropout=0.10, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = load_resnet18_backbone(pretrained=pretrained)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, feat_dim),
        )

        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)

        delta = self.adapter(f)
        z = self.norm(f + delta)
        return self.classifier(z)


class FinetuneResNet18(nn.Module):
    """
    Baseline: full fine-tuning.
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = load_resnet18_backbone(pretrained=pretrained)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f)


def build_model(method, num_classes, adapter_dim=128, dropout=0.10, pretrained=True):
    if method == "scratch_cnn":
        return SmallCNN(num_classes)

    if method == "linear_probe":
        return FrozenLinearProbe(num_classes, pretrained=pretrained)

    if method == "marine_adapter":
        return MarineAdapterModel(
            num_classes=num_classes,
            adapter_dim=adapter_dim,
            dropout=dropout,
            pretrained=pretrained,
        )

    if method == "finetune_resnet18":
        return FinetuneResNet18(num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Training / evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device, amp=False):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), y) * bs
        total_n += bs

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_acc / max(1, total_n),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    y_true = []
    y_pred = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (pred == y).float().mean().item() * bs
        total_n += bs

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    acc = total_acc / max(1, total_n)
    macro_f1 = macro_f1_from_lists(y_true, y_pred, num_classes)
    bal_acc = balanced_accuracy_from_lists(y_true, y_pred, num_classes)

    return {
        "loss": total_loss / max(1, total_n),
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "n": total_n,
    }


def make_optimizer(model, method, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]

    if method in ["linear_probe", "marine_adapter"]:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if method == "finetune_resnet18":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if method == "scratch_cnn":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_single_experiment(
    method,
    budget,
    seed,
    class_names,
    class_to_idx,
    train_samples_full,
    val_samples,
    test_samples,
    args,
    device,
):
    set_seed(seed)

    num_classes = len(class_names)

    train_samples = subsample_train_budget(
        train_samples_full,
        budget=budget,
        num_classes=num_classes,
        seed=seed,
    )

    train_tf, eval_tf = get_transforms(args.img_size)

    train_ds = FolderMarineDataset(train_samples, class_to_idx, transform=train_tf)
    val_ds = FolderMarineDataset(val_samples, class_to_idx, transform=eval_tf)
    test_ds = FolderMarineDataset(test_samples, class_to_idx, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(
        method=method,
        num_classes=num_classes,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = make_optimizer(
        model=model,
        method=method,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    amp = bool(args.amp and torch.cuda.is_available())

    best_val_macro_f1 = -1.0
    best_state = None
    best_epoch = -1

    start = time.time()

    print("\n" + "=" * 90)
    print(f"METHOD={method} | BUDGET={budget} | SEED={seed}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} classes={num_classes}")
    print(f"params_total={count_total_params(model):,}")
    print(f"params_trainable={count_trainable_params(model):,}")
    print("=" * 90)

    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, amp=amp)
        va = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_macro_f1": va["macro_f1"],
            "val_balanced_acc": va["balanced_acc"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f} "
            f"macroF1={va['macro_f1']:.4f} balAcc={va['balanced_acc']:.4f}"
        )

        if va["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = va["macro_f1"]
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val": va,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    te = evaluate(model, test_loader, device, num_classes)

    elapsed = time.time() - start

    result = {
        "method": method,
        "budget": budget,
        "budget_percent": int(round(budget * 100)),
        "seed": seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "num_classes": num_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "adapter_dim": args.adapter_dim if method == "marine_adapter" else "",
        "dropout": args.dropout if method == "marine_adapter" else "",
        "pretrained": not args.no_pretrained,
        "params_total": count_total_params(model),
        "params_trainable": count_trainable_params(model),
        "test_loss": te["loss"],
        "test_acc": te["acc"],
        "test_macro_f1": te["macro_f1"],
        "test_balanced_acc": te["balanced_acc"],
        "time_sec": elapsed,
        "time_human": human_time(elapsed),
    }

    print("\nRESULT:")
    for k, v in result.items():
        print(f"{k}: {v}")

    run_name = (
        f"{method}"
        f"_budget{int(round(budget * 100)):03d}"
        f"_seed{seed}"
        f"_ad{args.adapter_dim}"
    )

    run_dir = Path(args.out_dir) / "runs" / run_name
    ensure_dir(run_dir)

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "result": result,
        },
        run_dir / "model.pt",
    )

    return result


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_root",
        type=str,
        default="/home/tahiti/MARINE_DATASETS/FathomNet/images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/tahiti/MARINE_DATASETS/MARINE_RESULTS",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--min_images_per_class", type=int, default=5)
    parser.add_argument("--max_classes", type=int, default=30)
    parser.add_argument("--max_images_per_class", type=int, default=300)

    parser.add_argument(
        "--methods",
        type=str,
        default="scratch_cnn,linear_probe,marine_adapter,finetune_resnet18",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.01,0.05,0.10,1.0",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
    )

    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.out_dir)

    print("=" * 90)
    print("MARINE ADAPTER EXPERIMENTS")
    print("=" * 90)
    print("image_root:", args.image_root)
    print("out_dir:", args.out_dir)
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("torchvision_ok:", TORCHVISION_OK)
    if not TORCHVISION_OK:
        print("torchvision_error:", TORCHVISION_ERROR)
        raise RuntimeError("torchvision is required for this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    class_names, class_to_idx, samples_by_class = collect_samples(
        image_root=args.image_root,
        min_images_per_class=args.min_images_per_class,
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
    )

    print("\nClasses:", len(class_names))
    for i, c in enumerate(class_names[:50]):
        print(f"{i:03d}: {c} | n={len(samples_by_class[class_to_idx[c]])}")

    if len(class_names) < 2:
        raise RuntimeError("Need at least 2 classes.")

    train_samples, val_samples, test_samples = make_splits(
        samples_by_class,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=123,
    )

    print("\nSplit:")
    print("train:", len(train_samples))
    print("val  :", len(val_samples))
    print("test :", len(test_samples))

    split_info = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "n_classes": len(class_names),
        "n_train_full": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "args": vars(args),
    }

    with open(Path(args.out_dir) / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    print("\nMethods:", methods)
    print("Budgets:", budgets)
    print("Seeds:", seeds)

    all_results = []

    for seed in seeds:
        for budget in budgets:
            for method in methods:
                try:
                    result = run_single_experiment(
                        method=method,
                        budget=budget,
                        seed=seed,
                        class_names=class_names,
                        class_to_idx=class_to_idx,
                        train_samples_full=train_samples,
                        val_samples=val_samples,
                        test_samples=test_samples,
                        args=args,
                        device=device,
                    )
                    all_results.append(result)

                    results_csv = Path(args.out_dir) / "all_results.csv"
                    with open(results_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
                        writer.writeheader()
                        writer.writerows(all_results)

                    print("\nSaved:", results_csv)

                except Exception as e:
                    print("\nFAILED RUN")
                    print("method:", method)
                    print("budget:", budget)
                    print("seed:", seed)
                    print("error:", repr(e))

                    fail_record = {
                        "method": method,
                        "budget": budget,
                        "budget_percent": int(round(budget * 100)),
                        "seed": seed,
                        "status": "failed",
                        "error": repr(e),
                    }

                    fail_path = Path(args.out_dir) / "failed_runs.jsonl"
                    with open(fail_path, "a") as f:
                        f.write(json.dumps(fail_record) + "\n")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    if all_results:
        print("\nFinal results:")
        for r in sorted(all_results, key=lambda x: (x["budget"], x["method"])):
            print(
                f"{r['method']:20s} "
                f"budget={r['budget_percent']:3d}% "
                f"acc={r['test_acc']:.4f} "
                f"macroF1={r['test_macro_f1']:.4f} "
                f"balAcc={r['test_balanced_acc']:.4f} "
                f"trainable={r['params_trainable']}"
            )

        print("\nSaved to:", Path(args.out_dir) / "all_results.csv")
    else:
        print("No successful results.")


if __name__ == "__main__":
    main()