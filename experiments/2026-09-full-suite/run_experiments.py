import argparse
import copy
import gc
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch

# Avoid exhausting file descriptors after hundreds of DataLoader
# constructions in the full experimental suite.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from transformers import AutoModel


ROOT = Path("/home/tahiti/MARINE_DATASETS")
EXP = ROOT / "MARINE_EXPERIMENTS_V2"
MANIFESTS = EXP / "manifests"

DINO_PATH = Path("/home/tahiti/GreenToken/models/dinov2_small")

DATASETS = ["fathomnet", "aqua20"]
BACKBONES = ["resnet18", "dinov2_small"]

METHODS = [
    "linear_probe",
    "mlp_head",
    "marine_adapter",
    "last_block_ft",
    "full_finetune",
]

BUDGETS = ["001", "005", "010", "100"]
SEEDS = [42, 43, 44, 45, 46]

ADAPTER_DIM = 128
DROPOUT = 0.1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")


class MarineDataset(Dataset):
    def __init__(
        self,
        csv_path,
        dataset_name,
        label_map,
        transform,
    ):
        self.df = pd.read_csv(csv_path)
        self.dataset_name = dataset_name
        self.label_map = label_map
        self.transform = transform

        if dataset_name == "fathomnet":
            self.path_col = "roi_path"
            self.label_col = "category_id"
        else:
            self.path_col = "path"
            self.label_col = "label_id"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row[self.path_col]
        raw_label = int(row[self.label_col])

        with Image.open(path) as im:
            im = im.convert("RGB")
            image = self.transform(im)

        target = self.label_map[raw_label]

        return image, target


def make_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.80, 1.00),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tf, eval_tf


class BottleneckMLP(nn.Module):
    """
    Parameter-matched non-residual control for MarineAdapter.
    """
    def __init__(self, dim, bottleneck=128, dropout=0.1):
        super().__init__()

        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.drop(z)
        z = self.up(z)
        return self.norm(z)


class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=128, dropout=0.1):
        super().__init__()

        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.drop(z)
        z = self.up(z)

        return self.norm(x + z)


class MarineModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        method,
        n_classes,
        adapter_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.method = method

        if backbone_name == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )

            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name == "dinov2_small":
            self.backbone = AutoModel.from_pretrained(
                DINO_PATH,
                local_files_only=True,
            )

            feature_dim = int(self.backbone.config.hidden_size)

        else:
            raise ValueError(backbone_name)

        self.feature_dim = feature_dim

        if method == "mlp_head":
            self.feature_adapter = BottleneckMLP(
                feature_dim,
                adapter_dim,
                dropout,
            )

        elif method == "marine_adapter":
            self.feature_adapter = ResidualAdapter(
                feature_dim,
                adapter_dim,
                dropout,
            )

        else:
            self.feature_adapter = nn.Identity()

        self.classifier = nn.Linear(feature_dim, n_classes)

        self.configure_trainability()

    def configure_trainability(self):
        # Start frozen.
        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.method in [
            "linear_probe",
            "mlp_head",
            "marine_adapter",
        ]:
            pass

        elif self.method == "last_block_ft":

            if self.backbone_name == "resnet18":
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = True

            elif self.backbone_name == "dinov2_small":
                for p in self.backbone.encoder.layer[-1].parameters():
                    p.requires_grad = True

                if hasattr(self.backbone, "layernorm"):
                    for p in self.backbone.layernorm.parameters():
                        p.requires_grad = True

        elif self.method == "full_finetune":
            for p in self.backbone.parameters():
                p.requires_grad = True

        else:
            raise ValueError(self.method)

        # Heads always train.
        for p in self.feature_adapter.parameters():
            p.requires_grad = True

        for p in self.classifier.parameters():
            p.requires_grad = True

    def encode(self, x):

        if self.backbone_name == "resnet18":
            return self.backbone(x)

        out = self.backbone(pixel_values=x)

        # CLS token.
        return out.last_hidden_state[:, 0]

    def forward(self, x):
        z = self.encode(x)
        z = self.feature_adapter(z)
        return self.classifier(z)

    def set_training_mode(self):
        self.train()

        # Frozen backbone must not update BatchNorm/dropout state.
        if self.method in [
            "linear_probe",
            "mlp_head",
            "marine_adapter",
        ]:
            self.backbone.eval()

        elif self.method == "last_block_ft":
            self.backbone.eval()

            if self.backbone_name == "resnet18":
                self.backbone.layer4.train()

            else:
                self.backbone.encoder.layer[-1].train()

                if hasattr(self.backbone, "layernorm"):
                    self.backbone.layernorm.train()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def build_optimizer(model):
    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    head_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("backbone.")
    ]

    groups = []

    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": 3e-5,
        })

    if head_params:
        groups.append({
            "params": head_params,
            "lr": 3e-4,
        })

    return torch.optim.AdamW(
        groups,
        weight_decay=1e-4,
    )


def evaluate(model, loader, device, n_classes):
    model.eval()

    ys = []
    ps = []

    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                logits = model(images)

            preds = logits.argmax(dim=1)

            ys.extend(targets.cpu().numpy().tolist())
            ps.extend(preds.cpu().numpy().tolist())

    labels = list(range(n_classes))

    return {
        "accuracy": float(
            accuracy_score(ys, ps)
        ),

        "macro_f1": float(
            f1_score(
                ys,
                ps,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(ys, ps)
        ),
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
):
    model.set_training_mode()

    total_loss = 0.0
    total_n = 0

    amp_enabled = device.type == "cuda"

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = targets.size(0)

        total_loss += float(loss.detach()) * n
        total_n += n

    return total_loss / max(total_n, 1)


def get_paths(dataset, budget, seed):
    root = MANIFESTS / dataset

    train = root / f"train_budget{budget}_seed{seed}.csv"
    val = root / "val.csv"
    test = root / "test.csv"
    base_train = root / "train.csv"

    return train, val, test, base_train


def make_loaders(
    dataset,
    budget,
    seed,
    batch_size,
    workers,
):
    train_path, val_path, test_path, base_train = get_paths(
        dataset,
        budget,
        seed,
    )

    base_df = pd.read_csv(base_train)

    label_col = (
        "category_id"
        if dataset == "fathomnet"
        else "label_id"
    )

    raw_labels = sorted(
        int(x)
        for x in base_df[label_col].unique()
    )

    label_map = {
        raw: i
        for i, raw in enumerate(raw_labels)
    }

    train_tf, eval_tf = make_transforms()

    train_ds = MarineDataset(
        train_path,
        dataset,
        label_map,
        train_tf,
    )

    val_ds = MarineDataset(
        val_path,
        dataset,
        label_map,
        eval_tf,
    )

    test_ds = MarineDataset(
        test_path,
        dataset,
        label_map,
        eval_tf,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **common,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(label_map),
        len(train_ds),
        len(val_ds),
        len(test_ds),
        len(base_df),
    )


def cpu_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def refresh_global_results(run_root, output_csv):
    rows = []

    for p in run_root.rglob("result.json"):
        try:
            with open(p) as f:
                x = json.load(f)

            if x.get("status") == "complete":
                rows.append(x)

        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)

        sort_cols = [
            x for x in [
                "dataset",
                "backbone",
                "method",
                "budget",
                "seed",
            ]
            if x in df.columns
        ]

        if sort_cols:
            df = df.sort_values(sort_cols)

        df.to_csv(output_csv, index=False)


def run_one(
    dataset,
    backbone,
    method,
    budget,
    seed,
    epochs,
    batch_size,
    workers,
    run_root,
    keep_checkpoint=False,
):
    run_id = (
        f"{dataset}__{backbone}__{method}"
        f"__b{budget}__s{seed}"
    )

    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "result.json"

    if result_path.exists():
        try:
            with open(result_path) as f:
                old = json.load(f)

            if old.get("status") == "complete":
                print(
                    f"[SKIP] {run_id}",
                    flush=True,
                )
                return old

        except Exception:
            pass

    print()
    print("=" * 78, flush=True)
    print("RUN:", run_id, flush=True)
    print("=" * 78, flush=True)

    seed_everything(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        n_classes,
        n_train,
        n_val,
        n_test,
        n_full_train,
    ) = make_loaders(
        dataset,
        budget,
        seed,
        batch_size,
        workers,
    )

    print(
        f"N: train={n_train} val={n_val} "
        f"test={n_test} classes={n_classes}",
        flush=True,
    )

    model = MarineModel(
        backbone_name=backbone,
        method=method,
        n_classes=n_classes,
        adapter_dim=ADAPTER_DIM,
        dropout=DROPOUT,
    ).to(device)

    total_params, trainable_params = count_params(model)

    print(
        f"params total={total_params:,} "
        f"trainable={trainable_params:,}",
        flush=True,
    )

    optimizer = build_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    history = []

    best_epoch = -1
    best_val = -1.0
    best_state = None

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            n_classes,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            run_dir / "history.csv",
            index=False,
        )

        print(
            f"epoch {epoch:02d}/{epochs} "
            f"loss={train_loss:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)

    if best_state is None:
        raise RuntimeError("No best checkpoint was created")

    # Restore checkpoint selected ONLY by validation.
    model.load_state_dict(best_state)

    val_best = evaluate(
        model,
        val_loader,
        device,
        n_classes,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        n_classes,
    )

    runtime = time.time() - t0

    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    result = {
        "status": "complete",
        "run_id": run_id,

        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "budget": budget,
        "seed": seed,

        "epochs": epochs,
        "best_epoch": best_epoch,

        "adapter_dim": ADAPTER_DIM,
        "dropout": DROPOUT,

        "n_classes": n_classes,
        "n_train": n_train,
        "n_full_train": n_full_train,
        "realized_train_fraction":
            n_train / n_full_train,
        "n_val": n_val,
        "n_test": n_test,

        "total_params": total_params,
        "trainable_params": trainable_params,

        "val_accuracy":
            val_best["accuracy"],
        "val_macro_f1":
            val_best["macro_f1"],
        "val_balanced_accuracy":
            val_best["balanced_accuracy"],

        "test_accuracy":
            test_metrics["accuracy"],
        "test_macro_f1":
            test_metrics["macro_f1"],
        "test_balanced_accuracy":
            test_metrics["balanced_accuracy"],

        "runtime_sec": runtime,
        "peak_gpu_mem_gb": peak_mem,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    if keep_checkpoint:
        torch.save(
            best_state,
            run_dir / "best.pt",
        )

    print(
        f"BEST epoch={best_epoch} "
        f"val_f1={val_best['macro_f1']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"VRAM={peak_mem:.2f}GB",
        flush=True,
    )

    del best_state
    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del val_loader
    del test_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--keep-checkpoints",
        action="store_true",
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=1,
    )

    args = ap.parse_args()

    if args.smoke:
        jobs = [
            (
                "aqua20",
                "resnet18",
                "marine_adapter",
                "001",
                42,
            ),
            (
                "aqua20",
                "dinov2_small",
                "marine_adapter",
                "001",
                42,
            ),
        ]

        epochs = min(args.epochs, 2)

        run_root = EXP / "runs_smoke"
        results_csv = EXP / "smoke_results.csv"

    else:
        jobs = [
            (d, b, m, budget, seed)
            for d in DATASETS
            for b in BACKBONES
            for m in METHODS
            for budget in BUDGETS
            for seed in SEEDS
        ]

        epochs = args.epochs

        run_root = EXP / "runs"
        results_csv = EXP / "all_results.csv"

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("jobs:", len(jobs))
    print("epochs:", epochs)
    print("batch size:", args.batch_size)
    print("run root:", run_root)

    for i, job in enumerate(jobs, 1):
        d, b, m, budget, seed = job

        print(
            f"\n[{i}/{len(jobs)}] "
            f"{d} {b} {m} "
            f"budget={budget} seed={seed}",
            flush=True,
        )

        success = False

        for attempt in range(args.retries + 1):
            try:
                run_one(
                    dataset=d,
                    backbone=b,
                    method=m,
                    budget=budget,
                    seed=seed,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    run_root=run_root,
                    keep_checkpoint=args.keep_checkpoints,
                )

                success = True
                break

            except KeyboardInterrupt:
                print("\nSTOPPED BY CTRL+C")
                return

            except Exception as e:
                print(
                    f"\n[ERROR attempt "
                    f"{attempt + 1}/"
                    f"{args.retries + 1}]",
                    repr(e),
                    flush=True,
                )

                traceback.print_exc()

                fail_dir = run_root / (
                    f"{d}__{b}__{m}"
                    f"__b{budget}__s{seed}"
                )

                fail_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with open(
                    fail_dir / "failure.txt",
                    "a",
                ) as f:
                    f.write(
                        "\n\n"
                        + time.strftime("%Y-%m-%d %H:%M:%S")
                        + "\n"
                    )

                    traceback.print_exc(file=f)

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if attempt < args.retries:
                    print("retrying...", flush=True)
                    time.sleep(3)

        if not success:
            print(
                "[FAILED, moving to next run]",
                flush=True,
            )

        refresh_global_results(
            run_root,
            results_csv,
        )

    print()
    print("=" * 78)
    print("GRID COMPLETE")
    print("results:", results_csv)
    print("=" * 78)


if __name__ == "__main__":
    main()
