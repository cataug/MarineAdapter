import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path("/home/tahiti/MARINE_DATASETS")
EXP = ROOT / "MARINE_EXPERIMENTS_V2"
HERE = EXP

sys.path.insert(0, str(HERE))
import run_experiments as core


MAIN_MANIFESTS = EXP / "manifests"
SUITE_ROOT = EXP / "FULL_SUITE"

SENS_ROOT = EXP / "manifests_sensitivity"
SHOTS_ROOT = EXP / "manifests_shots"

SEEDS5 = [42, 43, 44, 45, 46]
SEEDS3 = [42, 43, 44]

BUDGETS = ["001", "005", "010", "100"]

PRETRAINED_METHODS = [
    "linear_probe",
    "mlp_head",
    "marine_adapter",
    "last_block_ft",
    "full_finetune",
]

ORIGINAL_MODEL = core.MarineModel
ORIGINAL_OPTIMIZER = core.build_optimizer


# ============================================================
# Scratch CNN baseline
# ============================================================

class ScratchCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.method = "scratch_cnn"
        self.backbone_name = "scratch_cnn"

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.feature_adapter = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.classifier(z)

    def set_training_mode(self):
        self.train()


def model_factory(
    backbone_name,
    method,
    n_classes,
    adapter_dim=128,
    dropout=0.1,
):
    if method == "scratch_cnn":
        return ScratchCNN(n_classes)

    return ORIGINAL_MODEL(
        backbone_name=backbone_name,
        method=method,
        n_classes=n_classes,
        adapter_dim=adapter_dim,
        dropout=dropout,
    )


def optimizer_factory(model):
    if getattr(model, "method", None) == "scratch_cnn":
        return torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )

    return ORIGINAL_OPTIMIZER(model)


core.MarineModel = model_factory
core.build_optimizer = optimizer_factory


# ============================================================
# Capture final evaluation predictions
# ============================================================

LAST_EVAL = {}


def evaluate_capture(
    model,
    loader,
    device,
    n_classes,
):
    global LAST_EVAL

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

            ys.extend(
                targets.detach().cpu().numpy().tolist()
            )

            ps.extend(
                preds.detach().cpu().numpy().tolist()
            )

    labels = list(range(n_classes))

    LAST_EVAL = {
        "y_true": ys,
        "y_pred": ps,
    }

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


core.evaluate = evaluate_capture


# ============================================================
# Helpers
# ============================================================

def run_id(d, b, m, budget, seed):
    return (
        f"{d}__{b}__{m}"
        f"__b{budget}__s{seed}"
    )


def decorate_result(
    run_root,
    d,
    b,
    m,
    budget,
    seed,
    metadata,
):
    rid = run_id(d, b, m, budget, seed)

    p = run_root / rid / "result.json"

    if not p.exists():
        return

    with open(p) as f:
        r = json.load(f)

    r.update(metadata)

    with open(p, "w") as f:
        json.dump(r, f, indent=2)


def save_test_predictions(
    run_root,
    manifest_root,
    d,
    b,
    m,
    budget,
    seed,
):
    global LAST_EVAL

    if not LAST_EVAL:
        return

    rid = run_id(d, b, m, budget, seed)

    run_dir = run_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    test_csv = (
        Path(manifest_root)
        / d
        / "test.csv"
    )

    if not test_csv.exists():
        return

    df = pd.read_csv(test_csv)

    y_true = LAST_EVAL.get("y_true", [])
    y_pred = LAST_EVAL.get("y_pred", [])

    if len(df) != len(y_true):
        print(
            "WARNING prediction length mismatch:",
            len(df),
            len(y_true),
            flush=True,
        )
        return

    df = df.copy()

    df["y_true_internal"] = y_true
    df["y_pred_internal"] = y_pred

    df.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )


def collect_results():
    rows = []

    if not SUITE_ROOT.exists():
        return

    for p in SUITE_ROOT.rglob("result.json"):
        try:
            with open(p) as f:
                r = json.load(f)

            if r.get("status") == "complete":
                rows.append(r)

        except Exception:
            pass

    if not rows:
        return

    df = pd.DataFrame(rows)

    sort_cols = [
        x
        for x in [
            "stage",
            "dataset",
            "backbone",
            "method",
            "adapter_dim",
            "budget",
            "split_seed",
            "shot_k",
            "seed",
        ]
        if x in df.columns
    ]

    if sort_cols:
        df = df.sort_values(sort_cols)

    shard = os.environ.get("MARINE_SHARD")
    if shard is None:
        name = "all_results_full.csv"
    else:
        name = f"all_results_full_shard_{shard}.csv"

    df.to_csv(
        SUITE_ROOT / name,
        index=False,
    )

    print(
        "completed total:",
        len(df),
        flush=True,
    )


def execute_job(
    stage,
    manifest_root,
    dataset,
    backbone,
    method,
    budget,
    seed,
    epochs,
    batch_size,
    workers,
    retries,
    adapter_dim=128,
    metadata=None,
):
    metadata = dict(metadata or {})

    # Job dictionaries use descriptive names; keep short aliases
    # internally for the existing runner code.
    d = dataset
    b = backbone
    m = method

    metadata["stage"] = stage
    metadata["adapter_dim"] = adapter_dim

    core.MANIFESTS = Path(manifest_root)
    core.ADAPTER_DIM = adapter_dim

    run_root = (
        SUITE_ROOT
        / stage
    )

    if stage == "width":
        run_root = (
            run_root
            / f"r{adapter_dim}"
        )

    if stage == "sensitivity":
        split_seed = metadata["split_seed"]

        run_root = (
            run_root
            / f"split_{split_seed}"
        )

    if stage == "shots":
        shot_k = metadata["shot_k"]

        run_root = (
            run_root
            / f"shot_{shot_k}"
        )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = False

    for attempt in range(retries + 1):
        try:
            LAST_EVAL.clear()

            result = core.run_one(
                dataset=d,
                backbone=b,
                method=m,
                budget=budget,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                workers=workers,
                run_root=run_root,
                keep_checkpoint=False,
            )

            decorate_result(
                run_root,
                d,
                b,
                m,
                budget,
                seed,
                metadata,
            )

            # If run was freshly evaluated, LAST_EVAL contains
            # the final test predictions. If it was skipped,
            # prediction file from previous run remains.
            if LAST_EVAL:
                save_test_predictions(
                    run_root,
                    manifest_root,
                    d,
                    b,
                    m,
                    budget,
                    seed,
                )

            success = True
            break

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(
                f"[ERROR] attempt "
                f"{attempt + 1}/{retries + 1}:",
                repr(e),
                flush=True,
            )

            traceback.print_exc()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if attempt < retries:
                print(
                    "retrying in 5 seconds...",
                    flush=True,
                )
                time.sleep(5)

    if not success:
        print(
            "[FAILED] moving to next job",
            flush=True,
        )

    collect_results()


# ============================================================
# Budget helper
# ============================================================

def stratified_budget(
    train,
    label_col,
    fraction,
    seed,
):
    if fraction >= 1.0:
        return train.copy()

    selected = []

    for c in sorted(train[label_col].unique()):
        idx = train.index[
            train[label_col] == c
        ].to_numpy().copy()

        rng = np.random.default_rng(
            seed * 100000
            + int(c) * 97
            + 17
        )

        rng.shuffle(idx)

        n = max(
            1,
            math.ceil(
                fraction * len(idx)
            )
        )

        selected.extend(
            idx[:n].tolist()
        )

    return train.loc[
        sorted(set(selected))
    ].copy()


# ============================================================
# Split-sensitivity manifests
# ============================================================

def prepare_sensitivity():
    src = (
        ROOT
        / "FathomNet_FGVC2025"
        / "splits_full"
        / "manifest_full.csv"
    )

    df = pd.read_csv(src)

    X = np.zeros(
        len(df),
        dtype=np.uint8,
    )

    classes = sorted(
        df["category_id"].unique()
    )

    class_map = {
        c: i
        for i, c in enumerate(classes)
    }

    y = (
        df["category_id"]
        .map(class_map)
        .to_numpy()
    )

    groups = df["group_id"].to_numpy()

    for split_seed in [123, 456, 789]:
        out = (
            SENS_ROOT
            / f"split_{split_seed}"
            / "fathomnet"
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        sgkf = StratifiedGroupKFold(
            n_splits=7,
            shuffle=True,
            random_state=split_seed,
        )

        folds = np.full(
            len(df),
            -1,
            dtype=np.int8,
        )

        for f, (_, idx) in enumerate(
            sgkf.split(
                X,
                y,
                groups,
            )
        ):
            folds[idx] = f

        C = len(classes)
        K = 7

        counts = np.zeros(
            (C, K),
            dtype=np.int32,
        )

        np.add.at(
            counts,
            (y, folds),
            1,
        )

        totals = counts.sum(axis=1)
        ideal = totals / K
        sizes = np.bincount(
            folds,
            minlength=K,
        )

        best = None

        for tf in range(K):
            for vf in range(K):
                if tf == vf:
                    continue

                tc = counts[:, tf]
                vc = counts[:, vf]

                min_eval = int(
                    min(
                        tc.min(),
                        vc.min(),
                    )
                )

                deviation = float(
                    np.mean(
                        np.abs(tc - ideal)
                        + np.abs(vc - ideal)
                    )
                )

                size_diff = int(
                    abs(
                        sizes[tf]
                        - sizes[vf]
                    )
                )

                score = (
                    min_eval,
                    -deviation,
                    -size_diff,
                )

                if (
                    best is None
                    or score > best["score"]
                ):
                    best = {
                        "score": score,
                        "test_fold": tf,
                        "val_fold": vf,
                    }

        z = df.copy()
        z["fold_sensitivity"] = folds

        z["split"] = "train"

        z.loc[
            z["fold_sensitivity"]
            == best["test_fold"],
            "split"
        ] = "test"

        z.loc[
            z["fold_sensitivity"]
            == best["val_fold"],
            "split"
        ] = "val"

        train = z[
            z["split"] == "train"
        ].copy()

        val = z[
            z["split"] == "val"
        ].copy()

        test = z[
            z["split"] == "test"
        ].copy()

        train.to_csv(
            out / "train.csv",
            index=False,
        )

        val.to_csv(
            out / "val.csv",
            index=False,
        )

        test.to_csv(
            out / "test.csv",
            index=False,
        )

        for seed in SEEDS3:
            ten = stratified_budget(
                train,
                "category_id",
                0.10,
                seed,
            )

            ten.to_csv(
                out
                / f"train_budget010_seed{seed}.csv",
                index=False,
            )

            train.to_csv(
                out
                / f"train_budget100_seed{seed}.csv",
                index=False,
            )

        tr_groups = set(
            train["group_id"]
        )

        va_groups = set(
            val["group_id"]
        )

        te_groups = set(
            test["group_id"]
        )

        assert not (
            tr_groups & va_groups
        )

        assert not (
            tr_groups & te_groups
        )

        assert not (
            va_groups & te_groups
        )

        print(
            f"sensitivity split={split_seed}: "
            f"train={len(train)} "
            f"val={len(val)} "
            f"test={len(test)} "
            f"worst_eval="
            f"{best['score'][0]}",
            flush=True,
        )


# ============================================================
# K-shot manifests
# ============================================================

def prepare_shots():
    src = (
        MAIN_MANIFESTS
        / "fathomnet"
    )

    out = (
        SHOTS_ROOT
        / "fathomnet"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        src / "train.csv"
    )

    shutil.copy2(
        src / "train.csv",
        out / "train.csv",
    )

    shutil.copy2(
        src / "val.csv",
        out / "val.csv",
    )

    shutil.copy2(
        src / "test.csv",
        out / "test.csv",
    )

    for seed in SEEDS3:
        for k in [1, 2, 5, 10]:
            selected = []

            for c in sorted(
                train["category_id"].unique()
            ):
                idx = train.index[
                    train["category_id"] == c
                ].to_numpy().copy()

                rng = np.random.default_rng(
                    seed * 100000
                    + int(c) * 97
                    + 991
                )

                rng.shuffle(idx)

                if len(idx) < k:
                    raise RuntimeError(
                        f"class {c} has only "
                        f"{len(idx)} samples"
                    )

                selected.extend(
                    idx[:k].tolist()
                )

            subset = train.loc[
                sorted(selected)
            ].copy()

            budget = f"shot{k:02d}"

            subset.to_csv(
                out
                / f"train_budget{budget}_seed{seed}.csv",
                index=False,
            )

            print(
                f"shot seed={seed} "
                f"k={k}: "
                f"N={len(subset)}",
                flush=True,
            )


# ============================================================
# Majority baseline
# ============================================================

def majority_baselines():
    rows = []

    for d in ["fathomnet", "aqua20"]:
        base = MAIN_MANIFESTS / d

        label_col = (
            "category_id"
            if d == "fathomnet"
            else "label_id"
        )

        test = pd.read_csv(
            base / "test.csv"
        )

        all_labels = sorted(
            pd.read_csv(
                base / "train.csv"
            )[label_col].unique()
        )

        for budget in BUDGETS:
            for seed in SEEDS5:
                tr = pd.read_csv(
                    base
                    / f"train_budget{budget}_seed{seed}.csv"
                )

                majority = int(
                    tr[label_col]
                    .value_counts()
                    .idxmax()
                )

                y = test[label_col].astype(int).to_numpy()

                p = np.full(
                    len(y),
                    majority,
                    dtype=int,
                )

                rows.append({
                    "stage": "majority",
                    "dataset": d,
                    "backbone": "none",
                    "method": "majority",
                    "budget": budget,
                    "seed": seed,
                    "n_train": len(tr),
                    "n_test": len(test),
                    "test_accuracy": float(
                        accuracy_score(y, p)
                    ),
                    "test_macro_f1": float(
                        f1_score(
                            y,
                            p,
                            labels=all_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "test_balanced_accuracy": float(
                        balanced_accuracy_score(
                            y,
                            p,
                        )
                    ),
                    "majority_class": majority,
                })

    out = pd.DataFrame(rows)

    SUITE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        SUITE_ROOT
        / "majority_baselines.csv",
        index=False,
    )

    print(
        "majority baseline rows:",
        len(out),
        flush=True,
    )


# ============================================================
# Jobs
# ============================================================

def build_jobs(stages):
    jobs = []

    if "core" in stages:
        for d in [
            "aqua20",
            "fathomnet",
        ]:
            for b in [
                "resnet18",
                "dinov2_small",
            ]:
                for m in PRETRAINED_METHODS:
                    for budget in BUDGETS:
                        for seed in SEEDS5:
                            jobs.append({
                                "stage": "core",
                                "manifest_root":
                                    MAIN_MANIFESTS,
                                "dataset": d,
                                "backbone": b,
                                "method": m,
                                "budget": budget,
                                "seed": seed,
                                "adapter_dim": 128,
                                "metadata": {},
                            })

    if "scratch" in stages:
        for d in [
            "aqua20",
            "fathomnet",
        ]:
            for budget in BUDGETS:
                for seed in SEEDS5:
                    jobs.append({
                        "stage": "scratch",
                        "manifest_root":
                            MAIN_MANIFESTS,
                        "dataset": d,
                        "backbone": "scratch_cnn",
                        "method": "scratch_cnn",
                        "budget": budget,
                        "seed": seed,
                        "adapter_dim": 128,
                        "metadata": {},
                    })

    if "width" in stages:
        for r in [64, 256]:
            for budget in [
                "010",
                "100",
            ]:
                for seed in SEEDS5:
                    jobs.append({
                        "stage": "width",
                        "manifest_root":
                            MAIN_MANIFESTS,
                        "dataset": "fathomnet",
                        "backbone": "resnet18",
                        "method": "marine_adapter",
                        "budget": budget,
                        "seed": seed,
                        "adapter_dim": r,
                        "metadata": {
                            "width_ablation": True,
                        },
                    })

    if "sensitivity" in stages:
        for split_seed in [
            123,
            456,
            789,
        ]:
            manifest_root = (
                SENS_ROOT
                / f"split_{split_seed}"
            )

            for m in [
                "linear_probe",
                "marine_adapter",
                "full_finetune",
            ]:
                for budget in [
                    "010",
                    "100",
                ]:
                    for seed in SEEDS3:
                        jobs.append({
                            "stage":
                                "sensitivity",
                            "manifest_root":
                                manifest_root,
                            "dataset":
                                "fathomnet",
                            "backbone":
                                "resnet18",
                            "method": m,
                            "budget": budget,
                            "seed": seed,
                            "adapter_dim": 128,
                            "metadata": {
                                "split_seed":
                                    split_seed,
                            },
                        })

    if "shots" in stages:
        for k in [
            1,
            2,
            5,
            10,
        ]:
            budget = f"shot{k:02d}"

            for m in PRETRAINED_METHODS:
                for seed in SEEDS3:
                    jobs.append({
                        "stage": "shots",
                        "manifest_root":
                            SHOTS_ROOT,
                        "dataset":
                            "fathomnet",
                        "backbone":
                            "resnet18",
                        "method": m,
                        "budget": budget,
                        "seed": seed,
                        "adapter_dim": 128,
                        "metadata": {
                            "shot_k": k,
                        },
                    })

    return jobs


# ============================================================
# Preflight
# ============================================================

def run_preflight(args):
    jobs = [
        {
            "stage": "preflight",
            "manifest_root": MAIN_MANIFESTS,
            "dataset": "aqua20",
            "backbone": "resnet18",
            "method": "linear_probe",
            "budget": "001",
            "seed": 42,
            "adapter_dim": 128,
            "metadata": {},
        },
        {
            "stage": "preflight",
            "manifest_root": MAIN_MANIFESTS,
            "dataset": "aqua20",
            "backbone": "resnet18",
            "method": "mlp_head",
            "budget": "001",
            "seed": 42,
            "adapter_dim": 128,
            "metadata": {},
        },
        {
            "stage": "preflight",
            "manifest_root": MAIN_MANIFESTS,
            "dataset": "aqua20",
            "backbone": "dinov2_small",
            "method": "last_block_ft",
            "budget": "001",
            "seed": 42,
            "adapter_dim": 128,
            "metadata": {},
        },
        {
            "stage": "preflight",
            "manifest_root": MAIN_MANIFESTS,
            "dataset": "aqua20",
            "backbone": "dinov2_small",
            "method": "full_finetune",
            "budget": "001",
            "seed": 42,
            "adapter_dim": 128,
            "metadata": {},
        },
        {
            "stage": "preflight",
            "manifest_root": MAIN_MANIFESTS,
            "dataset": "aqua20",
            "backbone": "scratch_cnn",
            "method": "scratch_cnn",
            "budget": "001",
            "seed": 42,
            "adapter_dim": 128,
            "metadata": {},
        },
    ]

    print(
        "PREFLIGHT JOBS:",
        len(jobs),
        flush=True,
    )

    for i, j in enumerate(jobs, 1):
        print(
            f"\n[PREFLIGHT {i}/{len(jobs)}]",
            j["backbone"],
            j["method"],
            flush=True,
        )

        execute_job(
            epochs=1,
            batch_size=args.batch_size,
            workers=args.workers,
            retries=args.retries,
            **j,
        )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "core",
            "scratch",
            "width",
            "sensitivity",
            "shots",
        ],
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
        "--retries",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--preflight",
        action="store_true",
    )

    args = ap.parse_args()

    SUITE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.preflight:
        run_preflight(args)
        return

    stages = args.stages

    if "all" in stages:
        stages = [
            "core",
            "scratch",
            "width",
            "sensitivity",
            "shots",
        ]

    skip_prepare = os.environ.get("MARINE_SKIP_PREP") == "1"

    if not skip_prepare:
        print(
            "Preparing auxiliary manifests...",
            flush=True,
        )

        if "sensitivity" in stages:
            prepare_sensitivity()

        if "shots" in stages:
            prepare_shots()

        majority_baselines()

    jobs = build_jobs(stages)

    num_shards = int(os.environ.get("MARINE_NUM_SHARDS", "1"))
    shard_id = int(os.environ.get("MARINE_SHARD", "0"))

    if num_shards > 1:
        all_jobs = jobs
        jobs = [
            job
            for i, job in enumerate(all_jobs)
            if i % num_shards == shard_id
        ]

        print(
            f"SHARD {shard_id}/{num_shards}: "
            f"{len(jobs)}/{len(all_jobs)} jobs",
            flush=True,
        )

    print()
    print("=" * 78)
    print("FULL SUITE")
    print("=" * 78)
    print("stages:", stages)
    print("training jobs:", len(jobs))
    print("epochs:", args.epochs)
    print("batch size:", args.batch_size)
    print("workers:", args.workers)
    print("=" * 78)

    for i, j in enumerate(jobs, 1):
        print()
        print(
            f"[{i}/{len(jobs)}]",
            j["stage"],
            j["dataset"],
            j["backbone"],
            j["method"],
            "budget=" + str(j["budget"]),
            "seed=" + str(j["seed"]),
            "r=" + str(j["adapter_dim"]),
            flush=True,
        )

        try:
            execute_job(
                epochs=args.epochs,
                batch_size=args.batch_size,
                workers=args.workers,
                retries=args.retries,
                **j,
            )

        except KeyboardInterrupt:
            print(
                "\nSTOPPED BY CTRL+C",
                flush=True,
            )
            collect_results()
            return

    collect_results()

    print()
    print("=" * 78)
    print("FULL SUITE COMPLETE")
    print(
        "results:",
        SUITE_ROOT / "all_results_full.csv",
    )
    print(
        "majority:",
        SUITE_ROOT / "majority_baselines.csv",
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
