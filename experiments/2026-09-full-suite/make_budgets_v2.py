from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2")

CONFIG = {
    "fathomnet": {
        "label": "category_id",
    },
    "aqua20": {
        "label": "label_id",
    },
}

BUDGETS = [0.01, 0.05, 0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

TAGS = {
    0.01: "001",
    0.05: "005",
    0.10: "010",
    1.00: "100",
}

summary = []

for dataset, cfg in CONFIG.items():

    base = ROOT / "manifests" / dataset
    train = pd.read_csv(base / "train.csv")
    label_col = cfg["label"]

    print(f"\n===== {dataset.upper()} =====", flush=True)
    print("full train:", len(train))
    print("classes   :", train[label_col].nunique())

    classes = sorted(train[label_col].unique())

    for seed in SEEDS:

        # One fixed random order per class.
        # Therefore budgets are automatically nested.
        shuffled = {}

        for c in classes:
            idx = train.index[
                train[label_col] == c
            ].to_numpy().copy()

            # Different deterministic permutation for each
            # dataset/seed/class without relying on Python hash().
            rng = np.random.default_rng(
                seed * 100000 + int(c) * 97 + 17
            )

            rng.shuffle(idx)
            shuffled[c] = idx

        previous = set()

        for frac in BUDGETS:

            if frac == 1.0:
                selected = set(train.index)

            else:
                selected = set()

                for c in classes:
                    idx = shuffled[c]

                    n = max(
                        1,
                        math.ceil(frac * len(idx))
                    )

                    selected.update(idx[:n].tolist())

                # Safety: enforce nesting explicitly.
                selected |= previous

            previous = set(selected)

            subset = train.loc[
                sorted(selected)
            ].copy()

            n_classes = subset[label_col].nunique()

            if n_classes != len(classes):
                raise RuntimeError(
                    f"{dataset} seed={seed} budget={frac}: "
                    f"only {n_classes}/{len(classes)} classes"
                )

            tag = TAGS[frac]

            out = base / f"train_budget{tag}_seed{seed}.csv"
            subset.to_csv(out, index=False)

            realized = len(subset) / len(train)

            min_class = (
                subset[label_col]
                .value_counts()
                .min()
            )

            max_class = (
                subset[label_col]
                .value_counts()
                .max()
            )

            print(
                f"seed={seed} budget={tag}: "
                f"rows={len(subset):5d} "
                f"realized={realized:.4f} "
                f"class_min={min_class} "
                f"class_max={max_class}",
                flush=True
            )

            summary.append({
                "dataset": dataset,
                "seed": seed,
                "nominal_budget": frac,
                "rows": len(subset),
                "full_train_rows": len(train),
                "realized_row_fraction": realized,
                "n_classes": n_classes,
                "min_class_samples": int(min_class),
                "max_class_samples": int(max_class),
            })

summary = pd.DataFrame(summary)

summary.to_csv(
    ROOT / "budget_summary_v2.csv",
    index=False
)

print("\n=== SAVED ===")
print(ROOT / "budget_summary_v2.csv")
