from pathlib import Path
import math
import pandas as pd
import numpy as np

ROOT = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2")

CONFIG = {
    "fathomnet": {
        "group": "group_id",
        "label": "category_id",
    },
    "aqua20": {
        "group": "group_id",
        "label": "label_id",
    },
}

BUDGETS = [0.01, 0.05, 0.10, 1.00]
SEEDS = [42, 43, 44, 45, 46]

summary = []

for dataset, cfg in CONFIG.items():

    base = ROOT / "manifests" / dataset
    train = pd.read_csv(base / "train.csv")

    group_col = cfg["group"]
    label_col = cfg["label"]

    print(f"\n===== {dataset.upper()} =====")
    print("full train rows:", len(train))
    print("groups:", train[group_col].nunique())
    print("classes:", train[label_col].nunique())

    # For every class, list source groups containing that class.
    class_groups = {
        c: np.array(
            sorted(
                train.loc[
                    train[label_col] == c,
                    group_col
                ].unique()
            )
        )
        for c in sorted(train[label_col].unique())
    }

    for seed in SEEDS:

        rng = np.random.default_rng(seed)

        # One deterministic shuffled order per class.
        shuffled = {}

        for c, groups in class_groups.items():
            x = groups.copy()
            rng.shuffle(x)
            shuffled[c] = x

        previous_groups = set()

        for frac in BUDGETS:

            if frac == 1.0:
                selected_groups = set(train[group_col].unique())

            else:
                selected_groups = set(previous_groups)

                # Select approximately frac of source groups
                # associated with each class.
                for c, groups in shuffled.items():

                    n = max(
                        1,
                        math.ceil(frac * len(groups))
                    )

                    selected_groups.update(
                        groups[:n].tolist()
                    )

            subset = train[
                train[group_col].isin(selected_groups)
            ].copy()

            # Guarantee nested budgets:
            # 1% ⊂ 5% ⊂ 10% ⊂ 100%.
            previous_groups = set(selected_groups)

            missing = (
                set(train[label_col].unique()) -
                set(subset[label_col].unique())
            )

            if missing:
                raise RuntimeError(
                    f"{dataset} seed={seed} frac={frac}: "
                    f"missing classes {sorted(missing)}"
                )

            tag = {
                0.01: "001",
                0.05: "005",
                0.10: "010",
                1.00: "100",
            }[frac]

            out = (
                ROOT /
                "manifests" /
                dataset /
                f"train_budget{tag}_seed{seed}.csv"
            )

            subset.to_csv(out, index=False)

            realized = len(subset) / len(train)

            print(
                f"seed={seed} budget={tag}: "
                f"rows={len(subset)} "
                f"groups={len(selected_groups)} "
                f"realized={realized:.4f}"
            )

            summary.append({
                "dataset": dataset,
                "seed": seed,
                "nominal_budget": frac,
                "rows": len(subset),
                "groups": len(selected_groups),
                "full_train_rows": len(train),
                "realized_row_fraction": realized,
                "n_classes": subset[label_col].nunique(),
            })

pd.DataFrame(summary).to_csv(
    ROOT / "budget_summary.csv",
    index=False
)

print("\nSAVED:", ROOT / "budget_summary.csv")
