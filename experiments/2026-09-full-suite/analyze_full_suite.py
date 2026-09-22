from pathlib import Path
from collections import Counter
from itertools import product
import json

import numpy as np
import pandas as pd


ROOT = Path("MARINE_EXPERIMENTS_V2/FULL_SUITE")
OUT = ROOT / "STATS"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "test_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
]

BOOT = 20000
RNG = np.random.default_rng(20260922)


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

rows = []

for p in ROOT.rglob("result.json"):
    try:
        r = json.loads(p.read_text())
        r["_path"] = str(p)
        rows.append(r)
    except Exception as e:
        print("BAD JSON:", p, e)

df = pd.DataFrame(rows)

print("=" * 90)
print("FULL SUITE AUDIT")
print("=" * 90)
print("TOTAL:", len(df))
print("STATUS:", Counter(df["status"]))
print("STAGE:", Counter(df["stage"]))

assert len(df) == 579, f"Expected 579, got {len(df)}"
assert (df["status"] == "complete").all(), "Some runs are not complete"

# Do not mix technical preflight into scientific results.
science = df[df["stage"] != "preflight"].copy()

for c in ["budget", "dataset", "backbone", "method", "stage"]:
    if c in science.columns:
        science[c] = science[c].astype(str)

print("SCIENTIFIC RUNS:", len(science))
assert len(science) == 574


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def bootstrap_mean_ci(x, n_boot=BOOT):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return x[0], x[0]

    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize(d, by):
    out = []

    for key, g in d.groupby(by, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(by, key))
        row["n"] = len(g)

        for metric in METRICS:
            if metric not in g.columns:
                continue

            x = pd.to_numeric(g[metric], errors="coerce").dropna().values

            if len(x) == 0:
                continue

            lo, hi = bootstrap_mean_ci(x)

            row[f"{metric}_mean"] = np.mean(x)
            row[f"{metric}_sd"] = np.std(x, ddof=1) if len(x) > 1 else 0.0
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi

        out.append(row)

    return pd.DataFrame(out)


def exact_signflip_p(diff):
    """
    Exact two-sided paired randomization/sign-flip test.
    With n=5, minimum attainable two-sided p is 0.0625.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]

    n = len(d)

    if n == 0:
        return np.nan

    obs = abs(d.mean())

    if n <= 20:
        vals = []
        for signs in product([-1.0, 1.0], repeat=n):
            vals.append(abs(np.mean(d * np.asarray(signs))))

        vals = np.asarray(vals)
        return float(np.mean(vals >= obs - 1e-15))

    # Monte-Carlo fallback, though current per-cell tests have n=5.
    signs = RNG.choice(
        [-1.0, 1.0],
        size=(100000, n),
        replace=True,
    )

    vals = np.abs((signs * d).mean(axis=1))

    return float(
        (np.sum(vals >= obs - 1e-15) + 1)
        / (len(vals) + 1)
    )


def bootstrap_delta_ci(diff, n_boot=BOOT):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return np.nan, np.nan

    if len(d) == 1:
        return d[0], d[0]

    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)

    valid = np.where(np.isfinite(pvals))[0]
    if len(valid) == 0:
        return out

    pv = pvals[valid]
    order = np.argsort(pv)
    m = len(pv)

    adjusted_sorted = np.empty(m)
    running = 0.0

    for rank, oi in enumerate(order):
        q = (m - rank) * pv[oi]
        running = max(running, q)
        adjusted_sorted[rank] = min(running, 1.0)

    for rank, oi in enumerate(order):
        out[valid[oi]] = adjusted_sorted[rank]

    return out


def fmt(x, n=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{n}f}"


# ---------------------------------------------------------------------
# CORE: 2 datasets x 2 backbones x 5 methods x 4 budgets x 5 seeds
# ---------------------------------------------------------------------

core = science[science["stage"] == "core"].copy()

print("\n" + "=" * 90)
print("CORE")
print("=" * 90)
print("rows:", len(core))
assert len(core) == 400

core_summary = summarize(
    core,
    ["dataset", "backbone", "method", "budget"],
)

core_summary = core_summary.sort_values(
    ["dataset", "backbone", "budget", "method"]
)

core_summary.to_csv(
    OUT / "core_summary.csv",
    index=False,
)


print("\nCORE MACRO-F1: mean ± SD [95% bootstrap CI]\n")

for (dataset, backbone), g in core_summary.groupby(
    ["dataset", "backbone"]
):
    print(f"\n### {dataset} / {backbone}")

    for budget in ["001", "005", "010", "100"]:
        z = g[g["budget"] == budget]

        print(f"\n  budget={budget}")

        for _, r in z.iterrows():
            print(
                f"    {r['method']:18s} "
                f"{r['test_macro_f1_mean']:.4f} ± "
                f"{r['test_macro_f1_sd']:.4f} "
                f"[{r['test_macro_f1_ci_low']:.4f}, "
                f"{r['test_macro_f1_ci_high']:.4f}]"
            )


# ---------------------------------------------------------------------
# Paired statistical comparisons
# ---------------------------------------------------------------------

# First two are the most important for the paper:
#   Adapter vs Linear = does nonlinear feature adaptation help?
#   Adapter vs MLP    = does the residual connection itself help?
#
# Other contrasts characterize adaptation depth.
CONTRASTS = [
    ("marine_adapter", "linear_probe", "Adapter - Linear"),
    ("marine_adapter", "mlp_head", "Adapter - MLP"),
    ("last_block_ft", "marine_adapter", "LastBlock - Adapter"),
    ("full_finetune", "marine_adapter", "FullFT - Adapter"),
    ("full_finetune", "last_block_ft", "FullFT - LastBlock"),
]

paired_rows = []

for (dataset, backbone, budget), g in core.groupby(
    ["dataset", "backbone", "budget"]
):
    pivot = g.pivot_table(
        index="seed",
        columns="method",
        values="test_macro_f1",
        aggfunc="first",
    )

    for a, b, name in CONTRASTS:
        if a not in pivot.columns or b not in pivot.columns:
            continue

        z = pivot[[a, b]].dropna()

        if len(z) == 0:
            continue

        diff = (z[a] - z[b]).values.astype(float)

        lo, hi = bootstrap_delta_ci(diff)
        p = exact_signflip_p(diff)

        sd = np.std(diff, ddof=1) if len(diff) > 1 else np.nan

        dz = (
            np.mean(diff) / sd
            if np.isfinite(sd) and sd > 0
            else np.nan
        )

        paired_rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "budget": budget,
            "contrast": name,
            "method_a": a,
            "method_b": b,
            "n_pairs": len(diff),
            "mean_a": z[a].mean(),
            "mean_b": z[b].mean(),
            "mean_delta": diff.mean(),
            "delta_ci_low": lo,
            "delta_ci_high": hi,
            "wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
            "cohen_dz": dz,
            "p_exact_signflip": p,
        })

paired = pd.DataFrame(paired_rows)

paired["p_holm"] = holm(
    paired["p_exact_signflip"].values
)

paired = paired.sort_values(
    ["contrast", "dataset", "backbone", "budget"]
)

paired.to_csv(
    OUT / "core_paired_statistics.csv",
    index=False,
)


print("\n" + "=" * 90)
print("PRIMARY PAIRED COMPARISONS — MACRO-F1")
print("=" * 90)

primary = paired[
    paired["contrast"].isin([
        "Adapter - Linear",
        "Adapter - MLP",
    ])
]

for contrast, gg in primary.groupby("contrast"):
    print(f"\n### {contrast}")

    for _, r in gg.iterrows():
        print(
            f"{r['dataset']:10s} "
            f"{r['backbone']:13s} "
            f"b={r['budget']:>3s}  "
            f"Δ={r['mean_delta']:+.4f} "
            f"[{r['delta_ci_low']:+.4f},"
            f"{r['delta_ci_high']:+.4f}]  "
            f"W/T/L={r['wins']}/{r['ties']}/{r['losses']}  "
            f"dz={fmt(r['cohen_dz'], 2)}  "
            f"p={r['p_exact_signflip']:.4f}"
        )


# ---------------------------------------------------------------------
# Cross-condition descriptive comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("CROSS-CONDITION ROBUSTNESS")
print("=" * 90)

# Average each method over seeds first -> one value per
# dataset/backbone/budget/method.
condition_means = (
    core.groupby(
        ["dataset", "backbone", "budget", "method"]
    )["test_macro_f1"]
    .mean()
    .reset_index()
)

cp = condition_means.pivot_table(
    index=["dataset", "backbone", "budget"],
    columns="method",
    values="test_macro_f1",
)

cross_rows = []

for a, b, name in CONTRASTS:
    if a not in cp.columns or b not in cp.columns:
        continue

    z = cp[[a, b]].dropna()
    d = (z[a] - z[b]).values

    lo, hi = bootstrap_delta_ci(d)

    cross_rows.append({
        "contrast": name,
        "n_conditions": len(d),
        "mean_delta": d.mean(),
        "median_delta": np.median(d),
        "ci_low": lo,
        "ci_high": hi,
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
        "p_exact_signflip": exact_signflip_p(d),
    })

cross = pd.DataFrame(cross_rows)

cross["p_holm"] = holm(
    cross["p_exact_signflip"].values
)

cross.to_csv(
    OUT / "cross_condition_statistics.csv",
    index=False,
)

print(
    cross.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)


# ---------------------------------------------------------------------
# Scratch CNN
# ---------------------------------------------------------------------

scratch = science[science["stage"] == "scratch"].copy()

print("\n" + "=" * 90)
print("SCRATCH CNN")
print("=" * 90)
print("rows:", len(scratch))
assert len(scratch) == 40

scratch_summary = summarize(
    scratch,
    ["dataset", "budget"],
)

scratch_summary.to_csv(
    OUT / "scratch_summary.csv",
    index=False,
)

print(
    scratch_summary[
        [
            "dataset",
            "budget",
            "n",
            "test_macro_f1_mean",
            "test_macro_f1_sd",
            "test_accuracy_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# Width ablation
# Include r=128 from canonical core for direct comparison.
# ---------------------------------------------------------------------

width = science[science["stage"] == "width"].copy()
print("\n" + "=" * 90)
print("WIDTH ABLATION")
print("=" * 90)
print("additional width rows:", len(width))
assert len(width) == 20

if "adapter_dim" in width.columns:
    r128 = core[
        (core["dataset"] == "fathomnet")
        & (core["backbone"] == "resnet18")
        & (core["method"] == "marine_adapter")
        & (core["budget"].isin(["010", "100"]))
    ].copy()

    r128["adapter_dim"] = 128

    ww = pd.concat(
        [width, r128],
        ignore_index=True,
        sort=False,
    )

    width_summary = summarize(
        ww,
        ["adapter_dim", "budget"],
    ).sort_values(
        ["budget", "adapter_dim"]
    )

    width_summary.to_csv(
        OUT / "width_summary_with_r128.csv",
        index=False,
    )

    print(
        width_summary[
            [
                "adapter_dim",
                "budget",
                "n",
                "test_macro_f1_mean",
                "test_macro_f1_sd",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
else:
    print("adapter_dim not found in width results")


# ---------------------------------------------------------------------
# Split sensitivity
# ---------------------------------------------------------------------

sens = science[science["stage"] == "sensitivity"].copy()

print("\n" + "=" * 90)
print("SPLIT SENSITIVITY")
print("=" * 90)
print("rows:", len(sens))
assert len(sens) == 54

sens_by = [
    c for c in
    ["split_seed", "method", "budget"]
    if c in sens.columns
]

print("grouping:", sens_by)

sens_summary = summarize(
    sens,
    sens_by,
)

sens_summary.to_csv(
    OUT / "sensitivity_summary.csv",
    index=False,
)

cols = sens_by + [
    "n",
    "test_macro_f1_mean",
    "test_macro_f1_sd",
]

print(
    sens_summary[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# K-shot
# ---------------------------------------------------------------------

shots = science[science["stage"] == "shots"].copy()

print("\n" + "=" * 90)
print("CLASS-BALANCED K-SHOT")
print("=" * 90)
print("rows:", len(shots))
assert len(shots) == 60

shots_summary = summarize(
    shots,
    ["method", "budget"],
)

shots_summary = shots_summary.sort_values(
    ["budget", "method"]
)

shots_summary.to_csv(
    OUT / "shots_summary.csv",
    index=False,
)

print(
    shots_summary[
        [
            "budget",
            "method",
            "n",
            "test_macro_f1_mean",
            "test_macro_f1_sd",
            "test_accuracy_mean",
            "test_balanced_accuracy_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# Trainable parameter counts
# ---------------------------------------------------------------------

param_col = None

for c in [
    "trainable_params",
    "params_trainable",
    "n_trainable_params",
]:
    if c in core.columns:
        param_col = c
        break

if param_col is not None:
    params = (
        core.groupby(
            ["dataset", "backbone", "method"]
        )[param_col]
        .median()
        .reset_index()
        .sort_values(["dataset", "backbone", "method"])
    )

    params.to_csv(
        OUT / "trainable_parameters.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("TRAINABLE PARAMETERS")
    print("=" * 90)
    print(params.to_string(index=False))


# ---------------------------------------------------------------------
# Final outputs
# ---------------------------------------------------------------------

science.to_csv(
    OUT / "all_scientific_runs.csv",
    index=False,
)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print("Output directory:", OUT)

for p in sorted(OUT.glob("*.csv")):
    print(" ", p)
