from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2/FULL_SUITE")
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES_NEW"
OUT.mkdir(parents=True, exist_ok=True)

core = pd.read_csv(STATS / "core_summary.csv")
paired = pd.read_csv(STATS / "core_paired_statistics.csv")
width = pd.read_csv(STATS / "width_summary_with_r128.csv")
sens = pd.read_csv(STATS / "sensitivity_summary.csv")
shots = pd.read_csv(STATS / "shots_summary.csv")
scratch = pd.read_csv(STATS / "scratch_summary.csv")
params = pd.read_csv(STATS / "trainable_parameters.csv")

def normalize_budget(v):
    s = str(v).strip()

    # Preserve class-balanced shot labels.
    if s.startswith("shot"):
        return s

    # pandas may read 001/005/010 as integers 1/5/10.
    try:
        n = int(float(s))
        if n in {1, 5, 10, 100}:
            return f"{n:03d}"
        return str(n)
    except Exception:
        return s


for df in [core, paired, width, sens, shots, scratch, params]:
    for c in ["dataset", "backbone", "method", "contrast"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "budget" in df.columns:
        df["budget"] = df["budget"].map(normalize_budget)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_ORDER = [
    "linear_probe",
    "mlp_head",
    "marine_adapter",
    "last_block_ft",
    "full_finetune",
]

METHOD_NAME = {
    "linear_probe": "Linear probe",
    "mlp_head": "MLP head",
    "marine_adapter": "MarineAdapter",
    "last_block_ft": "Last-block FT",
    "full_finetune": "Full FT",
    "scratch_cnn": "Scratch CNN",
}

COLORS = {
    "linear_probe":   "#4C78A8",
    "mlp_head":       "#F58518",
    "marine_adapter": "#54A24B",
    "last_block_ft":  "#B279A2",
    "full_finetune":  "#E45756",
    "scratch_cnn":    "#9D755D",
}

MARKERS = {
    "linear_probe": "o",
    "mlp_head": "s",
    "marine_adapter": "D",
    "last_block_ft": "^",
    "full_finetune": "P",
    "scratch_cnn": "X",
}

BUDGET_ORDER = ["001", "005", "010", "100"]
BUDGET_LABEL = {
    "001": "1%",
    "005": "5%",
    "010": "10%",
    "100": "100%",
    "shot01": "1-shot",
    "shot02": "2-shot",
    "shot05": "5-shot",
    "shot10": "10-shot",
}

DATASET_NAME = {
    "aqua20": "AQUA20",
    "fathomnet": "FathomNet",
}

BACKBONE_NAME = {
    "resnet18": "ResNet-18",
    "dinov2_small": "DINOv2-S",
    "scratch_cnn": "Scratch CNN",
}

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def method_handles(include_scratch=False):
    order = METHOD_ORDER.copy()
    if include_scratch:
        order = order + ["scratch_cnn"]
    handles = []
    for m in order:
        handles.append(
            Line2D(
                [0], [0],
                color=COLORS[m],
                marker=MARKERS[m],
                lw=2.2,
                markersize=7,
                label=METHOD_NAME[m]
            )
        )
    return handles

# ---------------------------------------------------------------------
# 01. Main core grid
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True)

panel_order = [
    ("aqua20", "resnet18"),
    ("aqua20", "dinov2_small"),
    ("fathomnet", "resnet18"),
    ("fathomnet", "dinov2_small"),
]

for ax, (dataset, backbone) in zip(axes.ravel(), panel_order):
    sub = core[
        (core["dataset"] == dataset) &
        (core["backbone"] == backbone)
    ].copy()

    for m in METHOD_ORDER:
        z = sub[sub["method"] == m].copy()
        z["budget"] = pd.Categorical(z["budget"], categories=BUDGET_ORDER, ordered=True)
        z = z.sort_values("budget")

        x = np.arange(len(z))
        y = z["test_macro_f1_mean"].values
        ylo = y - z["test_macro_f1_ci_low"].values
        yhi = z["test_macro_f1_ci_high"].values - y

        ax.errorbar(
            x, y,
            yerr=np.vstack([ylo, yhi]),
            color=COLORS[m],
            marker=MARKERS[m],
            lw=2.2,
            markersize=7,
            capsize=3,
        )

    ax.set_title(f"{DATASET_NAME[dataset]} / {BACKBONE_NAME[backbone]}")
    ax.set_xticks(np.arange(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABEL[b] for b in BUDGET_ORDER])
    ax.set_ylabel("Test macro-F1")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.legend(
    handles=method_handles(),
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03)
)
fig.suptitle("Core results across datasets, backbones, and label budgets", y=1.08)
fig.tight_layout()
save(fig, "01_core_macro_f1_grid")

# ---------------------------------------------------------------------
# 02/03. Forest plots for paired deltas
# ---------------------------------------------------------------------
condition_order = [
    ("aqua20", "resnet18", "001"),
    ("aqua20", "resnet18", "005"),
    ("aqua20", "resnet18", "010"),
    ("aqua20", "resnet18", "100"),
    ("aqua20", "dinov2_small", "001"),
    ("aqua20", "dinov2_small", "005"),
    ("aqua20", "dinov2_small", "010"),
    ("aqua20", "dinov2_small", "100"),
    ("fathomnet", "resnet18", "001"),
    ("fathomnet", "resnet18", "005"),
    ("fathomnet", "resnet18", "010"),
    ("fathomnet", "resnet18", "100"),
    ("fathomnet", "dinov2_small", "001"),
    ("fathomnet", "dinov2_small", "005"),
    ("fathomnet", "dinov2_small", "010"),
    ("fathomnet", "dinov2_small", "100"),
]

order_map = {k: i for i, k in enumerate(condition_order)}

def plot_forest(contrast, stem, title, color):
    sub = paired[paired["contrast"] == contrast].copy()
    sub["ord"] = sub.apply(
        lambda r: order_map[(r["dataset"], r["backbone"], r["budget"])],
        axis=1
    )
    sub = sub.sort_values("ord", ascending=False).reset_index(drop=True)

    labels = [
        f"{DATASET_NAME[r.dataset]} / {BACKBONE_NAME[r.backbone]} / {BUDGET_LABEL[r.budget]}"
        for _, r in sub.iterrows()
    ]

    y = np.arange(len(sub))
    x = sub["mean_delta"].values
    xerr = np.vstack([
        x - sub["delta_ci_low"].values,
        sub["delta_ci_high"].values - x
    ])

    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    ax.axvline(0.0, color="black", lw=1.2, alpha=0.8)
    ax.errorbar(
        x, y,
        xerr=xerr,
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=2.0,
        capsize=3,
        markersize=6.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paired mean Δ macro-F1")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # add W/T/L at the right
    xmax = max(sub["delta_ci_high"].max(), sub["mean_delta"].max()) if len(sub) else 0.1
    xmin = min(sub["delta_ci_low"].min(), sub["mean_delta"].min()) if len(sub) else -0.1
    span = xmax - xmin
    text_x = xmax + 0.06 * span

    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.text(
            text_x,
            yi,
            f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.text(
        text_x,
        y.max() + 0.9,
        "W/T/L",
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(xmin - 0.08 * span, xmax + 0.30 * span)
    fig.tight_layout()
    save(fig, stem)

plot_forest(
    "Adapter - Linear",
    "02_adapter_minus_linear_forest",
    "MarineAdapter vs Linear probe",
    COLORS["marine_adapter"]
)

plot_forest(
    "Adapter - MLP",
    "03_adapter_minus_mlp_forest",
    "MarineAdapter vs parameter-matched MLP",
    COLORS["mlp_head"]
)

# ---------------------------------------------------------------------
# 04. Width ablation
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6.3))

for budget, marker in [("010", "o"), ("100", "s")]:
    z = width[width["budget"] == budget].copy().sort_values("adapter_dim")
    ax.plot(
        z["adapter_dim"].values,
        z["test_macro_f1_mean"].values,
        marker=marker,
        lw=2.4,
        markersize=7,
        label=f"Budget {BUDGET_LABEL[budget]}",
    )
    ax.fill_between(
        z["adapter_dim"].values,
        z["test_macro_f1_ci_low"].values,
        z["test_macro_f1_ci_high"].values,
        alpha=0.12,
    )

ax.set_title("Width ablation (FathomNet / ResNet-18 / MarineAdapter)")
ax.set_xlabel("Adapter bottleneck dimension r")
ax.set_ylabel("Test macro-F1")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
save(fig, "04_width_ablation")

# ---------------------------------------------------------------------
# 05. Split sensitivity
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True)

methods_sens = ["linear_probe", "marine_adapter", "full_finetune"]
budget_panels = [("010", "10%"), ("100", "100%")]

for ax, (budget, title_budget) in zip(axes, budget_panels):
    sub = sens[sens["budget"] == budget].copy()
    for m in methods_sens:
        z = sub[sub["method"] == m].copy().sort_values("split_seed")
        x = np.arange(len(z))
        ax.errorbar(
            x,
            z["test_macro_f1_mean"].values,
            yerr=z["test_macro_f1_sd"].values,
            color=COLORS[m],
            marker=MARKERS[m],
            lw=2.2,
            markersize=7,
            capsize=3,
            label=METHOD_NAME[m],
        )
    ax.set_title(f"Budget {title_budget}")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([str(int(v)) for v in sorted(sub["split_seed"].unique())])
    ax.set_xlabel("Alternative split seed")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Test macro-F1")
fig.legend(
    handles=[Line2D([0],[0], color=COLORS[m], marker=MARKERS[m], lw=2.2,
                    markersize=7, label=METHOD_NAME[m]) for m in methods_sens],
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)
fig.suptitle("Split sensitivity (FathomNet / ResNet-18)", y=1.10)
fig.tight_layout()
save(fig, "05_split_sensitivity")

# ---------------------------------------------------------------------
# 06. K-shot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 6.4))

shot_order = ["shot01", "shot02", "shot05", "shot10"]
shot_x = np.array([1, 2, 5, 10])

for m in METHOD_ORDER:
    z = shots[shots["method"] == m].copy()
    z["budget"] = pd.Categorical(z["budget"], categories=shot_order, ordered=True)
    z = z.sort_values("budget")
    ax.errorbar(
        shot_x,
        z["test_macro_f1_mean"].values,
        yerr=z["test_macro_f1_sd"].values,
        color=COLORS[m],
        marker=MARKERS[m],
        lw=2.2,
        markersize=7,
        capsize=3,
        label=METHOD_NAME[m],
    )

ax.set_title("Class-balanced k-shot evaluation (FathomNet / ResNet-18)")
ax.set_xlabel("Shots per class")
ax.set_ylabel("Test macro-F1")
ax.set_xticks(shot_x)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
save(fig, "06_kshot")

# ---------------------------------------------------------------------
# 07. Scratch vs selected methods
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=True)

selected = ["scratch_cnn", "linear_probe", "marine_adapter", "mlp_head"]

for ax, dataset in zip(axes, ["aqua20", "fathomnet"]):
    parts = []

    s = scratch[scratch["dataset"] == dataset].copy()
    s["method"] = "scratch_cnn"
    s["backbone"] = "scratch_cnn"
    s["test_macro_f1_ci_low"] = s["test_macro_f1_mean"] - s["test_macro_f1_sd"]
    s["test_macro_f1_ci_high"] = s["test_macro_f1_mean"] + s["test_macro_f1_sd"]
    parts.append(s[[
        "dataset", "budget", "method",
        "test_macro_f1_mean",
        "test_macro_f1_ci_low",
        "test_macro_f1_ci_high"
    ]])

    c = core[
        (core["dataset"] == dataset) &
        (core["backbone"] == "resnet18") &
        (core["method"].isin(["linear_probe", "marine_adapter", "mlp_head"]))
    ].copy()
    parts.append(c[[
        "dataset", "budget", "method",
        "test_macro_f1_mean",
        "test_macro_f1_ci_low",
        "test_macro_f1_ci_high"
    ]])

    sub = pd.concat(parts, ignore_index=True)

    for m in selected:
        z = sub[sub["method"] == m].copy()
        if len(z) == 0:
            continue
        z["budget"] = pd.Categorical(z["budget"], categories=BUDGET_ORDER, ordered=True)
        z = z.sort_values("budget")
        x = np.arange(len(z))
        y = z["test_macro_f1_mean"].values
        ylo = y - z["test_macro_f1_ci_low"].values
        yhi = z["test_macro_f1_ci_high"].values - y
        ax.errorbar(
            x, y,
            yerr=np.vstack([ylo, yhi]),
            color=COLORS[m],
            marker=MARKERS[m],
            lw=2.2,
            markersize=7,
            capsize=3,
            label=METHOD_NAME[m],
        )

    ax.set_title(DATASET_NAME[dataset])
    ax.set_xticks(np.arange(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABEL[b] for b in BUDGET_ORDER])
    ax.set_xlabel("Training budget")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Test macro-F1")
fig.legend(
    handles=method_handles(include_scratch=True)[-1:] + [
        h for h in method_handles() if h.get_label() in
        [METHOD_NAME["linear_probe"], METHOD_NAME["marine_adapter"], METHOD_NAME["mlp_head"]]
    ],
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)
fig.suptitle("Scratch training vs frozen-feature adaptation (ResNet-18 family)", y=1.10)
fig.tight_layout()
save(fig, "07_scratch_vs_linear_adapter_mlp")

# ---------------------------------------------------------------------
# 08. Trainable parameters (corrected)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=False)

for ax, dataset in zip(axes, ["aqua20", "fathomnet"]):
    sub = params[params["dataset"] == dataset].copy()

    combos = [
        ("resnet18", "linear_probe"),
        ("resnet18", "mlp_head"),
        ("resnet18", "marine_adapter"),
        ("resnet18", "last_block_ft"),
        ("resnet18", "full_finetune"),
        ("dinov2_small", "linear_probe"),
        ("dinov2_small", "mlp_head"),
        ("dinov2_small", "marine_adapter"),
        ("dinov2_small", "last_block_ft"),
        ("dinov2_small", "full_finetune"),
    ]

    labels = []
    vals = []

    for bb, m in combos:
        z = sub[(sub["backbone"] == bb) & (sub["method"] == m)]
        if len(z) == 0:
            continue
        labels.append(f"{BACKBONE_NAME[bb]}\n{METHOD_NAME[m]}")
        vals.append(float(z["trainable_params"].iloc[0]))

    x = np.arange(len(vals))
    colors = [COLORS[c[1]] for c in combos[:len(vals)]]
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Trainable parameters (log scale)")
    ax.set_title(DATASET_NAME[dataset])
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.suptitle("Trainable parameter counts by dataset, backbone, and method", y=1.04)
fig.tight_layout()
save(fig, "08_trainable_params_bar")

print("WROTE FIGURES TO:", OUT)
for p in sorted(OUT.iterdir()):
    print(p.name)
