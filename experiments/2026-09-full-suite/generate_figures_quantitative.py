from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2/FULL_SUITE")
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES_NEW" / "QUANTITATIVE"
OUT.mkdir(parents=True, exist_ok=True)

core = pd.read_csv(STATS / "core_summary.csv")
paired = pd.read_csv(STATS / "core_paired_statistics.csv")
cross = pd.read_csv(STATS / "cross_condition_statistics.csv")
shots = pd.read_csv(STATS / "shots_summary.csv")
params = pd.read_csv(STATS / "trainable_parameters.csv")

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
def norm_budget(v):
    s = str(v).strip()

    if s.startswith("shot"):
        return s

    try:
        n = int(float(s))
        if n in {1, 5, 10, 100}:
            return f"{n:03d}"
        return str(n)
    except Exception:
        return s


for df in [core, paired, cross, shots, params]:
    for c in ["dataset", "backbone", "method", "contrast"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "budget" in df.columns:
        df["budget"] = df["budget"].map(norm_budget)


# If an old trainable_parameters.csv is still present, reconstruct it.
if "dataset" not in params.columns:
    runs = pd.read_csv(STATS / "all_scientific_runs.csv")

    param_col = None
    for c in ["trainable_params", "params_trainable", "n_trainable_params"]:
        if c in runs.columns:
            param_col = c
            break

    if param_col is None:
        raise RuntimeError("Cannot locate trainable parameter column.")

    params = (
        runs[runs["stage"].astype(str) == "core"]
        .groupby(["dataset", "backbone", "method"])[param_col]
        .median()
        .reset_index()
        .rename(columns={param_col: "trainable_params"})
    )


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 20,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHODS = [
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
}

SHORT = {
    "linear_probe": "Linear",
    "mlp_head": "MLP",
    "marine_adapter": "Adapter",
    "last_block_ft": "LastBlk",
    "full_finetune": "FullFT",
}

DATASET = {
    "aqua20": "AQUA20",
    "fathomnet": "FathomNet",
}

BACKBONE = {
    "resnet18": "ResNet-18",
    "dinov2_small": "DINOv2-S",
}

BUDGETS = ["001", "005", "010", "100"]

BUDGET_NAME = {
    "001": "1%",
    "005": "5%",
    "010": "10%",
    "100": "100%",
    "shot01": "1-shot",
    "shot02": "2-shot",
    "shot05": "5-shot",
    "shot10": "10-shot",
}

PANELS = [
    ("aqua20", "resnet18"),
    ("aqua20", "dinov2_small"),
    ("fathomnet", "resnet18"),
    ("fathomnet", "dinov2_small"),
]

VIR = mpl.cm.viridis
METHOD_COLORS = {
    m: VIR(x)
    for m, x in zip(METHODS, np.linspace(0.08, 0.92, len(METHODS)))
}

MARKERS = {
    "linear_probe": "o",
    "mlp_head": "s",
    "marine_adapter": "D",
    "last_block_ft": "^",
    "full_finetune": "P",
}


def bbox_text(
    ax,
    x,
    y,
    text,
    fontsize=9,
    ha="center",
    va="center",
    zorder=20,
    pad=0.20,
    alpha=0.82,
    red=True,
):
    edge = "#D62728" if red else "black"

    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle=f"round,pad={pad}",
            facecolor=(1, 1, 1, alpha),
            edgecolor=edge,
            linewidth=1.05,
        ),
        zorder=zorder,
    )


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{name}.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# 15. Cross-condition effect-size summary
# =====================================================================
wanted = [
    "Adapter - Linear",
    "Adapter - MLP",
    "LastBlock - Adapter",
    "FullFT - Adapter",
    "FullFT - LastBlock",
]

d = cross[cross["contrast"].isin(wanted)].copy()

order = {name: i for i, name in enumerate(wanted)}
d["ord"] = d["contrast"].map(order)
d = d.sort_values("ord", ascending=False)

fig, ax = plt.subplots(figsize=(10.8, 6.7))

y = np.arange(len(d))
x = d["mean_delta"].values

xerr = np.vstack([
    x - d["ci_low"].values,
    d["ci_high"].values - x,
])

cols = VIR(np.linspace(0.15, 0.90, len(d)))

ax.axvline(
    0,
    color="black",
    lw=1.3,
    zorder=1,
)

for i in range(len(d)):
    ax.errorbar(
        x[i],
        y[i],
        xerr=np.array([[xerr[0, i]], [xerr[1, i]]]),
        fmt="o",
        markersize=10,
        color=cols[i],
        ecolor=cols[i],
        markeredgecolor="black",
        markeredgewidth=0.8,
        elinewidth=3,
        capsize=5,
        zorder=4,
    )

    r = d.iloc[i]

    bbox_text(
        ax,
        x[i],
        y[i] + 0.28,
        f"{x[i]:+.3f}",
        fontsize=10,
    )

    ax.text(
        max(d["ci_high"]) + 0.02,
        y[i],
        f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}",
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )

ax.set_yticks(y)
ax.set_yticklabels(d["contrast"])

ax.set_xlabel("Mean Δ macro-F1 across 16 core conditions")
ax.set_title("Cross-condition effect summary")

ax.grid(
    True,
    axis="x",
    alpha=0.22,
)

ax.text(
    max(d["ci_high"]) + 0.02,
    len(d) - 0.20,
    "W/T/L",
    fontweight="bold",
    fontsize=11,
)

fig.tight_layout()
save(fig, "15_cross_condition_effects")


# =====================================================================
# 16. Pairwise dominance matrix
# =====================================================================
cond = (
    core.groupby(
        ["dataset", "backbone", "budget", "method"]
    )["test_macro_f1_mean"]
    .mean()
    .reset_index()
)

piv = cond.pivot_table(
    index=["dataset", "backbone", "budget"],
    columns="method",
    values="test_macro_f1_mean",
)

wins = np.zeros((len(METHODS), len(METHODS)), dtype=int)
loss = np.zeros_like(wins)

for i, a in enumerate(METHODS):
    for j, b in enumerate(METHODS):
        if i == j:
            continue

        z = piv[[a, b]].dropna()
        diff = z[a] - z[b]

        wins[i, j] = int((diff > 0).sum())
        loss[i, j] = int((diff < 0).sum())

fig, ax = plt.subplots(figsize=(9.2, 7.9))

im = ax.imshow(
    wins,
    cmap=VIR,
    vmin=0,
    vmax=16,
    alpha=0.96,
)

for i in range(len(METHODS) + 1):
    ax.axhline(
        i - 0.5,
        color="black",
        lw=0.8,
    )
    ax.axvline(
        i - 0.5,
        color="black",
        lw=0.8,
    )

for i in range(len(METHODS)):
    for j in range(len(METHODS)):

        if i == j:
            ax.text(
                j,
                i,
                "—",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
            )
            continue

        bbox_text(
            ax,
            j,
            i,
            f"{wins[i,j]}–{loss[i,j]}",
            fontsize=11,
        )

ax.set_xticks(np.arange(len(METHODS)))
ax.set_xticklabels(
    [SHORT[m] for m in METHODS],
    rotation=25,
    ha="right",
)

ax.set_yticks(np.arange(len(METHODS)))
ax.set_yticklabels(
    [METHOD_NAME[m] for m in METHODS]
)

ax.set_xlabel("Opponent")
ax.set_ylabel("Method")
ax.set_title("Pairwise dominance across 16 core conditions")

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
    fraction=0.045,
)

cbar.set_label("Number of conditions won")

fig.tight_layout()
save(fig, "16_pairwise_dominance_matrix")


# =====================================================================
# 17. Parameter-efficiency frontier at full budget
# =====================================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(15.8, 11.0),
)

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.92,
    bottom=0.08,
    wspace=0.22,
    hspace=0.28,
)

for ax, (dataset, backbone) in zip(axes.ravel(), PANELS):

    perf = core[
        (core["dataset"] == dataset) &
        (core["backbone"] == backbone) &
        (core["budget"] == "100")
    ].copy()

    prm = params[
        (params["dataset"].astype(str) == dataset) &
        (params["backbone"].astype(str) == backbone)
    ].copy()

    z = perf.merge(
        prm,
        on=["dataset", "backbone", "method"],
        how="left",
        suffixes=("", "_p"),
    )

    z = z.sort_values("trainable_params")

    # Pareto frontier:
    # for increasing parameter count, keep a point if it improves F1.
    frontier = []
    best_y = -np.inf

    for _, r in z.iterrows():
        yy = r["test_macro_f1_mean"]

        if yy > best_y:
            frontier.append(r)
            best_y = yy

    frontier = pd.DataFrame(frontier)

    # soft background gradient
    ymin = z["test_macro_f1_mean"].min() - 0.035
    ymax = z["test_macro_f1_mean"].max() + 0.035

    xx = np.linspace(
        np.log10(z["trainable_params"].min()) - 0.20,
        np.log10(z["trainable_params"].max()) + 0.20,
        400,
    )

    grad = np.linspace(0, 1, 256).reshape(1, -1)

    ax.imshow(
        grad,
        extent=[
            10**xx.min(),
            10**xx.max(),
            ymin,
            ymax,
        ],
        aspect="auto",
        cmap="viridis",
        alpha=0.08,
        origin="lower",
        zorder=0,
    )

    # Pareto frontier
    if len(frontier) > 1:
        ax.plot(
            frontier["trainable_params"],
            frontier["test_macro_f1_mean"],
            color="black",
            lw=2.5,
            alpha=0.78,
            zorder=2,
        )

        ax.fill_between(
            frontier["trainable_params"].values,
            ymin,
            frontier["test_macro_f1_mean"].values,
            color=VIR(0.65),
            alpha=0.08,
            zorder=1,
        )

    for _, r in z.iterrows():
        m = r["method"]

        ax.scatter(
            r["trainable_params"],
            r["test_macro_f1_mean"],
            s=170,
            color=METHOD_COLORS[m],
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

        bbox_text(
            ax,
            r["trainable_params"],
            r["test_macro_f1_mean"] + 0.018,
            f"{SHORT[m]}\n{r['test_macro_f1_mean']:.3f}",
            fontsize=8,
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Test macro-F1")

    ax.set_title(
        f"{DATASET[dataset]} / {BACKBONE[backbone]}"
    )

    ax.grid(
        True,
        alpha=0.20,
        zorder=0,
    )

fig.suptitle(
    "Performance–parameter efficiency at 100% budget",
    y=0.985,
)

save(fig, "17_parameter_efficiency_frontier")


# =====================================================================
# 18. Quantitative budget-response curves
# =====================================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(15.8, 10.8),
    sharex=True,
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.91,
    bottom=0.09,
    wspace=0.18,
    hspace=0.26,
)

x = np.arange(len(BUDGETS))

for ax, (dataset, backbone) in zip(axes.ravel(), PANELS):

    sub = core[
        (core["dataset"] == dataset) &
        (core["backbone"] == backbone)
    ].copy()

    for m in METHODS:
        zz = sub[sub["method"] == m].copy()

        zz["budget"] = pd.Categorical(
            zz["budget"],
            categories=BUDGETS,
            ordered=True,
        )

        zz = zz.sort_values("budget")

        y = zz["test_macro_f1_mean"].values
        lo = zz["test_macro_f1_ci_low"].values
        hi = zz["test_macro_f1_ci_high"].values

        col = METHOD_COLORS[m]

        ax.plot(
            x,
            y,
            color=col,
            marker=MARKERS[m],
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            lw=2.5,
            label=METHOD_NAME[m],
            zorder=4,
        )

        ax.fill_between(
            x,
            lo,
            hi,
            color=col,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )

        bbox_text(
            ax,
            x[-1],
            y[-1],
            f"{y[-1]:.3f}",
            fontsize=7.5,
            ha="left",
        )

    ax.set_title(
        f"{DATASET[dataset]} / {BACKBONE[backbone]}"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [BUDGET_NAME[b] for b in BUDGETS]
    )

    ax.set_xlabel("Training-label budget")
    ax.set_ylabel("Test macro-F1")

    ax.grid(
        True,
        alpha=0.20,
    )

handles = [
    Line2D(
        [0], [0],
        color=METHOD_COLORS[m],
        marker=MARKERS[m],
        markeredgecolor="black",
        markeredgewidth=0.5,
        lw=2.5,
        label=METHOD_NAME[m],
    )
    for m in METHODS
]

fig.legend(
    handles=handles,
    ncol=5,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.975),
)

fig.suptitle(
    "Quantitative response to increasing label budget",
    y=1.02,
)

save(fig, "18_budget_response_curves")


# =====================================================================
# 19. Seed-variability landscape
# =====================================================================
cond_labels = []
matrix = []

for m in METHODS:
    row = []

    for dataset, backbone in PANELS:
        for b in BUDGETS:
            z = core[
                (core["dataset"] == dataset) &
                (core["backbone"] == backbone) &
                (core["method"] == m) &
                (core["budget"] == b)
            ]

            row.append(
                float(z["test_macro_f1_sd"].iloc[0])
            )

    matrix.append(row)

for dataset, backbone in PANELS:
    ds = "AQ" if dataset == "aqua20" else "FN"
    bb = "R18" if backbone == "resnet18" else "DINO"

    for b in BUDGETS:
        cond_labels.append(
            f"{ds}-{bb}\n{BUDGET_NAME[b]}"
        )

matrix = np.asarray(matrix)

fig, ax = plt.subplots(
    figsize=(17.0, 6.2)
)

fig.subplots_adjust(
    left=0.13,
    right=0.92,
    top=0.87,
    bottom=0.24,
)

im = ax.imshow(
    matrix,
    cmap="viridis",
    aspect="auto",
    alpha=0.97,
)

for i in range(matrix.shape[0] + 1):
    ax.axhline(
        i - 0.5,
        color="black",
        lw=0.65,
    )

for j in range(matrix.shape[1] + 1):
    ax.axvline(
        j - 0.5,
        color="black",
        lw=0.35,
    )

for j in [4, 8, 12]:
    ax.axvline(
        j - 0.5,
        color="black",
        lw=1.5,
    )

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        bbox_text(
            ax,
            j,
            i,
            f"{matrix[i,j]:.3f}",
            fontsize=7.3,
            pad=0.13 if False else 0.20,
        )

ax.set_yticks(np.arange(len(METHODS)))
ax.set_yticklabels(
    [METHOD_NAME[m] for m in METHODS]
)

ax.set_xticks(
    np.arange(len(cond_labels))
)

ax.set_xticklabels(
    cond_labels,
    rotation=0,
)

ax.set_title(
    "Seed-to-seed variability across all core conditions"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.012,
    fraction=0.032,
)

cbar.set_label("Macro-F1 standard deviation")

save(fig, "19_seed_variability_landscape")


# =====================================================================
# 20. K-shot sample efficiency
# =====================================================================
shot_order = [
    "shot01",
    "shot02",
    "shot05",
    "shot10",
]

shot_x = np.array([1, 2, 5, 10], dtype=float)

fig, ax = plt.subplots(
    figsize=(10.5, 7.0)
)

# background viridis gradient
grad = np.linspace(0, 1, 256).reshape(1, -1)

ax.imshow(
    grad,
    extent=[0.75, 11.0, -0.005, 0.205],
    aspect="auto",
    cmap="viridis",
    alpha=0.055,
    origin="lower",
    zorder=0,
)

for m in METHODS:
    z = shots[
        shots["method"] == m
    ].copy()

    z["budget"] = pd.Categorical(
        z["budget"],
        categories=shot_order,
        ordered=True,
    )

    z = z.sort_values("budget")

    y = z["test_macro_f1_mean"].values
    sd = z["test_macro_f1_sd"].values

    col = METHOD_COLORS[m]

    ax.plot(
        shot_x,
        y,
        color=col,
        marker=MARKERS[m],
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=9,
        lw=2.8,
        label=METHOD_NAME[m],
        zorder=4,
    )

    ax.fill_between(
        shot_x,
        y - sd,
        y + sd,
        color=col,
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )

    bbox_text(
        ax,
        10,
        y[-1],
        f"{y[-1]:.3f}",
        fontsize=8,
        ha="left",
    )

ax.set_xscale("log", base=2)

ax.set_xticks(
    shot_x,
    ["1", "2", "5", "10"],
)

ax.set_xlabel("Labeled examples per class")
ax.set_ylabel("Test macro-F1")

ax.set_title(
    "Class-balanced sample efficiency on FathomNet / ResNet-18"
)

ax.grid(
    True,
    alpha=0.22,
)

ax.legend(
    frameon=False,
    loc="upper left",
    ncol=2,
)

fig.tight_layout()
save(fig, "20_kshot_sample_efficiency")


print("=" * 80)
print("QUANTITATIVE FIGURES COMPLETE")
print("OUTPUT:", OUT)
print("=" * 80)

for p in sorted(OUT.glob("*")):
    print(p.name)
