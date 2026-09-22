from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path("/home/tahiti/MARINE_DATASETS/MARINE_EXPERIMENTS_V2/FULL_SUITE")
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES_NEW" / "FANCY"
OUT.mkdir(parents=True, exist_ok=True)

core = pd.read_csv(STATS / "core_summary.csv")
paired = pd.read_csv(STATS / "core_paired_statistics.csv")
params = pd.read_csv(STATS / "trainable_parameters.csv")

# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------
def normalize_budget(v):
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

for df in [core, paired, params]:
    for c in ["dataset", "backbone", "method", "contrast"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "budget" in df.columns:
        df["budget"] = df["budget"].map(normalize_budget)

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.linewidth": 1.1,
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
    "linear_probe": "Linear",
    "mlp_head": "MLP",
    "marine_adapter": "Adapter",
    "last_block_ft": "LastBlk",
    "full_finetune": "FullFT",
}

METHOD_LONG = {
    "linear_probe": "Linear probe",
    "mlp_head": "MLP head",
    "marine_adapter": "MarineAdapter",
    "last_block_ft": "Last-block FT",
    "full_finetune": "Full FT",
}

BUDGET_ORDER = ["001", "005", "010", "100"]
BUDGET_LABEL = {"001": "1%", "005": "5%", "010": "10%", "100": "100%"}
DATASET_NAME = {"aqua20": "AQUA20", "fathomnet": "FathomNet"}
BACKBONE_NAME = {"resnet18": "ResNet-18", "dinov2_small": "DINOv2-S"}

PANEL_ORDER = [
    ("aqua20", "resnet18"),
    ("aqua20", "dinov2_small"),
    ("fathomnet", "resnet18"),
    ("fathomnet", "dinov2_small"),
]

# Viridis-based palette for methods
method_colors = mpl.cm.viridis(np.linspace(0.12, 0.92, len(METHOD_ORDER)))
METHOD_COLOR = {m: method_colors[i] for i, m in enumerate(METHOD_ORDER)}

def boxed_text(
    ax, x, y, text,
    fontsize=11,
    ha="center",
    va="center",
    red=True,
    fontweight="bold",
    pad=0.22,
    alpha=0.82,
    zorder=8,
):
    edge = "#D62728" if red else "black"

    return ax.text(
        x, y, text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color="black",
        bbox=dict(
            boxstyle=f"round,pad={pad}",
            facecolor=(1.0, 1.0, 1.0, alpha),
            edgecolor=edge,
            linewidth=1.15,
        ),
        zorder=zorder,
    )

def save(fig, stem):

    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Helper lookup
# ------------------------------------------------------------
core_lookup = {}
for _, r in core.iterrows():
    key = (r["dataset"], r["backbone"], r["method"], r["budget"])
    core_lookup[key] = r

# ------------------------------------------------------------
# 09. Best method map
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(16.5, 9.2),
)

fig.subplots_adjust(
    left=0.07,
    right=0.885,
    bottom=0.09,
    top=0.89,
    wspace=0.16,
    hspace=0.30,
)

score_min = core["test_macro_f1_mean"].min()
score_max = core["test_macro_f1_mean"].max()

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(
    vmin=score_min,
    vmax=score_max,
)

last_im = None

for ax, (dataset, backbone) in zip(axes.ravel(), PANEL_ORDER):

    values = []
    best_rows = []

    for budget in BUDGET_ORDER:
        sub = core[
            (core["dataset"] == dataset) &
            (core["backbone"] == backbone) &
            (core["budget"] == budget)
        ].copy()

        best = (
            sub.sort_values("test_macro_f1_mean", ascending=False)
               .iloc[0]
        )

        best_rows.append(best)
        values.append(best["test_macro_f1_mean"])

    mat = np.array([values])

    last_im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        alpha=0.96,
    )

    # clean black cell borders
    for j in range(len(BUDGET_ORDER) + 1):
        ax.axvline(j - 0.5, color="black", lw=0.9, zorder=5)
    ax.axhline(-0.5, color="black", lw=0.9, zorder=5)
    ax.axhline(0.5, color="black", lw=0.9, zorder=5)

    for j, best in enumerate(best_rows):
        # method label
        t = ax.text(
            j, -0.02,
            METHOD_NAME[best["method"]],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=7,
        )
        t.set_path_effects([
            pe.withStroke(linewidth=1.8, foreground="white", alpha=0.80)
        ])

        # score in bbox
        boxed_text(
            ax,
            j,
            0.18,
            f"{best['test_macro_f1_mean']:.3f}",
            fontsize=11,
            red=True,
        )

    ax.set_xticks(np.arange(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABEL[b] for b in BUDGET_ORDER])
    ax.set_yticks([])
    ax.set_title(f"{DATASET_NAME[dataset]} / {BACKBONE_NAME[backbone]}")

# dedicated colorbar axis outside the panels
cax = fig.add_axes([0.905, 0.19, 0.015, 0.63])
cbar = fig.colorbar(last_im, cax=cax)
cbar.set_label("Best test macro-F1")

fig.suptitle("Best method by budget, dataset, and backbone", y=0.965)
save(fig, "09_best_method_map")

# ------------------------------------------------------------
# 10. Delta heatmaps
# ------------------------------------------------------------
contrast_specs = [
    ("Adapter - Linear", "MarineAdapter minus Linear probe"),
    ("Adapter - MLP", "MarineAdapter minus parameter-matched MLP"),
]

fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.2))
fig.subplots_adjust(
    left=0.08,
    right=0.885,
    bottom=0.14,
    top=0.84,
    wspace=0.28,
)

all_vals = paired[
    paired["contrast"].isin([c[0] for c in contrast_specs])
]["mean_delta"].values

v = max(abs(all_vals.min()), abs(all_vals.max()))
norm = mpl.colors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
cmap = mpl.cm.RdYlGn

last_im = None

for ax, (contrast, title) in zip(axes, contrast_specs):
    sub = paired[paired["contrast"] == contrast].copy()

    rows = []
    row_labels = []

    for ds, bb in PANEL_ORDER:
        row_labels.append(f"{DATASET_NAME[ds]}\\n{BACKBONE_NAME[bb]}")
        vals = []
        for b in BUDGET_ORDER:
            z = sub[
                (sub["dataset"] == ds) &
                (sub["backbone"] == bb) &
                (sub["budget"] == b)
            ]
            vals.append(float(z["mean_delta"].iloc[0]))
        rows.append(vals)

    mat = np.array(rows)

    last_im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        alpha=0.95,
    )

    # black grid
    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color="black", lw=0.85, zorder=5)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color="black", lw=0.85, zorder=5)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            boxed_text(
                ax,
                j,
                i,
                f"{val:+.3f}",
                fontsize=10,
                red=True,
                pad=0.18,
                alpha=0.84,
            )

    ax.set_xticks(np.arange(len(BUDGET_ORDER)))
    ax.set_xticklabels([BUDGET_LABEL[b] for b in BUDGET_ORDER])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

cax = fig.add_axes([0.905, 0.20, 0.015, 0.56])
cbar = fig.colorbar(last_im, cax=cax)
cbar.set_label("Paired mean Δ macro-F1")

fig.suptitle("Condition-wise paired performance deltas", y=0.955)
save(fig, "10_delta_heatmaps")

# ------------------------------------------------------------
# 11. Pareto trade-off at 100% budget
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.1), sharey=False)

for ax, dataset in zip(axes, ["aqua20", "fathomnet"]):
    sub = core[(core["dataset"] == dataset) & (core["budget"] == "100")].copy()
    prm = params[params["dataset"] == dataset].copy()

    merged = sub.merge(
        prm,
        on=["dataset", "backbone", "method"],
        how="left",
        suffixes=("", "_param")
    )

    # color by macro-F1 via viridis
    score_norm = mpl.colors.Normalize(
        vmin=merged["test_macro_f1_mean"].min(),
        vmax=merged["test_macro_f1_mean"].max()
    )
    colors = mpl.cm.viridis(score_norm(merged["test_macro_f1_mean"].values))

    ax.scatter(
        merged["trainable_params"].values,
        merged["test_macro_f1_mean"].values,
        s=180,
        c=colors,
        edgecolors="black",
        linewidths=0.9,
        alpha=0.92,
        zorder=3
    )

    for _, r in merged.iterrows():
        short = f"{BACKBONE_NAME[r['backbone']]}\n{METHOD_NAME[r['method']]}"
        boxed_text(
            ax,
            r["trainable_params"] * 1.02,
            r["test_macro_f1_mean"] + 0.004,
            short,
            fontsize=8,
            ha="left",
            va="bottom"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test macro-F1 at 100% budget")
    ax.set_title(DATASET_NAME[dataset])
    ax.grid(True, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

fig.suptitle("Performance–adaptation trade-off at full budget", y=1.02)
fig.tight_layout()
save(fig, "11_pareto_tradeoff_budget100")

# ------------------------------------------------------------
# 12. Average rank bars
# ------------------------------------------------------------
rows = []
for ds, bb in PANEL_ORDER:
    for b in BUDGET_ORDER:
        sub = core[
            (core["dataset"] == ds) &
            (core["backbone"] == bb) &
            (core["budget"] == b)
        ].copy().sort_values("test_macro_f1_mean", ascending=False)

        vals = sub["test_macro_f1_mean"].values
        methods = sub["method"].values
        # rank 1 = best
        for rank, m in enumerate(methods, start=1):
            rows.append({
                "dataset": ds,
                "backbone": bb,
                "budget": b,
                "method": m,
                "rank": rank,
            })

rank_df = pd.DataFrame(rows)
rank_summary = rank_df.groupby("method")["rank"].agg(["mean", "std"]).reset_index()
rank_summary["method"] = pd.Categorical(rank_summary["method"], categories=METHOD_ORDER, ordered=True)
rank_summary = rank_summary.sort_values("mean", ascending=True)

fig, ax = plt.subplots(figsize=(9.4, 5.8))

vals = rank_summary["mean"].values
stds = rank_summary["std"].values
methods = rank_summary["method"].astype(str).tolist()
x = np.arange(len(methods))
colors = mpl.cm.viridis(np.linspace(0.12, 0.90, len(methods)))

bars = ax.bar(
    x, vals,
    yerr=stds,
    capsize=4,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.88
)

for xi, yi, m in zip(x, vals, methods):
    boxed_text(ax, xi, yi + 0.12, f"{yi:.2f}", fontsize=10, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([METHOD_LONG[m] for m in methods], rotation=18, ha="right")
ax.set_ylabel("Average rank (lower is better)")
ax.set_title("Average method rank across 16 core conditions")
ax.grid(True, axis="y", alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "12_average_rank_bars")

# ------------------------------------------------------------
# 13. Contrast by budget
# ------------------------------------------------------------
contrast_order = [
    "Adapter - Linear",
    "Adapter - MLP",
    "LastBlock - Adapter",
    "FullFT - Adapter",
]

fig, ax = plt.subplots(figsize=(9.8, 6.2))
budget_x = np.arange(len(BUDGET_ORDER))

for idx, contrast in enumerate(contrast_order):
    sub = paired[paired["contrast"] == contrast].copy()
    means = []
    lo = []
    hi = []

    for b in BUDGET_ORDER:
        z = sub[sub["budget"] == b]["mean_delta"].values.astype(float)
        means.append(np.mean(z))
        lo.append(np.quantile(z, 0.25))
        hi.append(np.quantile(z, 0.75))

    color = mpl.cm.viridis(0.15 + 0.20 * idx)
    ax.plot(
        budget_x, means,
        marker="o",
        markersize=8,
        lw=2.4,
        color=color,
        label=contrast,
        zorder=3
    )
    ax.fill_between(
        budget_x, lo, hi,
        color=color,
        alpha=0.18,
        zorder=2
    )

    for xx, yy in zip(budget_x, means):
        boxed_text(ax, xx, yy + 0.008, f"{yy:+.3f}", fontsize=8, va="bottom")

ax.axhline(0.0, color="black", lw=1.1)
ax.set_xticks(budget_x)
ax.set_xticklabels([BUDGET_LABEL[b] for b in BUDGET_ORDER])
ax.set_xlabel("Training budget")
ax.set_ylabel("Condition-mean Δ macro-F1")
ax.set_title("Contrast behavior across budgets")
ax.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
save(fig, "13_contrast_by_budget")

# ------------------------------------------------------------
# 14. Performance landscape
# rows = methods, cols = 16 conditions
# ------------------------------------------------------------
def short_cond_label(ds, bb, b):
    ds_s = "AQ" if ds == "aqua20" else "FN"
    bb_s = "R18" if bb == "resnet18" else "DINO"
    return f"{ds_s}-{bb_s}\\n{BUDGET_LABEL[b]}"

cond_labels = []
mat = []

for m in METHOD_ORDER:
    row = []
    for ds, bb in PANEL_ORDER:
        for b in BUDGET_ORDER:
            z = core[
                (core["dataset"] == ds) &
                (core["backbone"] == bb) &
                (core["budget"] == b) &
                (core["method"] == m)
            ]
            row.append(float(z["test_macro_f1_mean"].iloc[0]))
    mat.append(row)

for ds, bb in PANEL_ORDER:
    for b in BUDGET_ORDER:
        cond_labels.append(short_cond_label(ds, bb, b))

mat = np.array(mat)

fig, ax = plt.subplots(figsize=(17.0, 6.2))
fig.subplots_adjust(left=0.12, right=0.92, bottom=0.29, top=0.87)

im = ax.imshow(
    mat,
    cmap=mpl.cm.viridis,
    aspect="auto",
    alpha=0.97,
)

# thin grid
for i in range(mat.shape[0] + 1):
    ax.axhline(i - 0.5, color="black", lw=0.7, zorder=5)
for j in range(mat.shape[1] + 1):
    ax.axvline(j - 0.5, color="black", lw=0.35, zorder=5)

# thicker separators between the 4 panel-groups
for j in [4, 8, 12]:
    ax.axvline(j - 0.5, color="black", lw=1.4, zorder=6)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        boxed_text(
            ax,
            j,
            i,
            f"{val:.3f}",
            fontsize=8.2,
            red=True,
            pad=0.12,
            alpha=0.80,
        )

ax.set_yticks(np.arange(len(METHOD_ORDER)))
ax.set_yticklabels([METHOD_LONG[m] for m in METHOD_ORDER])
ax.set_xticks(np.arange(len(cond_labels)))
ax.set_xticklabels(cond_labels, rotation=0, ha="center")
ax.set_title("Performance landscape across all 16 core conditions")

cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.035)
cbar.set_label("Mean test macro-F1")

save(fig, "14_performance_landscape")

print("WROTE FANCY FIGURES TO:", OUT)
for p in sorted(OUT.iterdir()):
    print(p.name)

# rows = methods, cols = 16 conditions
# ------------------------------------------------------------
cond_labels = []
mat = []

for m in METHOD_ORDER:
    row = []
    for ds, bb in PANEL_ORDER:
        for b in BUDGET_ORDER:
            z = core[
                (core["dataset"] == ds) &
                (core["backbone"] == bb) &
                (core["budget"] == b) &
                (core["method"] == m)
            ]
            row.append(float(z["test_macro_f1_mean"].iloc[0]))
            cond_labels.append((ds, bb, b))
        # cond_labels gets duplicated if inside outer loop; fix later
    mat.append(row)

# rebuild condition labels properly
cond_labels = []
for ds, bb in PANEL_ORDER:
    for b in BUDGET_ORDER:
        cond_labels.append(f"{DATASET_NAME[ds]}\n{BACKBONE_NAME[bb]}\n{BUDGET_LABEL[b]}")

mat = np.array(mat)

fig, ax = plt.subplots(figsize=(15.8, 5.6))
im = ax.imshow(mat, cmap=mpl.cm.viridis, aspect="auto", alpha=0.95)

for i in range(mat.shape[0] + 1):
    ax.axhline(i - 0.5, color="black", lw=0.6)
for j in range(mat.shape[1] + 1):
    ax.axvline(j - 0.5, color="black", lw=0.35)

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat[i, j]
        t = ax.text(
            j, i, f"{val:.3f}",
            ha="center", va="center",
            fontsize=9.2,
            color="black",
        )
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white", alpha=0.75)])

ax.set_yticks(np.arange(len(METHOD_ORDER)))
ax.set_yticklabels([METHOD_LONG[m] for m in METHOD_ORDER])
ax.set_xticks(np.arange(len(cond_labels)))
ax.set_xticklabels(cond_labels, rotation=45, ha="right")
ax.set_title("Performance landscape across all 16 core conditions")
cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.set_label("Mean test macro-F1")
fig.tight_layout()
save(fig, "14_performance_landscape")

print("WROTE FANCY FIGURES TO:", OUT)
for p in sorted(OUT.iterdir()):
    print(p.name)
