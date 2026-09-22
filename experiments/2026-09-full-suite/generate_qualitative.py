from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


# =====================================================================
# Paths
# =====================================================================

BASE = Path("/home/tahiti/MARINE_DATASETS")

ROOT = (
    BASE /
    "MARINE_EXPERIMENTS_V2" /
    "FULL_SUITE"
)

OUT = ROOT / "QUALITATIVE"

OUT_AQ = OUT / "AQUA20"
OUT_FN = OUT / "FathomNet"
OUT_SEL = OUT / "selections"

for p in [OUT, OUT_AQ, OUT_FN, OUT_SEL]:
    p.mkdir(parents=True, exist_ok=True)


MANIFESTS = {
    "aqua20": (
        BASE /
        "MARINE_EXPERIMENTS_V2" /
        "manifests" /
        "aqua20"
    ),
    "fathomnet": (
        BASE /
        "MARINE_EXPERIMENTS_V2" /
        "manifests" /
        "fathomnet"
    ),
}


DATA_ROOTS = {
    "aqua20": [
        BASE / "AQUA20_PREPARED",
        BASE / "AQUA20",
        BASE,
    ],
    "fathomnet": [
        BASE / "FathomNet_FGVC2025",
        BASE,
    ],
}


# =====================================================================
# Style / constants
# =====================================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 20,
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


SHORT_METHOD = {
    "linear_probe": "Linear",
    "mlp_head": "MLP",
    "marine_adapter": "Adapter",
    "last_block_ft": "Last-block",
    "full_finetune": "Full FT",
}


DATASET_NAME = {
    "aqua20": "AQUA20",
    "fathomnet": "FathomNet",
}


BACKBONE_NAME = {
    "resnet18": "ResNet-18",
    "dinov2_small": "DINOv2-S",
}


BUDGETS = ["001", "005", "010", "100"]


BUDGET_NAME = {
    "001": "1%",
    "005": "5%",
    "010": "10%",
    "100": "100%",
}


AQUA_NAMES = {
    0: "coral",
    1: "crab",
    2: "diver",
    3: "eel",
    4: "fish",
    5: "fishInGroups",
    6: "flatworm",
    7: "jellyfish",
    8: "marine_dolphin",
    9: "octopus",
    10: "rayfish",
    11: "seaAnemone",
    12: "seaCucumber",
    13: "seaSlug",
    14: "seaUrchin",
    15: "shark",
    16: "shrimp",
    17: "squid",
    18: "starfish",
    19: "turtle",
}


VIR = mpl.cm.viridis

CORRECT_EDGE = "#18864B"
WRONG_EDGE = "#D62728"
GT_EDGE = "#D62728"


def normalize_budget(v):
    s = str(v).strip()

    try:
        n = int(float(s))
        if n in {1, 5, 10, 100}:
            return f"{n:03d}"
    except Exception:
        pass

    return s


# =====================================================================
# Result discovery
# =====================================================================

run_index = defaultdict(list)

for result_path in ROOT.rglob("result.json"):
    try:
        r = json.loads(result_path.read_text())
    except Exception:
        continue

    if r.get("stage") != "core":
        continue

    if r.get("status") != "complete":
        continue

    dataset = str(r.get("dataset"))
    backbone = str(r.get("backbone"))
    method = str(r.get("method"))
    budget = normalize_budget(r.get("budget"))
    seed = int(r.get("seed"))

    pred_path = result_path.parent / "test_predictions.csv"

    if not pred_path.exists():
        continue

    key = (
        dataset,
        backbone,
        method,
        budget,
    )

    run_index[key].append({
        "seed": seed,
        "prediction_file": pred_path,
        "result_file": result_path,
    })


print(
    "core conditions with prediction files:",
    len(run_index),
    flush=True,
)


# =====================================================================
# Manifest handling
# =====================================================================

def load_manifest(dataset, split="test"):
    p = MANIFESTS[dataset] / f"{split}.csv"

    if not p.exists():
        raise FileNotFoundError(p)

    return pd.read_csv(p)


def find_label_column(df):
    candidates = [
        "label_id",
        "category_id",
        "target",
        "label",
        "class_id",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"Cannot find label column. Columns={list(df.columns)}"
    )


def find_path_column(df):
    candidates = [
        "roi_path",
        "path",
        "image_path",
        "filepath",
        "file",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        f"Cannot find image path column. Columns={list(df.columns)}"
    )


def resolve_path(dataset, value):
    p = Path(str(value))

    if p.exists():
        return p

    for root in DATA_ROOTS[dataset]:
        q = root / p

        if q.exists():
            return q

    return p


def build_class_names(dataset, *dfs):
    if dataset == "aqua20":
        return dict(AQUA_NAMES)

    name_candidates = [
        "category_name",
        "class_name",
        "label_name",
        "concept_name",
        "concept",
        "name",
    ]

    mapping = {}

    for df in dfs:
        if df is None:
            continue

        label_col = find_label_column(df)

        name_col = None

        for c in name_candidates:
            if c in df.columns:
                name_col = c
                break

        if name_col is None:
            continue

        for _, r in df[[label_col, name_col]].dropna().iterrows():
            try:
                k = int(r[label_col])
            except Exception:
                continue

            mapping[k] = str(r[name_col])

    return mapping


# =====================================================================
# Prediction handling
# =====================================================================

def detect_prediction_columns(df):
    true_candidates = [
        "y_true",
        "true",
        "target",
        "label",
        "true_label",
        "target_id",
        "label_id",
        "category_id",
    ]

    pred_candidates = [
        "y_pred",
        "pred",
        "prediction",
        "pred_label",
        "pred_id",
        "prediction_id",
    ]

    true_col = None
    pred_col = None

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in true_candidates:
        if c in lower:
            true_col = lower[c]
            break

    for c in pred_candidates:
        if c in lower:
            pred_col = lower[c]
            break

    if true_col is None:
        for c in df.columns:
            lc = str(c).lower()

            if (
                "true" in lc
                or "target" in lc
            ):
                true_col = c
                break

    if pred_col is None:
        for c in df.columns:
            lc = str(c).lower()

            if (
                "pred" in lc
                and "prob" not in lc
            ):
                pred_col = c
                break

    if true_col is None or pred_col is None:
        raise RuntimeError(
            "Cannot detect prediction columns: "
            f"{list(df.columns)}"
        )

    return true_col, pred_col


_prediction_cache = {}


def read_prediction_file(path):
    path = Path(path)

    if path in _prediction_cache:
        return _prediction_cache[path]

    df = pd.read_csv(path)

    true_col, pred_col = detect_prediction_columns(df)

    # Prediction logs may store either integer IDs or human-readable
    # class names (e.g. "coral").  For qualitative analysis we keep a
    # canonical string representation; this also makes majority voting
    # independent of the label encoding used by a dataset.
    y_true = (
        df[true_col]
        .astype(str)
        .str.strip()
        .to_numpy(dtype=object)
    )

    y_pred = (
        df[pred_col]
        .astype(str)
        .str.strip()
        .to_numpy(dtype=object)
    )

    _prediction_cache[path] = (y_true, y_pred)

    return y_true, y_pred


def majority_vote(arrays):
    stack = np.stack(arrays, axis=0)

    n_seed, n = stack.shape

    vote = np.empty(n, dtype=object)
    agreement = np.empty(n, dtype=float)

    for i in range(n):
        vals, counts = np.unique(
            stack[:, i],
            return_counts=True,
        )

        k = int(np.argmax(counts))

        vote[i] = vals[k]
        agreement[i] = float(counts[k]) / n_seed

    return vote, agreement


def load_method_votes(
    dataset,
    backbone,
    budget,
    expected_n,
):
    out = {}

    shared_true = None

    for method in METHODS:
        key = (
            dataset,
            backbone,
            method,
            budget,
        )

        runs = sorted(
            run_index.get(key, []),
            key=lambda x: x["seed"],
        )

        if len(runs) != 5:
            print(
                "WARNING:",
                key,
                "prediction seeds=",
                len(runs),
                flush=True,
            )

        pred_arrays = []

        for run in runs:
            y_true, y_pred = read_prediction_file(
                run["prediction_file"]
            )

            if len(y_true) != expected_n:
                raise RuntimeError(
                    f"Length mismatch {key}: "
                    f"{len(y_true)} != {expected_n}"
                )

            if shared_true is None:
                shared_true = y_true.copy()

            elif not np.array_equal(
                shared_true,
                y_true,
            ):
                raise RuntimeError(
                    f"y_true differs across runs for {key}"
                )

            pred_arrays.append(y_pred)

        if not pred_arrays:
            continue

        vote, agreement = majority_vote(
            pred_arrays
        )

        out[method] = {
            "pred": vote,
            "agreement": agreement,
            "n_seeds": len(pred_arrays),
        }

    return shared_true, out


# =====================================================================
# Visual helpers
# =====================================================================

def class_name(mapping, label):
    # Prediction files may already contain class names.
    s = str(label).strip()

    # Numeric-looking label -> resolve through dataset mapping.
    try:
        f = float(s)
        if f.is_integer():
            k = int(f)
            if k in mapping:
                return str(mapping[k])
            return f"class {k}"
    except Exception:
        pass

    # Already a human-readable label.
    return s


def clipped_name(s, n=26):
    s = str(s)

    if len(s) <= n:
        return s

    return s[:n - 1] + "…"


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img

    except Exception:
        return Image.new(
            "RGB",
            (512, 384),
            color=(225, 225, 225),
        )


def add_cell_background(ax, strength):
    strength = float(
        np.clip(strength, 0, 1)
    )

    color = VIR(
        0.10 + 0.82 * strength
    )

    rect = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        facecolor=color,
        alpha=0.20,
        edgecolor="black",
        linewidth=0.8,
        clip_on=False,
    )

    ax.add_patch(rect)


def prediction_bbox(
    ax,
    text,
    correct,
    agreement,
):
    edge = (
        CORRECT_EDGE
        if correct
        else WRONG_EDGE
    )

    symbol = "✓" if correct else "×"

    seed_txt = (
        f"{int(round(agreement * 5))}/5 seeds"
    )

    ax.text(
        0.5,
        0.57,
        f"{symbol} {clipped_name(text)}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor=(1, 1, 1, 0.84),
            edgecolor=edge,
            linewidth=1.30,
        ),
        zorder=10,
    )

    ax.text(
        0.5,
        0.28,
        seed_txt,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )


def save(fig, path_without_ext):
    fig.savefig(
        str(path_without_ext) + ".png",
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        str(path_without_ext) + ".pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Dataset overview
# =====================================================================

def dataset_overview(dataset, n_examples=16):
    train = load_manifest(dataset, "train")
    test = load_manifest(dataset, "test")

    label_col = find_label_column(test)
    path_col = find_path_column(test)

    mapping = build_class_names(
        dataset,
        train,
        test,
    )

    train_label_col = find_label_column(train)

    counts = (
        train[train_label_col]
        .value_counts()
        .sort_values()
    )

    test_labels = set(
        test[label_col]
        .astype(int)
        .unique()
    )

    available = [
        int(x)
        for x in counts.index
        if int(x) in test_labels
    ]

    if len(available) > n_examples:
        ix = np.linspace(
            0,
            len(available) - 1,
            n_examples,
        ).round().astype(int)

        chosen_classes = [
            available[i]
            for i in ix
        ]

    else:
        chosen_classes = available

    rng = np.random.default_rng(
        20260922
    )

    chosen_rows = []

    for cls in chosen_classes:
        z = test[
            test[label_col].astype(int) == cls
        ]

        if len(z) == 0:
            continue

        row = z.iloc[
            int(rng.integers(0, len(z)))
        ]

        chosen_rows.append(
            (
                cls,
                row,
                int(counts.loc[cls]),
            )
        )

    n = len(chosen_rows)

    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15.0, 3.8 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (cls, row, n_train) in zip(
        axes,
        chosen_rows,
    ):
        path = resolve_path(
            dataset,
            row[path_col],
        )

        img = load_image(path)

        ax.imshow(img)
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)

        ax.text(
            0.5,
            -0.04,
            clipped_name(
                class_name(mapping, cls),
                30,
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=(1, 1, 1, 0.88),
                edgecolor=GT_EDGE,
                linewidth=1.20,
            ),
        )

        ax.text(
            0.5,
            -0.17,
            f"train n={n_train}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.suptitle(
        f"{DATASET_NAME[dataset]} — qualitative dataset overview\n"
        "classes sampled from rare to common",
        y=1.01,
    )

    out_dir = (
        OUT_AQ
        if dataset == "aqua20"
        else OUT_FN
    )

    save(
        fig,
        out_dir / "00_dataset_overview",
    )


# =====================================================================
# Method comparison grid
# =====================================================================

def comparison_grid(
    dataset,
    backbone,
    budget,
    indices,
    stem,
    subtitle,
):
    test = load_manifest(dataset, "test")
    train = load_manifest(dataset, "train")

    label_col = find_label_column(test)
    path_col = find_path_column(test)

    mapping = build_class_names(
        dataset,
        train,
        test,
    )

    y_true, votes = load_method_votes(
        dataset,
        backbone,
        budget,
        len(test),
    )

    if y_true is None:
        return

    indices = list(indices)

    if len(indices) == 0:
        print(
            "NO EXAMPLES:",
            dataset,
            backbone,
            budget,
            stem,
        )
        return

    indices = indices[:8]

    nrows = len(indices)
    ncols = 1 + len(METHODS)

    fig = plt.figure(
        figsize=(17.8, 2.55 * nrows + 1.2)
    )

    gs = fig.add_gridspec(
        nrows,
        ncols,
        width_ratios=[
            2.15,
            1,
            1,
            1,
            1,
            1,
        ],
        hspace=0.22,
        wspace=0.08,
    )

    selection_records = []

    for ri, idx in enumerate(indices):

        row = test.iloc[int(idx)]

        gt = y_true[idx]

        # image
        ax = fig.add_subplot(
            gs[ri, 0]
        )

        path = resolve_path(
            dataset,
            row[path_col],
        )

        img = load_image(path)

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.85)

        gt_name = class_name(
            mapping,
            gt,
        )

        ax.text(
            0.5,
            0.03,
            f"GT: {clipped_name(gt_name, 32)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.27",
                facecolor=(1, 1, 1, 0.86),
                edgecolor=GT_EDGE,
                linewidth=1.20,
            ),
        )

        if ri == 0:
            ax.set_title(
                "Image / ground truth",
                fontsize=13,
                fontweight="bold",
                pad=9,
            )

        record = {
            "dataset": dataset,
            "backbone": backbone,
            "budget": budget,
            "index": int(idx),
            "path": str(path),
            "ground_truth_id": gt,
            "ground_truth": gt_name,
            "selection": stem,
        }

        # prediction cells
        for ci, method in enumerate(
            METHODS,
            start=1,
        ):
            axp = fig.add_subplot(
                gs[ri, ci]
            )

            axp.set_xlim(0, 1)
            axp.set_ylim(0, 1)
            axp.set_xticks([])
            axp.set_yticks([])

            p = votes[method]["pred"][idx]

            agr = float(
                votes[method]["agreement"][idx]
            )

            add_cell_background(
                axp,
                agr,
            )

            prediction_bbox(
                axp,
                class_name(mapping, p),
                p == gt,
                agr,
            )

            if ri == 0:
                axp.set_title(
                    METHOD_NAME[method],
                    fontsize=13,
                    fontweight="bold",
                    pad=9,
                )

            record[
                f"{method}_pred_id"
            ] = p

            record[
                f"{method}_pred"
            ] = class_name(
                mapping,
                p,
            )

            record[
                f"{method}_agreement"
            ] = agr

            record[
                f"{method}_correct"
            ] = int(p == gt)

        selection_records.append(
            record
        )

    fig.suptitle(
        f"{DATASET_NAME[dataset]} / "
        f"{BACKBONE_NAME[backbone]} / "
        f"{BUDGET_NAME[budget]} labels\n"
        f"{subtitle} — majority vote across five seeds",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )

    out_dir = (
        OUT_AQ
        if dataset == "aqua20"
        else OUT_FN
    )

    name = (
        f"{stem}_"
        f"{backbone}_"
        f"b{budget}"
    )

    save(
        fig,
        out_dir / name,
    )

    pd.DataFrame(
        selection_records
    ).to_csv(
        OUT_SEL / f"{dataset}_{name}.csv",
        index=False,
    )


# =====================================================================
# Selection rules
# =====================================================================

def get_condition_arrays(
    dataset,
    backbone,
    budget,
):
    test = load_manifest(
        dataset,
        "test",
    )

    y, votes = load_method_votes(
        dataset,
        backbone,
        budget,
        len(test),
    )

    return test, y, votes


def rank_disagreement(
    y,
    votes,
):
    n = len(y)

    rows = []

    for i in range(n):
        preds = np.array([
            votes[m]["pred"][i]
            for m in METHODS
        ])

        correct = preds == y[i]

        unique = len(
            np.unique(preds)
        )

        mixed = int(
            correct.any()
            and (~correct).any()
        )

        mean_agreement = np.mean([
            votes[m]["agreement"][i]
            for m in METHODS
        ])

        score = (
            10 * mixed
            + unique
            + 0.20 * mean_agreement
        )

        rows.append(
            (score, i)
        )

    rows.sort(
        reverse=True
    )

    return [
        i for _, i in rows
    ]


def adapter_rescue_indices(
    y,
    votes,
):
    lp = votes[
        "linear_probe"
    ]["pred"]

    ad = votes[
        "marine_adapter"
    ]["pred"]

    mask = (
        (lp != y)
        & (ad == y)
    )

    inds = np.where(mask)[0]

    return sorted(
        inds,
        key=lambda i: (
            votes["marine_adapter"]["agreement"][i]
            + votes["linear_probe"]["agreement"][i]
        ),
        reverse=True,
    )


def mlp_rescue_indices(
    y,
    votes,
):
    ad = votes[
        "marine_adapter"
    ]["pred"]

    mlp = votes[
        "mlp_head"
    ]["pred"]

    mask = (
        (ad != y)
        & (mlp == y)
    )

    inds = np.where(mask)[0]

    return sorted(
        inds,
        key=lambda i: (
            votes["mlp_head"]["agreement"][i]
            + votes["marine_adapter"]["agreement"][i]
        ),
        reverse=True,
    )


def deep_rescue_indices(
    y,
    votes,
):
    ad = votes[
        "marine_adapter"
    ]["pred"]

    mlp = votes[
        "mlp_head"
    ]["pred"]

    last = votes[
        "last_block_ft"
    ]["pred"]

    full = votes[
        "full_finetune"
    ]["pred"]

    mask = (
        (ad != y)
        & (mlp != y)
        & (
            (last == y)
            | (full == y)
        )
    )

    inds = np.where(mask)[0]

    return sorted(
        inds,
        key=lambda i: max(
            votes["last_block_ft"]["agreement"][i],
            votes["full_finetune"]["agreement"][i],
        ),
        reverse=True,
    )


# =====================================================================
# Budget-evolution grid
# =====================================================================

def budget_evolution(
    dataset,
    backbone,
    method,
    n_examples=8,
):
    test = load_manifest(
        dataset,
        "test",
    )

    train = load_manifest(
        dataset,
        "train",
    )

    label_col = find_label_column(test)
    path_col = find_path_column(test)

    mapping = build_class_names(
        dataset,
        train,
        test,
    )

    budget_data = {}

    shared_y = None

    for budget in BUDGETS:
        y, votes = load_method_votes(
            dataset,
            backbone,
            budget,
            len(test),
        )

        if shared_y is None:
            shared_y = y

        budget_data[budget] = votes[
            method
        ]

    candidates = []

    for i in range(len(test)):
        gt = shared_y[i]

        preds = [
            budget_data[b]["pred"][i]
            for b in BUDGETS
        ]

        correct = np.array([
            p == gt
            for p in preds
        ])

        # Prefer clear low-budget failure -> high-budget success.
        score = (
            20 * int(
                (not correct[0])
                and correct[-1]
            )
            + 3 * int(correct[-1])
            + len(np.unique(preds))
            + correct.sum()
        )

        candidates.append(
            (score, i)
        )

    candidates.sort(
        reverse=True
    )

    indices = [
        i
        for _, i in candidates[:n_examples]
    ]

    nrows = len(indices)

    fig = plt.figure(
        figsize=(15.5, 2.55 * nrows + 1.0)
    )

    gs = fig.add_gridspec(
        nrows,
        5,
        width_ratios=[
            2.1,
            1,
            1,
            1,
            1,
        ],
        hspace=0.22,
        wspace=0.08,
    )

    records = []

    for ri, idx in enumerate(indices):
        row = test.iloc[idx]
        gt = shared_y[idx]

        ax = fig.add_subplot(
            gs[ri, 0]
        )

        path = resolve_path(
            dataset,
            row[path_col],
        )

        ax.imshow(
            load_image(path)
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.85)

        ax.text(
            0.5,
            0.03,
            f"GT: {clipped_name(class_name(mapping, gt), 30)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor=(1, 1, 1, 0.86),
                edgecolor=GT_EDGE,
                linewidth=1.2,
            ),
        )

        if ri == 0:
            ax.set_title(
                "Image / ground truth",
                fontsize=13,
                fontweight="bold",
            )

        rec = {
            "dataset": dataset,
            "backbone": backbone,
            "method": method,
            "index": idx,
            "path": str(path),
            "ground_truth": class_name(
                mapping,
                gt,
            ),
        }

        for ci, budget in enumerate(
            BUDGETS,
            start=1,
        ):
            axp = fig.add_subplot(
                gs[ri, ci]
            )

            axp.set_xlim(0, 1)
            axp.set_ylim(0, 1)
            axp.set_xticks([])
            axp.set_yticks([])

            pred = budget_data[
                budget
            ]["pred"][idx]

            agr = float(
                budget_data[
                    budget
                ]["agreement"][idx]
            )

            add_cell_background(
                axp,
                agr,
            )

            prediction_bbox(
                axp,
                class_name(mapping, pred),
                pred == gt,
                agr,
            )

            if ri == 0:
                axp.set_title(
                    BUDGET_NAME[budget],
                    fontsize=13,
                    fontweight="bold",
                )

            rec[
                f"{budget}_pred"
            ] = class_name(
                mapping,
                pred,
            )

            rec[
                f"{budget}_correct"
            ] = int(
                pred == gt
            )

        records.append(
            rec
        )

    fig.suptitle(
        f"{DATASET_NAME[dataset]} / "
        f"{BACKBONE_NAME[backbone]} — "
        f"{METHOD_NAME[method]}\n"
        "Prediction evolution as the annotation budget increases",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )

    out_dir = (
        OUT_AQ
        if dataset == "aqua20"
        else OUT_FN
    )

    stem = (
        f"05_budget_evolution_"
        f"{method}_"
        f"{backbone}"
    )

    save(
        fig,
        out_dir / stem,
    )

    pd.DataFrame(
        records
    ).to_csv(
        OUT_SEL /
        f"{dataset}_{stem}.csv",
        index=False,
    )


# =====================================================================
# Generate
# =====================================================================

print(
    "\n[1] DATASET OVERVIEWS",
    flush=True,
)

for dataset in [
    "aqua20",
    "fathomnet",
]:
    print(
        " overview:",
        dataset,
        flush=True,
    )

    dataset_overview(
        dataset,
    )


print(
    "\n[2] METHOD COMPARISONS",
    flush=True,
)

# 10%: richer causal/comparison panels
for dataset in [
    "aqua20",
    "fathomnet",
]:
    for backbone in [
        "resnet18",
        "dinov2_small",
    ]:
        budget = "010"

        print(
            dataset,
            backbone,
            budget,
            flush=True,
        )

        test, y, votes = get_condition_arrays(
            dataset,
            backbone,
            budget,
        )

        disagreement = rank_disagreement(
            y,
            votes,
        )

        comparison_grid(
            dataset,
            backbone,
            budget,
            disagreement,
            "01_method_disagreement",
            "high-disagreement examples",
        )

        comparison_grid(
            dataset,
            backbone,
            budget,
            adapter_rescue_indices(
                y,
                votes,
            ),
            "02_adapter_rescues_linear",
            "cases where MarineAdapter corrects the linear probe",
        )

        comparison_grid(
            dataset,
            backbone,
            budget,
            mlp_rescue_indices(
                y,
                votes,
            ),
            "03_mlp_rescues_adapter",
            "cases where the non-residual MLP corrects MarineAdapter",
        )

        comparison_grid(
            dataset,
            backbone,
            budget,
            deep_rescue_indices(
                y,
                votes,
            ),
            "04_deeper_adaptation_rescues",
            "cases requiring deeper backbone adaptation",
        )


print(
    "\n[3] FULL-BUDGET DISAGREEMENT",
    flush=True,
)

for dataset in [
    "aqua20",
    "fathomnet",
]:
    for backbone in [
        "resnet18",
        "dinov2_small",
    ]:
        budget = "100"

        test, y, votes = get_condition_arrays(
            dataset,
            backbone,
            budget,
        )

        comparison_grid(
            dataset,
            backbone,
            budget,
            rank_disagreement(
                y,
                votes,
            ),
            "01_method_disagreement",
            "high-disagreement examples at full supervision",
        )


print(
    "\n[4] BUDGET EVOLUTION",
    flush=True,
)

for dataset in [
    "aqua20",
    "fathomnet",
]:
    for backbone in [
        "resnet18",
        "dinov2_small",
    ]:
        for method in [
            "mlp_head",
            "last_block_ft",
        ]:
            print(
                " evolution:",
                dataset,
                backbone,
                method,
                flush=True,
            )

            budget_evolution(
                dataset,
                backbone,
                method,
            )


print()
print("=" * 86)
print("QUALITATIVE GENERATION COMPLETE")
print("=" * 86)
print("OUTPUT:", OUT)

for d in [
    OUT_AQ,
    OUT_FN,
    OUT_SEL,
]:
    print(
        d,
        "files=",
        len(list(d.glob("*"))),
    )
