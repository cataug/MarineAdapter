**Empirical analysis of pretrained visual adaptation for label-efficient underwater and marine image recognition.**

MarineAdapter started as a compact residual feature adapter for frozen pretrained visual encoders. The project has since grown into a controlled empirical study of a broader question:

> **How much of a pretrained visual representation should be adapted for marine image recognition when labels are scarce?**

The current repository contains the complete September 2026 experimental suite: scripts, split manifests, per-run logs, predictions, statistical summaries, quantitative and qualitative analyses, and publication-ready figures.

Datasets, downloaded image collections, pretrained model weights, and checkpoints are intentionally excluded.

---

## Main findings

The expanded study compares five adaptation strategies across two marine datasets, two pretrained backbone families, four label budgets, and five random seeds.

The central result is more nuanced than the original residual-adapter hypothesis:

- nonlinear feature adaptation **consistently improves over linear probing**;
- MarineAdapter beats the linear probe in **16/16 core conditions**;
- however, a parameter-matched non-residual MLP exceeds MarineAdapter in **15/16 conditions**;
- adapting the final backbone block exceeds MarineAdapter in **15/16 conditions**;
- full fine-tuning also exceeds MarineAdapter in **15/16 conditions**;
- full fine-tuning is **not universally superior** to last-block fine-tuning;
- the preferred adaptation depth depends strongly on the dataset, backbone, and annotation regime.

Across the 16 dataset/backbone/budget conditions:

| Contrast | Mean Δ macro-F1 | 95% bootstrap CI | W / T / L |
|---|---:|---:|---:|
| MarineAdapter − Linear probe | **+0.0526** | [+0.0312, +0.0804] | **16 / 0 / 0** |
| MarineAdapter − MLP | **−0.0397** | [−0.0601, −0.0227] | **1 / 0 / 15** |
| Last-block FT − MarineAdapter | **+0.0804** | [+0.0438, +0.1246] | **15 / 0 / 1** |
| Full FT − MarineAdapter | **+0.0922** | [+0.0546, +0.1311] | **15 / 0 / 1** |
| Full FT − Last-block FT | +0.0118 | [−0.0194, +0.0416] | 10 / 0 / 6 |

These results suggest that the benefit over a linear probe is primarily associated with **additional nonlinear adaptation capacity**, rather than with the residual connection alone.

---

## Experimental scope

The completed suite contains:

| Experiment block | Runs |
|---|---:|
| Core factorial study | **400** |
| Scratch-CNN baselines | **40** |
| Adapter-width ablation | **20** |
| Split-sensitivity analysis | **54** |
| Class-balanced k-shot study | **60** |
| Scientific runs | **574** |
| Technical preflight runs | **5** |
| **Total completed runs** | **579** |

All 579 recorded runs completed successfully.

### Core factorial design

The core study is:

```text
2 datasets
× 2 pretrained backbones
× 5 adaptation methods
× 4 label budgets
× 5 random seeds
= 400 runs
````

Random seeds:

```text
42, 43, 44, 45, 46
```

Label budgets:

```text
1%, 5%, 10%, 100%
```

Macro-F1 is the primary evaluation metric.

Accuracy and balanced accuracy are also reported.

---

## Datasets

### AQUA20

The canonical protocol uses all **8,171 labeled records** across **20 classes**.

| Split      |   Records |
| ---------- | --------: |
| Train      |     5,842 |
| Validation |     1,164 |
| Test       |     1,165 |
| **Total**  | **8,171** |

Classes include coral, crab, diver, eel, fish, jellyfish, octopus, rayfish, sea anemone, sea cucumber, sea slug, sea urchin, shark, shrimp, squid, starfish, turtle, and related categories.

The original data were audited for exact duplicates before experimentation. Exact-duplicate groups are kept within a single split to avoid cross-split leakage.

### FathomNet FGVC

The study uses the complete labeled training portion of the FathomNet FGVC benchmark:

* **23,699 labeled ROIs**
* **79 classes**
* derived from approximately 9k source images.

Canonical split:

| Split      | Labeled ROIs |
| ---------- | -----------: |
| Train      |       16,688 |
| Validation |        3,622 |
| Test       |        3,389 |
| **Total**  |   **23,699** |

The official competition test labels are not used for the local supervised evaluation. Instead, the full labeled benchmark training set is partitioned with group-aware splitting.

Exact duplicate ROIs were audited and duplicate groups were prevented from crossing train, validation, and test partitions.

---

## Models

### ResNet-18

A supervised ImageNet-pretrained ResNet-18 provides a 512-dimensional representation.

### DINOv2-Small

A self-supervised DINOv2-Small ViT provides a 384-dimensional representation.

Using both CNN and transformer-style pretrained encoders makes it possible to test whether adaptation behavior is backbone-specific.

---

## Adaptation strategies

Five pretrained adaptation regimes are evaluated.

### 1. Linear probe

The backbone is frozen and only a classification layer is trained:

```text
image → frozen backbone → linear classifier
```

This is the minimum-adaptation reference.

### 2. Parameter-matched MLP head

A nonlinear bottleneck transformation is inserted before the classifier:

```text
image
  → frozen backbone
  → Linear(d, r)
  → ReLU
  → Dropout
  → Linear(r, d)
  → LayerNorm
  → classifier
```

This control has approximately the same trainable feature-transformation capacity as MarineAdapter, but **without the residual connection**.

It is therefore essential for testing whether the residual connection itself is responsible for performance gains.

### 3. MarineAdapter

MarineAdapter uses the same compact bottleneck transformation with a residual feature correction:

```text
image
  → frozen backbone
  → feature F
  → F + Adapter(F)
  → LayerNorm
  → classifier
```

Conceptually,

```text
F = E(x)

F_adapted = LN(F + A(F))

y_hat = argmax g(F_adapted)
```

The backbone remains frozen.

### 4. Last-block fine-tuning

Only the final representation block and classification head are updated.

For ResNet-18 this corresponds to the final residual stage.

For DINOv2-Small the final transformer encoder block and associated normalization are adapted.

### 5. Full fine-tuning

The complete pretrained backbone and classifier are optimized.

---

## Scratch baseline

A four-stage convolutional classifier is additionally trained from scratch.

This provides a non-pretrained reference and demonstrates the importance of pretrained representations in the studied marine-recognition regimes.

At full supervision, scratch training remains substantially below the pretrained approaches:

```text
AQUA20     macro-F1 ≈ 0.152
FathomNet  macro-F1 ≈ 0.219
```

---

## Representative full-budget results

Mean test macro-F1 across five seeds:

| Dataset / Backbone    | Linear |       MLP | MarineAdapter | Last-block FT |   Full FT |
| --------------------- | -----: | --------: | ------------: | ------------: | --------: |
| AQUA20 / ResNet-18    |  0.425 | **0.651** |         0.645 |         0.626 |     0.643 |
| AQUA20 / DINOv2-S     |  0.812 |     0.853 |         0.834 |     **0.877** |     0.836 |
| FathomNet / ResNet-18 |  0.418 |     0.522 |         0.517 |         0.643 | **0.690** |
| FathomNet / DINOv2-S  |  0.638 |     0.668 |         0.666 |         0.729 | **0.810** |

The different orderings are important: no single adaptation depth dominates all dataset/backbone combinations.

---

## Label-scarcity behavior

The core experiments use nested 1%, 5%, 10%, and 100% subsets of the canonical training split.

Examples of the resulting pattern include:

### FathomNet / ResNet-18

| Budget | Linear | MarineAdapter |       MLP | Last-block FT |   Full FT |
| ------ | -----: | ------------: | --------: | ------------: | --------: |
| 1%     |  0.012 |         0.028 | **0.040** |         0.036 |     0.031 |
| 5%     |  0.059 |         0.127 |     0.160 |     **0.178** |     0.155 |
| 10%    |  0.148 |         0.233 |     0.271 |     **0.303** |     0.274 |
| 100%   |  0.418 |         0.517 |     0.522 |         0.643 | **0.690** |

### AQUA20 / DINOv2-S

| Budget | Linear | MarineAdapter |   MLP | Last-block FT |   Full FT |
| ------ | -----: | ------------: | ----: | ------------: | --------: |
| 1%     |  0.056 |         0.080 | 0.125 |         0.147 | **0.190** |
| 5%     |  0.134 |         0.147 | 0.229 |     **0.386** |     0.314 |
| 10%    |  0.226 |         0.241 | 0.394 |     **0.543** |     0.408 |
| 100%   |  0.812 |         0.834 | 0.853 |     **0.877** |     0.836 |

This illustrates why the project is now framed as an **adaptation-depth study**, rather than simply an adapter-versus-baseline comparison.

---

## Class-balanced k-shot study

A separate FathomNet / ResNet-18 experiment controls the exact number of labeled examples per class:

```text
1-shot
2-shot
5-shot
10-shot
```

Mean macro-F1:

| Shots / class | Linear | MarineAdapter |       MLP | Last-block FT | Full FT |
| ------------- | -----: | ------------: | --------: | ------------: | ------: |
| 1             |  0.009 |         0.013 | **0.023** |         0.017 |   0.015 |
| 2             |  0.010 |         0.020 | **0.034** |         0.028 |   0.025 |
| 5             |  0.024 |         0.047 | **0.066** |         0.065 |   0.056 |
| 10            |  0.060 |         0.119 |     0.154 |     **0.175** |   0.152 |

The results suggest a progression from lightweight nonlinear heads in the most extreme few-shot regime toward deeper backbone adaptation as additional labels become available.

---

## Adapter-width ablation

MarineAdapter bottleneck dimensions:

```text
r = 64, 128, 256
```

FathomNet / ResNet-18 results:

| Width | 10% macro-F1 | 100% macro-F1 |
| ----: | -----------: | ------------: |
|    64 |        0.204 |         0.505 |
|   128 |        0.233 |         0.517 |
|   256 |    **0.272** |     **0.530** |

Performance increases monotonically over the tested range.

The main factorial experiment retains `r=128` as a fixed moderate-capacity configuration rather than claiming it to be an optimum.

---

## Split sensitivity

To test whether conclusions depend on one favorable train/validation/test split, additional FathomNet / ResNet-18 experiments were performed for three alternative split seeds:

```text
123
456
789
```

At 10% supervision:

| Split | Linear | MarineAdapter |   Full FT |
| ----: | -----: | ------------: | --------: |
|   123 |  0.157 |         0.240 | **0.285** |
|   456 |  0.146 |         0.229 | **0.264** |
|   789 |  0.153 |         0.231 | **0.257** |

At full supervision:

| Split | Linear | MarineAdapter |   Full FT |
| ----: | -----: | ------------: | --------: |
|   123 |  0.432 |         0.535 | **0.698** |
|   456 |  0.428 |         0.532 | **0.691** |
|   789 |  0.419 |         0.514 | **0.672** |

The qualitative ordering persists across the alternative partitions.

---

## Statistics

The analysis reports:

* mean and standard deviation across seeds;
* 95% bootstrap confidence intervals;
* paired method differences;
* paired win/tie/loss counts;
* standardized paired effect sizes;
* exact sign-flip tests for seed-matched comparisons;
* Holm correction for collections of hypothesis tests;
* cross-condition consistency summaries.

With only five seeds per core condition, exact two-sided sign-flip tests have a minimum attainable p-value of `0.0625`.

For this reason, the study emphasizes:

1. paired effect magnitude;
2. confidence intervals;
3. consistency across seeds;
4. replication across datasets, backbones, budgets, and alternative splits;

rather than relying on a binary per-cell significance threshold.

---

## Figures

Publication-oriented figures are generated in several complementary styles.

### Main figures

```text
experiments/2026-09-full-suite/
└── FULL_SUITE/
    └── FIGURES_NEW/
```

Examples include:

* core macro-F1 grid;
* paired Adapter-vs-Linear forest plot;
* paired Adapter-vs-MLP forest plot;
* width ablation;
* split sensitivity;
* k-shot evaluation;
* scratch-vs-pretraining comparisons;
* trainable-parameter analysis.

### Fancy summary figures

```text
FULL_SUITE/FIGURES_NEW/FANCY/
```

These include:

* best-method maps;
* contrast heatmaps;
* performance–adaptation trade-offs;
* average method ranks;
* budget-dependent contrast behavior;
* complete performance landscapes.

### Quantitative figures

```text
FULL_SUITE/FIGURES_NEW/QUANTITATIVE/
```

These include:

* cross-condition effect summaries;
* pairwise dominance matrices;
* parameter-efficiency frontiers;
* quantitative budget-response curves;
* seed-variability landscapes;
* k-shot sample-efficiency curves.

All primary figures are exported in publication-ready PDF and high-resolution PNG formats.

---

## Qualitative analysis

The September suite also contains qualitative inspection utilities.

```text
FULL_SUITE/QUALITATIVE/
├── AQUA20/
├── FathomNet/
└── selections/
```

The qualitative pipeline uses saved test predictions rather than rerunning inference.

Examples are selected automatically using reproducible criteria such as:

* high disagreement among adaptation methods;
* cases where MarineAdapter corrects the linear probe;
* cases where the parameter-matched MLP corrects MarineAdapter;
* cases where deeper adaptation corrects both feature-level approaches;
* prediction changes as the label budget increases.

Predictions are aggregated across seeds, allowing the visualizations to report both the predicted class and seed-level agreement.

Selection manifests are saved as CSV files so that qualitative examples are auditable rather than manually cherry-picked.

---

## Repository layout

The repository contains both the original MarineAdapter experiments and the expanded September 2026 study.

```text
MarineAdapter/
│
├── README.md
├── .gitignore
│
├── run_marine_adapter_experiments.py
├── launch_marine_dynamic_gpu.py
├── download_fathomnet_real.py
├── Models.ipynb
│
├── run_marineov_clip_experiment.py
├── marineov_clip_run.log
│
├── MARINE_EXPERIMENTS/
├── MARINE_RESULTS/
├── MARINE_RESULTS_DYNAMIC/
│
└── experiments/
    └── 2026-09-full-suite/
        │
        ├── run_experiments.py
        ├── run_full_suite.py
        ├── analyze_full_suite.py
        ├── make_budgets_v2.py
        │
        ├── generate_figures_new.py
        ├── generate_figures_fancy.py
        ├── generate_figures_quantitative.py
        ├── generate_qualitative.py
        │
        ├── manifests/
        │   ├── aqua20/
        │   └── fathomnet/
        │
        └── FULL_SUITE/
            ├── all_results_full.csv
            ├── majority_baselines.csv
            │
            ├── STATS/
            │   ├── all_scientific_runs.csv
            │   ├── core_summary.csv
            │   ├── core_paired_statistics.csv
            │   ├── cross_condition_statistics.csv
            │   ├── scratch_summary.csv
            │   ├── sensitivity_summary.csv
            │   ├── shots_summary.csv
            │   ├── trainable_parameters.csv
            │   └── width_summary_with_r128.csv
            │
            ├── FIGURES_NEW/
            │   ├── FANCY/
            │   └── QUANTITATIVE/
            │
            └── QUALITATIVE/
                ├── AQUA20/
                ├── FathomNet/
                └── selections/
```

The top-level scripts and result folders document the earlier development of MarineAdapter.

The canonical expanded evaluation is the September 2026 suite under:

```text
experiments/2026-09-full-suite/
```

---

## Reproducing the September suite

The original experiment environment used:

```text
Python       3.12
PyTorch      2.6
torchvision  0.21
CUDA         12.4
NumPy        1.26
pandas       2.2
scikit-learn 1.5
transformers 4.46
```

The primary experiments were executed on an NVIDIA A100 40 GB GPU.

### Important

The repository does **not** include:

* AQUA20 image data;
* FathomNet image/ROI data;
* pretrained DINOv2 weights;
* torchvision model caches;
* checkpoints;
* virtual environments.

Dataset and model paths in the scripts correspond to the original experimental environment and should be adjusted when reproducing the suite elsewhere.

### Full experiment suite

From the September experiment directory:

```bash
cd experiments/2026-09-full-suite

python3 run_full_suite.py \
  --stages all \
  --epochs 10 \
  --batch-size 128 \
  --workers 2 \
  --retries 2
```

The runner supports resumption: completed runs are detected and skipped.

### Statistical analysis

```bash
python3 analyze_full_suite.py
```

### Main figures

```bash
python3 generate_figures_new.py
```

### Additional summary figures

```bash
python3 generate_figures_fancy.py
```

### Quantitative figures

```bash
python3 generate_figures_quantitative.py
```

### Qualitative analysis

```bash
python3 generate_qualitative.py
```

---

## Saved artifacts

The repository preserves the non-heavy experimental artifacts needed for auditing and analysis, including:

* experiment scripts;
* canonical split manifests;
* nested budget manifests;
* configuration metadata;
* per-run `result.json` files;
* training histories;
* test predictions;
* console logs;
* aggregate result tables;
* statistical summaries;
* figure-generation scripts;
* publication figures;
* qualitative-selection manifests.

Heavy datasets and model binaries are intentionally omitted.

---

## Earlier MarineAdapter experiments

The repository also preserves the original development pipeline, including:

```text
run_marine_adapter_experiments.py
launch_marine_dynamic_gpu.py
MARINE_EXPERIMENTS/
MARINE_RESULTS/
MARINE_RESULTS_DYNAMIC/
```

and earlier CLIP/open-vocabulary exploration:

```text
run_marineov_clip_experiment.py
marineov_clip_run.log
```

These files are retained for provenance.

For the current empirical study and reported multi-benchmark results, use the September 2026 suite.

---

## Interpretation

The expanded experiments change the interpretation of MarineAdapter.

The evidence supports the claim that **a frozen pretrained feature space often benefits from learnable nonlinear adaptation**.

It does not support the stronger claim that a residual bottleneck is generally superior to a parameter-matched nonlinear transformation.

Instead, the experiments indicate a broader adaptation hierarchy:

```text
Linear probe
    ↓
lightweight nonlinear feature adaptation
    ↓
partial backbone adaptation
    ↓
full fine-tuning
```

The best point on this hierarchy depends on:

* dataset size and composition;
* pretrained backbone;
* label budget;
* degree of domain shift;
* desired parameter efficiency.

This trade-off is the central empirical focus of the current project.

---

## Project status

**September 2026 experimental suite: complete.**

```text
579 / 579 runs complete
574 scientific runs
400 core factorial runs
```

The repository contains the corresponding analyses, logs, predictions, tables, and visual artifacts.


