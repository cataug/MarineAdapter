# MarineAdapter

**MarineAdapter** is a parameter-efficient framework for label-efficient underwater marine image recognition.  
The method uses a frozen pretrained visual backbone and trains only a compact residual feature adapter together with a classification head.

The repository contains the code, notebooks, logs, and experiment outputs used for the MarineAdapter experiments.  
Datasets, pretrained model weights, and downloaded image folders are intentionally excluded from the repository.

---

## Overview

Underwater marine image recognition is affected by several practical constraints:

- underwater domain shift;
- color attenuation, blur, haze, and low contrast;
- fine-grained and visually similar marine categories;
- class imbalance;
- limited expert annotation.

MarineAdapter treats the task as **label-efficient image-level recognition**. Instead of training all parameters from scratch or fully fine-tuning the whole backbone, the method freezes a pretrained visual encoder and learns a lightweight residual adapter in the feature space.

Given an image \(x\), the frozen encoder extracts a feature vector:

\[
F = E_v(x).
\]

The adapter applies a residual feature correction:

\[
\tilde{F} = \mathrm{LN}(F + \mathcal{A}_{\theta}(F)).
\]

The adapted feature is passed to a classification head:

\[
z = g(\tilde{F}), \qquad \hat{y} = \arg\max_k z_k.
\]

This design is positioned between:

- **linear probing**, where only the classifier is trained;
- **full fine-tuning**, where the whole backbone is updated.

---

## Repository structure

```text
MarineAdapter/
├── run_marine_adapter_experiments.py     # Main experiment script
├── launch_marine_dynamic_gpu.py          # Dynamic GPU launcher
├── download_fathomnet_real.py            # FathomNet download script
├── Models.ipynb                          # Notebook with model/download checks
├── run_marineov_clip_experiment.py        # Earlier CLIP/Open-Vocabulary experiment script
├── marineov_clip_run.log                 # Earlier CLIP experiment log
├── MARINE_EXPERIMENTS/                   # Experiment-related files
├── MARINE_RESULTS/                       # Saved experiment results
├── MARINE_RESULTS_DYNAMIC/               # Dynamic launcher outputs
└── .gitignore
