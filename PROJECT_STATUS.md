# PROJECT STATUS / HANDOFF — IITK Galaxy Bar Detection

**Read this first when you come back to the project.** Last updated after the
regression breakthrough. Everything below is the current, verified state.

---

## 1. WHAT THIS PROJECT IS (one paragraph)

Train a CNN to detect (and measure) **stellar bars** in galaxy images rendered
from a GADGET N-body simulation. Labels come from physics — each snapshot's
A2/A0 Fourier bar-strength (in `extras/bar_labels.csv`), no manual labeling.
The journey: binary classifier 76.5% → 86.4% → 91.8% (with TTA+ensemble) →
**reframed as regression → 96.0%** (the final model). The final model doesn't
just classify; it *measures* bar strength to ±0.02.

**Final model = ResNet50 regression. 96.0% accuracy, MAE 0.0217.**

---

## 2. THE FINAL RESULT (what to quote)

| Model | Accuracy | Notes |
|---|---|---|
| **Regression ResNet50 (FINAL)** | **96.0%** | MAE 0.022; measures bar strength |
| v3 binary + TTA + 3-seed ensemble | 91.8% | earlier best classifier |
| v3 binary (single) | 86.4% | the baseline regression beat |
| v4 kinematic (velocity channels) | 83.3% | NEGATIVE result — didn't help |

Why regression won (+9.6 over binary): switching from bar/no-bar labels to the
continuous A2/A0 value (a) unlocked 60 mid-strength snapshots binary labeling
had to discard, (b) cured "boundary blindness" (binary never saw the 0.10-0.22
region where the 0.19 threshold lives), (c) gave richer supervision per image.
No overfitting: train MAE (0.025) == test MAE (0.022).

---

## 3. WHERE EVERYTHING IS

### A) LOCAL — this folder `E:\IIT KNP PROJ\windowpynbody2\`
```
code/                  35 scripts (pipeline + analysis + legacy physics)
results/               every experiment's reports/figures/logs (the LEARNINGS)
  regression_resnet50/   ★ THE FINAL MODEL (model.weights.h5, 283MB) + reports
  regression_convnext/, regression_effnetv2b0/   other archs (reports only, weights deleted)
  regression_ensemble/   final comparison report + pred_vs_true.png scatter
  v1_binary/ v2_binary/ v3_binary/ v3_binary_seed1|2/   binary history (reports only)
  v4_kinematic/          the negative-result evidence
data/
  dataset_v3_binary/     the 3,050-image dataset (for evaluation)
  dataset_mid_regression/ the 1,500 mid-strength images
gadget_snapshots/      raw simulation binaries snapshot_000..501 (~48GB, git-ignored)
extras/bar_labels.csv  THE LABEL SOURCE (A2/A0 per snapshot)
images/ pdf/           original astronomy visualizations + reference papers
veny/                  Python virtualenv  (⚠ SEE WARNING below)
.env                   HF write token (git-ignored)
```
Only the FINAL model's weights are kept locally (others were pushed to HF then
deleted to save ~1GB). All reports/figures for every experiment are kept.

### B) GITHUB (public, code + reports, no big files)
https://github.com/GuptaOum/IITK_astronomy_project
Has all of `code/` and `results/` (reports/figures/logs). README tells the full
story. Weights, datasets, snapshots are git-ignored (see HuggingFace for weights).

### C) HUGGINGFACE (public, the model weights)
https://huggingface.co/kjfk/galaxy-bar-detection-resnet50
- `regression_resnet50.weights.h5`  ← THE FINAL MODEL
- `bar_resnet50_v1/v2/v3.keras`      ← binary history
- model card (README) with load instructions, `pred_vs_true_final.png`, reports

### D) AWS EC2 (two instances — CHECK/STOP THESE)
| Name | ID | Type | Purpose |
|---|---|---|---|
| iitk-gpu-regression | i-0e7df6c19e64f679e | g4dn.xlarge (T4 GPU) | training |
| iitk-v4-render-train | i-0659350225e374908 | c7i.2xlarge (CPU) | rendering, holds 182 snapshots |
- Key for BOTH: `C:\Users\hp\.ssh\face-attendance.pem`, user `ubuntu`.
- IPs CHANGE every stop/start — look them up in AWS console, don't trust old IPs.
- ⚠ As of writing, the GPU box was left RUNNING (billing ~₹44/hr). STOP it:
  `aws ec2 stop-instances --instance-ids i-0e7df6c19e64f679e`
- These are only needed for NEW training. All results are already downloaded.
  Safe to terminate both if fully done (you'd re-upload snapshots for future runs).

---

## 4. HOW TO USE THE MODEL (inference)

From this folder:
```
.\veny\Scripts\python.exe code\predict.py gadget_snapshots\snapshot_350
```
Outputs: `BAR / NO BAR  (predicted bar strength A2/A0 = 0.214, threshold 0.19)`.
Also accepts a rendered grayscale PNG. See `code/predict.py` header for details.

To reproduce the 96.0% report:  `python code\eval_regression.py`
(train/val splits: `python code\eval_regression.py train`)

---

## 5. ⚠ WARNINGS / GOTCHAS FOR FUTURE YOU

- **The venv (`veny/`) may be broken.** Its base Python (a Windows Store
  Python 3.13) moved, so `veny\Scripts\python.exe` currently errors. Fix: either
  recreate the venv (`python -m venv veny2 && veny2\Scripts\pip install tensorflow
  pynbody pandas matplotlib scipy huggingface_hub`) or repoint it.
- **Model saved as WEIGHTS ONLY** (`.weights.h5`), not a full `.keras` file —
  a Keras serialization bug. To load: rebuild the architecture then
  `load_weights()`. The recipe is in `code/predict.py` (build_model) and the HF
  model card.
- **Inference must match training render EXACTLY** (grayscale density, log clip
  1e7-1e9, 224px, 40kpc) or predictions are garbage. predict.py handles this.
- **Rotate the HuggingFace write token** in `.env` — it was pasted in a chat.

---

## 6. IF YOU WANT TO DO MORE (optional next steps)

1. **Ablation**: train regression on v3-data-only (no mid images) to measure how
   much the mid data vs the regression framing each contributed. ~15 min on GPU.
2. **Scratch-CNN baseline** (no ImageNet) — answers "did transfer learning help?"
3. **>40° edge-on test** — where does the model finally break beyond training range.
4. **More galaxies** — everything is ONE simulated galaxy evolving; other
   simulation runs would be the real ceiling-raiser. Ask professor.

---

## 7. FOR AN ML INTERVIEW / PRESENTATION

The story to tell: diagnosis over guessing (accuracy-vs-tilt showed labels, not
model, were wrong at high tilt), experimental controls (same-test-set
comparisons), honest negatives (kinematics failed, ensembles diluted), and the
winning insight — **how you pose the problem (classification vs regression)
matters more than the model**. Numbers to know: 76.5→86.4→91.8→96.0, MAE 0.022,
502 snapshots, 3,200 training images. Known limitation: one simulated galaxy;
sim-to-real gap untested.
