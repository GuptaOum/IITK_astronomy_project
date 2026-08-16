# Project Conclusion — IITK Galaxy Bar Detection

*A CNN that detects and measures stellar bars in simulated galaxies.*

---

## What we did

Trained a CNN to find **stellar bars** in galaxy images rendered from a GADGET
N-body simulation (502 snapshots). Labels are physical, not manual — each
snapshot's A2/A0 Fourier bar-strength. The project went through five stages,
each fixing the previous one's weakness:

| Stage | Approach | Accuracy |
|---|---|---|
| v1 | binary classifier, all tilts | 76.5% |
| v2 | dropped high-tilt views (invisible bars = bad labels) | 86.8% |
| v3 | more snapshots | 86.4% |
| v3 + TTA + 3-seed ensemble | inference-time tricks | 91.8% |
| v4 | added velocity channels (kinematics) | 83.3% *(negative result)* |
| **REGRESSION (final)** | **predict continuous bar strength, not yes/no** | **96.0%** |

**Final model: ResNet50 regression — 96.0% accuracy, measures bar strength to
±0.02 (MAE 0.0217).** Verified no overfitting (train MAE = test MAE).

**Why regression won (+9.6 pts):** predicting the continuous A2/A0 value instead
of a bar/no-bar label (a) unlocked 60 "mid-strength" snapshots that binary
labeling had to throw away, (b) cured the classifier's blindness around its own
decision threshold, (c) gave richer supervision per image. Same test set, only
the training signal changed.

---

## Where everything is hosted

### 💻 Local — `E:\IIT KNP PROJ\windowpynbody2\`
- `results/regression_resnet50/` — **the final model** (`model.weights.h5`) + reports
- `results/` — every experiment's reports, figures, logs (the learnings)
- `code/` — all scripts; `predict.py` runs the final model, `eval_regression.py` reproduces 96%
- `data/` — datasets · `gadget_snapshots/` — raw simulation · `extras/bar_labels.csv` — labels

### 🐙 GitHub (public) — code + reports
**https://github.com/GuptaOum/IITK_astronomy_project**
All code and all reports/figures. README = the full story. (Big files git-ignored.)

### 🤗 Hugging Face (public) — the model weights
**https://huggingface.co/kjfk/galaxy-bar-detection-resnet50**
- `regression_resnet50.weights.h5` — the final model
- `bar_resnet50_v1/v2/v3.keras` — binary history
- Model card with load instructions + the pred-vs-true figure

### ☁️ AWS EC2 — training machines (only needed for NEW training)
| Name | ID | Type | State |
|---|---|---|---|
| iitk-gpu-regression | i-0e7df6c19e64f679e | g4dn.xlarge (T4 GPU) | *check console* |
| iitk-v4-render-train | i-0659350225e374908 | c7i.2xlarge (CPU) | *check console* |
- SSH key (both): `C:\Users\hp\.ssh\face-attendance.pem`, user `ubuntu`
- IPs change on every stop/start — look them up in the AWS console
- ⚠ **Stop instances when not training** — they bill by the hour.
  `aws ec2 stop-instances --instance-ids i-0e7df6c19e64f679e`

---

## How to use the model

```powershell
cd "E:\IIT KNP PROJ\windowpynbody2"
.\veny\Scripts\python.exe code\predict.py gadget_snapshots\snapshot_350
# -> BAR / NO BAR  (predicted bar strength A2/A0 = 0.214, threshold 0.19)
```

**Note:** the model is saved as weights-only (a Keras bug) — load it by
rebuilding the architecture then `load_weights()`; the recipe is in
`code/predict.py` and the Hugging Face model card.

*For full details, folder tree, warnings, and next steps, see `PROJECT_STATUS.md`.*
