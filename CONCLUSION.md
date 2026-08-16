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

## How the model was trained — start to finish

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1  RAW SIMULATION                                               │
│ 502 GADGET snapshots. Each = ~2 million star particles in 3D         │
│ (positions + velocities). NOT images yet — just numbers.             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2  PHYSICAL LABELS  (done earlier, extras/bar_labels.csv)       │
│ For each snapshot, compute A2/A0 (Fourier bar-strength).             │
│ This number is the "answer" — no manual labeling needed.            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3  RENDER IMAGES        [CPU box, pynbody, 4 parallel workers]  │
│ Pick clean snapshots. For each, view the galaxy face-on and at 25    │
│ tilt angles (0-40°). Turn each view into a 224×224 grayscale         │
│ density image. Result: thousands of labeled images.                 │
│ (generate_dataset.py / generate_dataset_mid.py)                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4  SPLIT  BY SNAPSHOT  (never by image — prevents leakage)     │
│ 70% train  /  15% validation  /  15% test.                          │
│ All 25 views of one galaxy stay in the same split.                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5  TRAIN  ResNet50      [GPU box, TensorFlow]                   │
│ Transfer learning, two phases:                                      │
│   Phase 1  freeze ImageNet backbone, train new head   (8 epochs)    │
│   Phase 2  unfreeze, fine-tune everything, tiny steps (≤18 epochs)  │
│ Target = the A2/A0 number (regression).  (train_regression.py)      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6  EVALUATE  on the held-out TEST snapshots (never seen)       │
│ 96.0% classification accuracy, MAE 0.022.                           │
│ Checked per-tilt + train-vs-test (no overfitting). (eval_regression)│
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 7  USE IT                                                       │
│ predict.py: give it any snapshot → predicted bar strength + verdict.│
│ Model published on Hugging Face; code + reports on GitHub.          │
└─────────────────────────────────────────────────────────────────────┘
```

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

### ☁️ AWS EC2 — two machines, two different jobs

**Why two?** Rendering images and training the CNN need different hardware, so
we split the work:

| Box | ID | Hardware | What it actually does |
|---|---|---|---|
| **CPU box** `iitk-v4-render-train` | i-0659350225e374908 | c7i.2xlarge (8 CPU cores, no GPU) | **RENDERS images** from the raw GADGET snapshots using pynbody (this is CPU-only work — a GPU can't help here). Holds all 182 snapshots + the datasets. This is where `generate_dataset*.py` runs, 4 workers in parallel. |
| **GPU box** `iitk-gpu-regression` | i-0e7df6c19e64f679e | g4dn.xlarge (Tesla T4 GPU) | **TRAINS the CNN** on the rendered images (`train_regression.py`). ~10× faster than CPU for training. Has TensorFlow + CUDA preinstalled (Deep Learning AMI). |

Typical flow: render on the CPU box → copy the small dataset to the GPU box →
train on the GPU → download the model. Neither is needed once results are
downloaded (they already are).

- SSH key (both): `C:\Users\hp\.ssh\face-attendance.pem`, user `ubuntu`.
  On the GPU box, run `source /opt/tensorflow/bin/activate` before `python`.
- IPs change on every stop/start — look them up in the AWS console.
- ⚠ **Stop instances when not training** — they bill by the hour.
  `aws ec2 stop-instances --instance-ids i-0e7df6c19e64f679e i-0659350225e374908`

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
