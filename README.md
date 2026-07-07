# IITK Astronomy Plotting

Analysis and visualization pipeline for an N-body galaxy simulation, built on
[pynbody](https://pynbody.github.io/pynbody/). The project has two goals:

1. **Reproduce the diagnostic charts from published bar/buckling-instability
   papers** (see `pdf/papers/`) — bar strength, buckling strength, pattern
   speed, and velocity dispersion — computed directly from the simulation's
   501 snapshots, to check the simulation against known results.
2. **Generate augmented training data for a CNN bar-detection model.** While
   rendering the simulation at many viewing angles for the plots above, the
   rotated/rendered snapshot images doubled as an image dataset (rotated
   views, multiple viewing angles) intended to train a CNN to classify/detect
   bars in galaxy images. This half of the project is a **work in progress**
   feeding into a separate, follow-on project — the CNN training scripts here
   (`imagegeneration_cnn.py`, `transfer_learning.py`, `predict_bar.py`) are
   early/exploratory, not a finished model.

## What this project does

- Loads a time series of N-body simulation snapshots and measures, following
  the methodology of the reference papers:
  - **Bar strength** (A2/A0 Fourier amplitude)
  - **Buckling strength** and **BPX strength** (out-of-plane bending signatures)
  - **Pattern speed** evolution of the bar
  - **Radial vs. vertical velocity dispersion** (σ_R vs. σ_z), including a
    second "manual" implementation cross-checked against the paper's exact
    method
  - Mean vertical height of the disc over time
- Renders snapshot images and rotated 3D views of the galaxy at many angles —
  both for visual inspection/animation and as raw material for an augmented
  image dataset.
- Includes early CNN image-classification experiments for detecting bars in
  the rendered images (exploratory, feeding a separate future project).

## Repository structure

```
windowpynbody2/
├── code/                  # All scripts: physics analysis + CNN pipeline (see tables below)
├── results/               # Every ML experiment's outputs, chronological
│   ├── v1_binary/            # run 1: tilts<=60, 76.5% (reports, tilt figure)
│   ├── v2_binary/            # run 2: tilts<=40, 86.8%
│   ├── v3_binary/            # run 3: 122 snaps, 86.4% (+TTA/ensemble reports)
│   ├── v3_binary_seed1/, v3_binary_seed2/   # ensemble members
│   ├── v4_kinematic/         # velocity-channel experiment (negative result)
│   ├── regression_resnet50/  # THE FINAL MODEL: 96.0% acc, MAE 0.027
│   ├── regression_convnext/, regression_effnetv2b0/
│   └── regression_ensemble/  # final report + pred_vs_true scatter figure
├── data/                  # rendered datasets (gitignored, regenerable)
│   ├── dataset_v3_binary/     # 3,050 imgs, train/val/test by snapshot
│   └── dataset_v1_tilt60/     # historical v1 dataset
├── gadget_snapshots/      # raw GADGET binaries snapshot_000..501 (gitignored, ~48GB)
├── images/                # astronomy visualizations (snapshots, rotated views, plots)
├── pdf/                   # reference papers + analysis PDF exports
├── extras/                # bar_labels.csv (the CNN label source!) + misc outputs
└── veny/                  # Python virtualenv (gitignored)
```

## Notes on excluded data

Raw GADGET snapshots (`gadget_snapshots/`, ~48GB), rendered datasets (`data/`),
model weights (`*.keras`, `*.weights.h5`), and the virtualenv are gitignored —
large and regenerable. Weights are published on Hugging Face:
https://huggingface.co/kjfk/galaxy-bar-detection-resnet50

NOTE: scripts meant to run on an EC2 box (`generate_dataset*.py`,
`train_*.py`, `ensemble_regression_eval.py`) use the flat home-directory
layout (snapshots + `dataset/` in cwd). Locally-run analysis scripts use the
`results/` + `data/` layout above. Legacy physics scripts (buck.py etc.)
expect the snapshots in their cwd - run them from `gadget_snapshots/`.

## Key scripts (`code/`)

| Script | Purpose |
|---|---|
| `buck.py`, `bpx.py` | Compute buckling and BPX strength over all snapshots |
| `patternspeed.py` | Track the bar's pattern speed over time |
| `sigmaZvssigmaR.py`, `sigmazvssigmar2(manual).py` | Radial vs. vertical velocity dispersion |
| `meanverticalheight.py` | Mean vertical disc height over time |
| `generate_rotated_views.py` | Render rotated 3D views of a snapshot |
| `imagegeneration(1-501).py` | Batch-render snapshot images for all 501 snapshots |
| `imagegeneration_cnn.py`, `transfer_learning.py` | CNN-based bar detection on rendered images |
| `make_pdf_grid.py`, `make_slideshow.py` | Compile snapshot grids into PDF/video summaries |
| `predict_bar.py` | Run bar-detection prediction on new images |

## CNN bar detection (transfer learning experiments)

The follow-on project: train a ResNet50 (TensorFlow, ImageNet transfer
learning) to classify rendered galaxy images as barred / unbarred. Labels come
from the physical A2/A0 bar-strength measurement (`bar_labels.csv`), so no
manual labeling is needed. Datasets are generated by re-rendering snapshots at
many viewing angles (physically genuine augmentation), always split into
train/val/test **by snapshot** to prevent leakage.

Pipeline scripts (`code/`): `generate_dataset.py` (labeled dataset rendering,
supports multi-process sharding), `train_resnet.py` (two-phase transfer
learning), `analyze_results.py` (accuracy-vs-inclination + decision-threshold
tuning), `compare_models.py` (same-test-set model comparison),
`aws_train.ps1` (EC2 training automation).

### Results across all experiments

| Experiment | Approach | Test accuracy |
|---|---|---|
| v1 | binary, 40 snapshots, tilts 0-60 deg | 76.5% |
| v2 | binary, tilts capped 40 deg (label noise pruned) | 86.8% |
| v3 | binary, 122 snapshots, 3,050 imgs | 86.4% (AUC 0.944) |
| v3 + TTA | avg over 8 rotations/flips at inference | 90.2% |
| v3 3-seed ensemble + TTA | | 91.8% (AUC 0.974) |
| v4 morpho-kinematic | velocity+dispersion input channels | 83.3% (negative result) |
| **REGRESSION ResNet50 (final)** | **predict continuous A2/A0, threshold 0.19** | **96.0%, MAE 0.027** |

### The regression breakthrough (86.4% -> 96.0%)

Instead of binary bar/no-bar labels, the final model predicts the *continuous*
A2/A0 bar strength. This (a) unlocked 60 mid-strength snapshots
(0.10 < A2/A0 < 0.22) that binary labeling had to discard, (b) cured the
"boundary blindness" of a classifier that never saw examples near its own
0.19 threshold, and (c) gave richer supervision per image. Same test set,
same scoring rule - only the training signal changed. The model doubles as a
bar-strength *measuring instrument*: MAE 0.027 in A2/A0 units (see
`results/regression_ensemble/pred_vs_true.png`). Architecture comparison:
ResNet50 (96.0%) decisively beat ConvNeXt-Tiny (91.6%) and EfficientNetV2-B0
(84.0%); ensembling diluted the champion, so the final model is the single
regression ResNet50.

Key findings (see `models_run1/`, `models_run2/`, `models_v3/` for reports,
per-epoch logs, and figures):

- **Accuracy is limited by physics, not the model.** v1's accuracy-vs-tilt
  analysis showed near-perfect classification face-on, degrading steadily with
  inclination - a tilted bar is progressively hidden by projection. Views
  beyond 40 deg are effectively label noise and were pruned in v2/v3.
- **Same-test-set comparison** showed v1's and v2's models are statistically
  identical on answerable (<=40 deg) views (~86.5%): the headline improvement
  came from removing ill-posed test cases, and deep networks proved robust to
  the label noise in v1's training data.
- Final model (v3): 87.1% accuracy, 89% bar recall at tuned threshold, on a
  450-image test set from 18 never-seen snapshots.

Trained model weights (~215MB) and rendered datasets are not in the repo -
both are reproducible from the scripts (datasets require the raw GADGET
snapshots).

## Project context

Developed as part of an astrophysics research project at IIT Kanpur (IITK),
studying bar formation and buckling instability in simulated disc galaxies.
