---
name: iitk-bar-cnn-unified
description: "IIT Kanpur galaxy stellar-bar detection CNN project — v1-v4 + regression arc, pynbody API reference, script inventory, AWS infra, published models"
metadata:
  node_type: memory
  type: project
---

IIT Kanpur astrophysics research internship project (Dr. S.K. Kataria; user is a beginner student, professor-directed — explain transfer-learning concepts simply with analogies). Task: CNNs that detect stellar bars in images rendered from a GADGET N-body simulation (collisionless: stars + dark matter, NO gas/temperature).

502 snapshots at `gadget_snapshots/snapshot_XXX`, 96MB each (exactly **96,000,312 bytes** — useful for upload integrity checks), ~2M star particles each with position + velocity. **Never read snapshot files directly** (huge) — always go through pynbody.

---

## Method / conventions (stable across all runs)

- Labels: `extras/bar_labels.csv` (peak A2/A0 per snapshot; bar threshold 0.19).
  - Binary runs used clean margins: bar >= 0.22 (only 61 exist), no_bar <= 0.10 (313 exist), mid 0.10-0.22 = 128 snapshots (excluded from binary, later EXPLOITED by the regression approach).
- Split ALWAYS by snapshot, never by image (image-level split would leak). Seed 42 everywhere.
- Views: X/Y tilts 0-40 deg, 10-deg steps = 25 images/snapshot. >40 deg dropped (bar hidden by projection = label noise, proven by v1's tilt analysis).
- Render recipe (must match EXACTLY at inference — training/serving skew otherwise): grayscale stellar density, `sph.image` qty="rho" width=40kpc res=224, log10 clip 1e7-1e9, normalize 0-1, grayscale replicated to 3 channels.
- Two-phase transfer learning everywhere: phase 1 — freeze ImageNet backbone + train head (8 epochs, lr 1e-3); phase 2 — unfreeze + fine-tune (up to 18 epochs, lr 1e-5, early stopping).

---

## Results ladder (test = 450 imgs / 18 unseen snapshots for v3-comparable numbers)

1. **v1 binary** (tilt<=60, 1960 imgs): 76.5%, AUC 0.836. Errors = high-tilt barred views.
2. **v2 binary** (tilt<=40, 1500 imgs): 86.8%. **Key finding:** v1-vs-v2 on the SAME test set are statistically identical (~86.5%) — the apparent jump was from removing ill-posed test cases, not a better model. Deep nets are robust to v1's label noise.
3. **v3 binary** (122 snaps, 3050 imgs): 86.4%, AUC 0.944 (87.1% with threshold 0.66).
4. **v3 + TTA** (average of 8 rotations/flips): 90.2%. Ensemble of 3 seeds + TTA: 91.8%, AUC 0.974.
5. **v4 morpho-kinematic** (R=residual density, G=<vz>, B=sigma_z; `bar_cnn_v4/` folder): 83.3% — **negative result**. Kinematics did not help at <=40deg tilt, worst at 40deg. "Morphology alone suffices at answerable inclinations."
6. **REGRESSION (2026-07-08, the breakthrough)**: predict continuous A2/A0 instead of binary classification. Unlocks the mid-strength snapshots as usable training data. Dataset = v3's 3050 imgs + 1500 new (60 mid snaps spaced 0.101-0.218, rendered as `dataset_mid/`, flat layout). train=3200/val=675/test=675; classification scored via `pred>=0.19` on the v3-test subset (450).
   - **ResNet50: MAE 0.0267, classification 96.0% = THE FINAL MODEL** (weights local: `results/regression_resnet50/model.weights.h5`; rebuild architecture + `load_weights` to use it — full `model.save()` hits a Keras "Cannot serialize Ellipsis" bug from the embedded `preprocess_input`).
   - ConvNeXt-Tiny: MAE 0.0365, 91.6%. EfficientNetV2B0: MAE 0.0429, 84.0% (weak — aggressive early downsampling loses central bar detail).
   - Ensembles/TTA all DILUTED the champion (ens3+TTA 90.2%, ens2+TTA 93.6%; rn50+TTA better MAE 0.0238 but worse 94.0% classification). **Lesson repeated 3× this project: a committee loses to a decisively-best single model.** A hybrid binary+regression ensemble was abandoned mid-eval per the user ("keep it simple, use the 96% ResNet50").
   - **Why regression won (+9.6 over binary):** (a) 60 more real snapshots usable as training data, (b) cured "boundary blindness" — binary classification never saw the 0.10-0.22 mid region where the 0.19 decision threshold actually lives, (c) a continuous target is richer supervision than a binary label. Scatter figure (report centerpiece): `results/regression_ensemble/pred_vs_true.png`.

---

## Pipeline code (all in `code/`, pushed to GitHub except `train_regression.py`/`generate_dataset_mid.py` — those still need pushing)

- `generate_dataset.py` — binary v3 renderer, SHARD/NSHARDS multiprocess.
- `generate_dataset_mid.py` — mid-strength flat renderer, reads `mid_snapshots.txt`.
- `bar_cnn_v4/generate_dataset_v4.py` — kinematic (R/G/B) renderer.
- `train_resnet.py` — binary, CLI seed arg → `models_seed<N>/`.
- `train_regression.py` — CLI arch: `resnet50|effnetv2b0|convnext_tiny` → `models_reg_<arch>/`; Huber loss, sigmoid output, target = a2a0/0.30; saves `model.weights.h5` only (see the Keras serialization bug above).
- `analyze_results.py` — tilt curve + threshold tuning.
- `compare_models.py` — same-test-set comparison.
- `tta_eval.py`, `ensemble_tta_eval.py` — test-time augmentation and ensembling.
- `predict.py` — end-to-end inference with 3-model+TTA, grayscale-input guard.
- `aws_train.ps1` — AWS training launcher.

---

## pynbody API reference (patterns actually used)

### Loading & units
- `pynbody.load("snapshot_NNN")` — loads Gadget snapshots.
- `s.physical_units()` — converts to physical units.
- `s.properties['time'].in_units('Gyr')` — access simulation time.

### Particle families
- `s.star`, `s.dm` — star / dark-matter SubSnaps.
- `s.families()` — list available families.
- `s.all_keys` — list available arrays.

### Particle arrays
- `s['pos']`, `s['mass']`, `s['vel']` — position, mass, velocity.
- `s['x']`, `s['y']`, `s['z']` — position component shorthand.
- `s['vx']`, `s['vy']`, `s['vz']` — velocity component shorthand.

### Analysis modules
- `pynbody.analysis.halo.center(s, mode='ssc')` — shrinking-sphere centering.
- `pynbody.analysis.angmom.faceon(s.star, disk_size=N, move_all=True, already_centered=True)` — rotate to face-on.
- `pynbody.analysis.angmom.sideon(s)` — rotate to side-on (edge-on).
- `pynbody.analysis.angmom.ang_mom_vec(s.star)` — compute angular momentum vector.
- `pynbody.analysis.profile.Profile(s.star, min=0, max=10, nbins=30)` — radial profile; `p['vr_disp']`, `p['vz_disp']`, `p['mass']` per bin.

### Visualization
- `pynbody.plot.sph.image(s.star, qty="rho", width=56.0, cmap=mycmap, log=True, resolution=1000, approximate_fast=False)` — SPH-rendered image.

### Physics analyses done (and the scripts that do them)
| Script | Purpose |
|--------|---------|
| `imagegeneration(1-501).py` | Generates face-on SPH stellar density images for snapshots 501+ |
| `bpx.py` | BPX (boxy/peanut) strength over time — RMS vertical height in 2-8 kpc annulus |
| `buck.py` | Buckling amplitude time series with peak detection |
| `check.py` | Radial & vertical velocity dispersion (σ_r, σ_z) vs time via `profile.Profile` |
| `patternspeed.py` | Bar pattern speed (Ω_p) via Fourier phase tracking & unwrapping |
| `exp.py` | Disk vs halo angular momentum magnitude over time |
| `momentumExchange.py` | ΔLz (change in z-component of angular momentum) for disk & halo |
| `meanverticalheight.py` | Mean |z| of stars within 10 kpc over time |
| `sigmaZvssigmaR.py` | σ_z/σ_r ratio evolution (without centering/alignment) |
| `tick.py` | Matplotlib tick formatting test (no pynbody) |
| `exp2.py`, `expcheckradius.py`, `radialdispersioon(exp).py`, `ramcrash.py`, `sigmazvssigmar2(manual).py` | Additional analysis variants |

---

## AWS infra (root CLI works, region ap-south-1, key `~/.ssh/face-attendance.pem`, user ubuntu, SG `sg-0dca629c63e847a32`; IPs CHANGE on stop/start — always re-check)

- **CPU box** `i-0659350225e374908` (c7i.2xlarge): holds ALL 182 snapshots (122 clean + 60 mid), `dataset/` (v3), `dataset_mid/`, `dataset_v4/`, all scripts. **STOP (never terminate)** between uses — parking the 12GB+ snapshots avoids multi-hour re-uploads.
- **GPU box** `i-0e7df6c19e64f679e` (g4dn.xlarge, Tesla T4 15GB, Deep Learning AMI TF 2.18): **must `source /opt/tensorflow/bin/activate`** before running python (bare `python3` lacks numpy/TF). Has `regdata.tgz` (dataset+dataset_mid), `train_regression.py`, `extras/`. GPU trains ~10x faster than the CPU box (epoch ~10-35s vs ~5min).
- GPU quota `L-DB2E81BA` APPROVED (=4) 2026-07-07 after appeal; root cause of all earlier AWS blocks was a missing payment method — user added UPI AutoPay, which also unblocked Bedrock (`ap-south-1` AUTHORIZED, 207M tokens/day; `us-east-1` still NOT_AUTHORIZED).

## Hard-won ops lessons
- User's home upload ~2.8MB/s clean but flaky: connections reset mid-file, and concurrent streaming (bufferbloat) crushes upload ~10x. For big uploads: per-file scp loop with size-verify (expect exactly 96000312 bytes) + retry + timeout, or hand the user manual scp commands.
- Background/task-stop mechanisms do NOT kill child ssh/tar/scp processes on Windows — verify with `Get-Process ssh,scp,tar` and `Stop-Process`, else ghost streams overwrite each other.
- `scp` exit code lies on flaky links — ALWAYS verify remote file size after.
- Windows `md5sum` prints `*file` vs Linux ` file` — diff on the hash column only.
- Long-running background jobs get killed after ~10 min in some environments — run long jobs via `nohup` ON the EC2 box, with marker files (`touch DONE`) + a local until-loop watcher. Delete stale marker files before relaunching a chain (a stale `ALL_TRAINING_DONE` caused a false "already complete" read once).
- Small dataset tarballs (16-60MB, e.g. `v3ds.tar.gz`/`regdata.tgz`) move fine via a local PC relay (CPU box → PC → GPU box) — trivial compared to the raw snapshot uploads.

## Layout (reorganized 2026-07-08)
`gadget_snapshots/` (502 raw), `data/{dataset_v3_binary,dataset_v1_tilt60}`, `results/{v1_binary,v2_binary,v3_binary,v3_binary_seed1|2,v4_kinematic,regression_resnet50|convnext|effnetv2b0,regression_ensemble}`, `code/`, `images/`, `pdf/`, `extras/`, `venv/`. Locally-run scripts use these new paths; EC2-run scripts keep a flat home layout (README documents both). The old `bar_cnn_v4/` folder was deleted (evidence salvaged into `results/v4_kinematic` + `code/`).

## Published
- **GitHub**: https://github.com/GuptaOum/IITK_astronomy_project (title "IITK Astronomy Plotting"). README documents v1-v3 + TTA/ensemble.
- **HuggingFace**: https://huggingface.co/kjfk/galaxy-bar-detection-resnet50 (`kjfk` = user's own HF account, same person as GitHub GuptaOum) — regression weights + report + scatter plot + 96.0%-headline model card uploaded 2026-07-08, includes the rebuild+`load_weights` recipe. Local weights pruned to keep ONLY the final model (~1GB freed).
- These accuracy numbers (96.0% classification, 0.027 MAE) are cited in the user's resume.

## Status: project complete through the regression phase
Both EC2 instances STOPPED (CPU box parked with all 182 snapshots; GPU box stopped). All final models downloaded locally under `results/`.

## TODO next session (optional polish, not required)
1. Scratch-CNN baseline (the professor's transfer-learning comparison question — GPU is cheap now).
2. >40deg edge-on kinematics test; ask the professor for more simulation runs to find the true ceiling.
3. **User should rotate the HF write token** (`HF_TOKEN_WRITE` in `.env`, gitignored) — it was pasted through chat at one point.
