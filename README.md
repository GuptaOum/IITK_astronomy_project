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
├── code/                # All analysis and plotting scripts (Python)
├── images/
│   ├── snapshots/       # Rendered face-on snapshots of the simulation (501+ frames)
│   ├── rotated_views_501/ # Rotated 3D viewing angles of the galaxy
│   ├── buckling_plots/  # Plots of buckling/bar-strength/dispersion results
│   ├── cmap_samples/    # Reference charts of matplotlib colormaps used in plotting
│   └── download.jpg     # Project reference image
├── pdf/
│   ├── papers/          # Reference research papers on bar/buckling instabilities
│   └── buckling_plots/  # PDF exports of buckling analysis (time-series grids, snapshot grids)
└── extras/               # Supplementary outputs (CSV data, HTML plot, slideshow video)
```

## Notes on excluded data

Raw simulation snapshot binaries (`snapshot_000` … `snapshot_501`, tens of GB
of GADGET-format N-body data) and the local Python virtual environment
(`veny/`) are excluded from this repository — they are large, regenerable/
machine-specific, and not needed to review the code or results.

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

## Project context

Developed as part of an astrophysics research project at IIT Kanpur (IITK),
studying bar formation and buckling instability in simulated disc galaxies.
