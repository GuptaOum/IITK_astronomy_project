# IITK Astronomy Plotting

Analysis and visualization pipeline for an N-body galaxy simulation, built on
[pynbody](https://pynbody.github.io/pynbody/). The project studies the
formation and evolution of a stellar bar in a simulated disc galaxy across
501 simulation snapshots, tracking bar strength, buckling instability,
pattern speed, and vertical/radial velocity dispersion over time.

## What this project does

- Loads a time series of N-body simulation snapshots and measures:
  - **Bar strength** (A2/A0 Fourier amplitude)
  - **Buckling strength** and **BPX strength** (out-of-plane bending signatures)
  - **Pattern speed** evolution of the bar
  - **Radial vs. vertical velocity dispersion** (σ_R vs. σ_z)
  - Mean vertical height of the disc over time
- Generates rendered snapshot images and rotated 3D views of the galaxy
  for visual inspection and animation.
- Includes a CNN-based image classification experiment for detecting bars
  in simulated galaxy images.

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
