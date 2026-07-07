"""
v4 dataset generator - morpho-kinematic 3-channel images from GADGET snapshots.

Improvements over windowpynbody2/code/generate_dataset.py (v3):
  1. THREE physical channels instead of replicated grayscale:
       R = log stellar density, residual-enhanced (unsharp mask) so the
           non-axisymmetric bar structure stands out from the smooth disk
       G = density-weighted mean line-of-sight velocity (bar streaming
           motions survive projection even when the bar shape is hidden)
       B = line-of-sight velocity dispersion (bars are dynamically hot)
  2. Rendered at 448px and block-averaged down to 224px - suppresses
     SPH particle speckle noise.
  3. Same snapshots, tilt cap (<=40 deg), and split seed as v3, so v4 vs v3
     is a controlled comparison: only the image content changes.

Run from this folder (snapshots stay in windowpynbody2):
    python generate_dataset_v4.py            # full render
    python generate_dataset_v4.py 0 4        # shard 0 of 4 (parallel)
Set PREVIEW_ONLY=True to render a small check grid first.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph
from scipy.ndimage import gaussian_filter
import os
import gc
import sys
import time as timer

# -------------------------------
# CONFIG
# -------------------------------
SNAP_DIR = r"E:\IIT KNP PROJ\windowpynbody2"
LABELS_CSV = os.path.join(SNAP_DIR, "extras", "bar_labels.csv")
OUTPUT_DIR = "dataset_v4"

IMG_SIZE = 224
SUPERSAMPLE = 2                 # render at 448, average down to 224
WIDTH = 40.0                    # kpc field of view

# Channel normalization ranges
RHO_VMIN, RHO_VMAX = 1e7, 1e9   # log density clip (validated in v2/v3)
UNSHARP_SIGMA = 15              # px (at 224 scale) smooth-disk removal scale
UNSHARP_CLIP = 0.25             # residual range mapped to [0,1]
VLOS_CLIP = 150.0               # km/s, mean LOS velocity mapped to [0,1]
SIGMA_CLIP = 160.0              # km/s, dispersion mapped to [0,1]

TILT_STEP = 10
MAX_TILT = 40
TILTS = range(0, MAX_TILT + 1, TILT_STEP)   # 5 x 5 = 25 views per snapshot

BAR_MIN = 0.22
NOBAR_MAX = 0.10
PER_CLASS = 61
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

PREVIEW_ONLY = False


# -------------------------------
# SNAPSHOT SELECTION / SPLIT (identical logic + seed to v3)
# -------------------------------
def pick_snapshots():
    df = pd.read_csv(LABELS_CSV)
    bar = df[df.peak_a2a0 >= BAR_MIN].snapshot.tolist()
    nobar = df[df.peak_a2a0 <= NOBAR_MAX].snapshot.tolist()

    def spaced(lst, n):
        if len(lst) <= n:
            return lst
        idx = np.linspace(0, len(lst) - 1, n).astype(int)
        return [lst[i] for i in idx]

    bar = spaced(bar, PER_CLASS)
    nobar = spaced(nobar, PER_CLASS)
    print(f"bar: {len(bar)} snapshots | no_bar: {len(nobar)} snapshots")
    return {"bar": bar, "no_bar": nobar}


def split_snapshots(snaps):
    out = {}
    for cls, lst in snaps.items():
        lst = list(lst)
        rng = np.random.default_rng(42)
        rng.shuffle(lst)
        n = len(lst)
        n_train = max(1, round(n * SPLIT["train"]))
        n_val = max(1, round(n * SPLIT["val"]))
        out[cls] = {"train": lst[:n_train],
                    "val": lst[n_train:n_train + n_val],
                    "test": lst[n_train + n_val:]}
    return out


# -------------------------------
# RENDERING
# -------------------------------
def load_aligned(snap_name):
    sim = pynbody.load(os.path.join(SNAP_DIR, snap_name))
    sim.physical_units()
    sim["pos"] *= 1.0 / 1000.0
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
    # derived array for the dispersion channel: sigma^2 = <vz^2> - <vz>^2
    sim.star["vz2"] = sim.star["vz"] ** 2
    return sim


def _img(sim, qty, av_z):
    return sph.image(sim.star, qty=qty, width=WIDTH,
                     resolution=IMG_SIZE * SUPERSAMPLE, log=False,
                     av_z=av_z, noplot=True, show_cbar=False,
                     approximate_fast=False)


def _downsample(a, f=SUPERSAMPLE):
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def render_channels(sim):
    """Return (224,224,3) float array in [0,1]: residual-density, vlos, sigma."""
    rho = _downsample(_img(sim, "rho", av_z=False))
    vlos = _downsample(_img(sim, "vz", av_z=True))
    vz2 = _downsample(_img(sim, "vz2", av_z=True))

    # R: log density -> [0,1], then unsharp residual re-normalized
    d = np.log10(np.clip(rho, RHO_VMIN, RHO_VMAX))
    d = (d - np.log10(RHO_VMIN)) / (np.log10(RHO_VMAX) - np.log10(RHO_VMIN))
    resid = d - gaussian_filter(d, UNSHARP_SIGMA)
    r = np.clip(resid, -UNSHARP_CLIP, UNSHARP_CLIP) / (2 * UNSHARP_CLIP) + 0.5
    # keep some absolute brightness so empty sky != disk: blend 50/50
    r = 0.5 * r + 0.5 * d

    # G: mean LOS velocity, symmetric around 0.5
    g = np.clip(vlos, -VLOS_CLIP, VLOS_CLIP) / (2 * VLOS_CLIP) + 0.5

    # B: LOS velocity dispersion
    sig = np.sqrt(np.clip(vz2 - vlos ** 2, 0, None))
    b = np.clip(sig, 0, SIGMA_CLIP) / SIGMA_CLIP

    # mask empty sky (no density) in kinematic channels to neutral values
    sky = rho <= RHO_VMIN * 0.5
    g[sky] = 0.5
    b[sky] = 0.0

    return np.dstack([r, g, b]).astype(np.float32)


def save_rgb(arr, path):
    plt.imsave(path, np.clip(arr, 0, 1), origin="lower")


def render_snapshot(snap_name, out_dir, progress=None):
    sim = load_aligned(snap_name)
    aligned = sim["pos"].copy()
    aligned_vel = sim["vel"].copy()
    n = 0
    n_views = len(TILTS) ** 2
    for rx in TILTS:
        for ry in TILTS:
            sim["pos"][:] = aligned
            sim["vel"][:] = aligned_vel     # velocities must rotate too!
            if rx:
                sim.rotate_x(rx)
            if ry:
                sim.rotate_y(ry)
            arr = render_channels(sim)
            save_rgb(arr, os.path.join(out_dir, f"{snap_name}_X{rx:02d}_Y{ry:02d}.png"))
            n += 1
            if progress:
                done, total, t0 = progress
                done += n
                el = timer.time() - t0
                print(f"  {snap_name} {n:2d}/{n_views} | overall {done}/{total} "
                      f"({100*done/total:.1f}%) | ETA {el/done*(total-done)/60:.0f} min",
                      flush=True)
    del sim
    gc.collect()
    return n


# -------------------------------
# PREVIEW
# -------------------------------
def preview(snaps):
    out = os.path.join(OUTPUT_DIR, "preview")
    os.makedirs(out, exist_ok=True)
    for cls in ("bar", "no_bar"):
        snap = snaps[cls][-1]
        sim = load_aligned(snap)
        for rx in (0, 30):
            sim.rotate_x(rx) if rx else None
            arr = render_channels(sim)
            save_rgb(arr, os.path.join(out, f"{cls}_{snap}_X{rx:02d}_rgb.png"))
            for i, name in enumerate(("R_resid_density", "G_vlos", "B_sigma")):
                plt.imsave(os.path.join(out, f"{cls}_{snap}_X{rx:02d}_{name}.png"),
                           arr[:, :, i], cmap="gray", vmin=0, vmax=1, origin="lower")
            if rx:
                sim.rotate_x(-rx)
        del sim
        gc.collect()
        print(f"preview: {cls} ({snap})")
    print(f"Inspect {out}/ - bar should show elongated core in R, butterfly/"
          f"quadrupole pattern in G, boxy hot center in B.")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    SHARD, NSHARDS = (int(sys.argv[1]), int(sys.argv[2])) \
        if len(sys.argv) == 3 else (0, 1)

    snaps = pick_snapshots()
    if PREVIEW_ONLY:
        preview(snaps)
        raise SystemExit

    splits = split_snapshots(snaps)
    if NSHARDS > 1:
        idx = 0
        for cls in splits:
            for split in splits[cls]:
                keep = []
                for s in splits[cls][split]:
                    if idx % NSHARDS == SHARD:
                        keep.append(s)
                    idx += 1
                splits[cls][split] = keep

    n_snaps = sum(len(l) for c in splits for l in splits[c].values())
    grand = n_snaps * len(TILTS) ** 2
    print(f"Rendering {n_snaps} snapshots x {len(TILTS)**2} views = {grand} images")
    t0 = timer.time()
    total = 0
    for cls in splits:
        for split, lst in splits[cls].items():
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for snap in lst:
                total += render_snapshot(snap, out_dir, (total, grand, t0))
    print(f"\nDone: {total} images in {OUTPUT_DIR}/ ({(timer.time()-t0)/60:.0f} min)")
