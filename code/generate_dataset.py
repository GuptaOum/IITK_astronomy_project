"""
Generate a labeled CNN training dataset from simulation snapshots.

Fixes over generate_rotated_views.py:
  1. Tuned density range so the bar is actually visible (BAR-tuned vmin/vmax)
  2. Grayscale single-channel output (no colormap artifacts)
  3. Exact fixed pixel size (renders the raw SPH array, no matplotlib bbox)
  4. Renders only distinct inclinations (X/Y tilts, capped at MAX_TILT);
     in-plane Z rotation is left to cheap train-time augmentation
  5. Snapshots chosen automatically from bar_labels.csv with a clean margin
     around the threshold, and split into train/val/test BY SNAPSHOT to
     avoid data leakage

Output layout (ready for tf.keras.utils.image_dataset_from_directory):
    dataset/
      train/bar/*.png   train/no_bar/*.png
      val/bar/*.png     val/no_bar/*.png
      test/bar/*.png    test/no_bar/*.png

Run from the folder containing the snapshot_XXX files:
    python code/generate_dataset.py
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
import os
import gc
import time as timer

# -------------------------------
# CONFIG
# -------------------------------
LABELS_CSV = "extras/bar_labels.csv"
OUTPUT_DIR = "dataset"

IMG_SIZE = 224          # ResNet50 native input; exact pixel size guaranteed
WIDTH = 40.0            # kpc field of view

# Density range (log10 rho). The old 1e6..3e9 saturated the core and hid the
# bar. Start narrower and inspect a preview grid before a full run.
VMIN = 1e7
VMAX = 1e9

# Inclination grid: tilt about X and Y only. Z (in-plane spin) is skipped -
# do it at train time with tf.keras RandomRotation, it is a pure 2D rotation.
# Capped at 60 deg because near-edge-on views hide the bar (label noise).
TILT_STEP = 10
MAX_TILT = 40   # run1 analysis: labels get noisy past 40 deg (bar invisible)
TILTS = range(0, MAX_TILT + 1, TILT_STEP)   # 5 x 5 = 25 views per snapshot

# Class selection: keep a clean margin around the A2/A0 = 0.19 threshold so
# ambiguous snapshots never enter the dataset. (Max A2/A0 in this sim: 0.28)
BAR_MIN = 0.22          # peak_a2a0 >= this  -> class "bar"
NOBAR_MAX = 0.10        # peak_a2a0 <= this  -> class "no_bar"
PER_CLASS = 61          # snapshots per class (61 = every clean barred snapshot)

# Fractions of *snapshots* (not images) per split
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

PREVIEW_ONLY = False    # True: render 1 snapshot/class face-on at several
                        # vmin/vmax choices into dataset/preview/ and exit


# -------------------------------
# SNAPSHOT SELECTION
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
    print(f"Selected {len(bar)} bar snapshots: {bar[0]} .. {bar[-1]}")
    print(f"Selected {len(nobar)} no_bar snapshots: {nobar[0]} .. {nobar[-1]}")
    return {"bar": bar, "no_bar": nobar}


def split_snapshots(snaps):
    """Assign whole snapshots to train/val/test. Never split one snapshot's
    images across sets - that leaks near-identical images into the test set."""
    out = {}
    for cls, lst in snaps.items():
        lst = list(lst)
        rng = np.random.default_rng(42)
        rng.shuffle(lst)
        n = len(lst)
        n_train = max(1, round(n * SPLIT["train"]))
        n_val = max(1, round(n * SPLIT["val"]))
        out[cls] = {
            "train": lst[:n_train],
            "val": lst[n_train:n_train + n_val],
            "test": lst[n_train + n_val:],
        }
        print(f"{cls}: train={len(out[cls]['train'])} "
              f"val={len(out[cls]['val'])} test={len(out[cls]['test'])}")
    return out


# -------------------------------
# RENDERING
# -------------------------------
def load_aligned(snap_file):
    sim = pynbody.load(snap_file)
    sim.physical_units()
    sim["pos"] *= 1.0 / 1000.0
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
    return sim


def render_array(sim, vmin=VMIN, vmax=VMAX):
    """Render star density to a normalized 2D float array, no figure at all.
    noplot=True makes sph.image return the raw grid -> exact IMG_SIZE pixels."""
    im = sph.image(
        sim.star, qty="rho", width=WIDTH, resolution=IMG_SIZE,
        log=False, noplot=True, show_cbar=False, approximate_fast=False,
    )
    im = np.log10(np.clip(im, vmin, vmax))
    im = (im - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    return im  # 0..1, shape (IMG_SIZE, IMG_SIZE)


def save_gray(arr, path):
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")


def render_snapshot(snap_file, out_dir, progress=None):
    """progress: optional (done_so_far, grand_total, t_start) for a live ETA."""
    sim = load_aligned(snap_file)
    aligned = sim["pos"].copy()
    n = 0
    n_views = len(TILTS) ** 2
    for rx in TILTS:
        for ry in TILTS:
            sim["pos"][:] = aligned
            if rx:
                sim.rotate_x(rx)
            if ry:
                sim.rotate_y(ry)
            arr = render_array(sim)
            fname = f"{snap_file}_X{rx:02d}_Y{ry:02d}.png"
            save_gray(arr, os.path.join(out_dir, fname))
            n += 1
            if progress:
                done, grand_total, t_start = progress
                done += n
                elapsed = timer.time() - t_start
                eta = elapsed / done * (grand_total - done)
                print(f"  {snap_file} view {n:2d}/{n_views} "
                      f"| overall {done}/{grand_total} "
                      f"({100*done/grand_total:.1f}%) "
                      f"| ETA {eta/60:.0f} min", flush=True)
    del sim
    gc.collect()
    return n


# -------------------------------
# PREVIEW MODE - tune VMIN/VMAX here first
# -------------------------------
def preview(snaps):
    out = os.path.join(OUTPUT_DIR, "preview")
    os.makedirs(out, exist_ok=True)
    ranges = [(1e6, 3e9), (1e7, 1e9), (3e7, 1e9), (1e7, 3e8), (1e8, 1e9)]
    for cls in ("bar", "no_bar"):
        snap = snaps[cls][-1]
        sim = load_aligned(snap)
        for vmin, vmax in ranges:
            arr = render_array(sim, vmin, vmax)
            save_gray(arr, os.path.join(
                out, f"{cls}_{snap}_vmin{vmin:.0e}_vmax{vmax:.0e}.png"))
        del sim
        gc.collect()
        print(f"preview rendered for {cls} ({snap})")
    print(f"Inspect {out}/ and set VMIN/VMAX so the bar is clearly visible "
          f"in 'bar' images and absent in 'no_bar' images.")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    import sys

    # Optional sharding for parallel rendering: python generate_dataset.py 0 4
    # renders only snapshots where index % 4 == 0. Run 4 processes to use all
    # cores; they write to the same folders without conflict (distinct files).
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
        print(f"Shard {SHARD}/{NSHARDS}: rendering "
              f"{sum(len(l) for c in splits for l in splits[c].values())} snapshots")
    n_snaps = sum(len(lst) for cls in splits for lst in splits[cls].values())
    grand_total = n_snaps * len(TILTS) ** 2
    print(f"Rendering {n_snaps} snapshots x {len(TILTS)**2} views "
          f"= {grand_total} images\n")
    t0 = timer.time()
    total = 0
    for cls in splits:
        for split, lst in splits[cls].items():
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for snap in lst:
                t = timer.time()
                n = render_snapshot(snap, out_dir,
                                    progress=(total, grand_total, t0))
                total += n
                print(f"[{split}/{cls}] {snap}: {n} views "
                      f"in {timer.time()-t:.0f}s (total {total} images, "
                      f"{timer.time()-t0:.0f}s elapsed)")
    print(f"\nDone: {total} images in {OUTPUT_DIR}/ "
          f"({timer.time()-t0:.0f}s)")
