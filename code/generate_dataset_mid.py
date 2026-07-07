"""
Render the 60 mid-strength snapshots (0.10 < A2/A0 < 0.22) for REGRESSION
training. These were excluded from the binary datasets because they fit
neither class; for regression their continuous A2/A0 value IS the label.

Same grayscale recipe as generate_dataset.py (v3): 224px, log density
1e7-1e9, tilts 0-40 deg in 10 deg steps (25 views/snapshot). Output is FLAT
(no class folders) - labels come from extras/bar_labels.csv at train time:

    dataset_mid/train/*.png   dataset_mid/val/*.png   dataset_mid/test/*.png

Split by snapshot, seed 42, 70/15/15 - same convention as always.
Run on the EC2 box (snapshots in home dir):
    python3 generate_dataset_mid.py            # or with shards: 0 4
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph
import os, gc, sys
import time as timer

SNAP_DIR = os.path.expanduser("~")
MID_LIST = "mid_snapshots.txt"
OUTPUT_DIR = "dataset_mid"
IMG_SIZE = 224
WIDTH = 40.0
VMIN, VMAX = 1e7, 1e9
TILTS = range(0, 41, 10)

def render_snapshot(snap, out_dir, progress=None):
    sim = pynbody.load(os.path.join(SNAP_DIR, snap))
    sim.physical_units()
    sim["pos"] *= 1.0 / 1000.0
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
    aligned = sim["pos"].copy()
    n = 0
    for rx in TILTS:
        for ry in TILTS:
            sim["pos"][:] = aligned
            if rx: sim.rotate_x(rx)
            if ry: sim.rotate_y(ry)
            im = sph.image(sim.star, qty="rho", width=WIDTH, resolution=IMG_SIZE,
                           log=False, noplot=True, show_cbar=False,
                           approximate_fast=False)
            im = np.log10(np.clip(im, VMIN, VMAX))
            im = (im - np.log10(VMIN)) / (np.log10(VMAX) - np.log10(VMIN))
            plt.imsave(os.path.join(out_dir, f"{snap}_X{rx:02d}_Y{ry:02d}.png"),
                       im, cmap="gray", vmin=0, vmax=1, origin="lower")
            n += 1
            if progress:
                done, total, t0 = progress
                done += n
                el = timer.time() - t0
                print(f"  {snap} {n:2d}/25 | {done}/{total} | "
                      f"ETA {el/done*(total-done)/60:.0f} min", flush=True)
    del sim
    gc.collect()
    return n

if __name__ == "__main__":
    SHARD, NSHARDS = (int(sys.argv[1]), int(sys.argv[2])) \
        if len(sys.argv) == 3 else (0, 1)
    snaps = [l.strip() for l in open(MID_LIST) if l.strip()]
    rng = np.random.default_rng(42)
    rng.shuffle(snaps)
    n = len(snaps)
    splits = {"train": snaps[:round(n*0.70)],
              "val": snaps[round(n*0.70):round(n*0.85)],
              "test": snaps[round(n*0.85):]}
    total = sum(len(v) for v in splits.values() if True)
    mine = 0
    idx = 0
    todo = []
    for split, lst in splits.items():
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
        for s in lst:
            if idx % NSHARDS == SHARD:
                todo.append((split, s))
            idx += 1
    grand = len(todo) * 25
    print(f"shard {SHARD}/{NSHARDS}: {len(todo)} snapshots = {grand} images")
    t0 = timer.time()
    done = 0
    for split, s in todo:
        done += render_snapshot(s, os.path.join(OUTPUT_DIR, split),
                                (done, grand, t0))
    print(f"done: {done} images ({(timer.time()-t0)/60:.0f} min)")
