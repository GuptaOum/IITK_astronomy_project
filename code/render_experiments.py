"""
Render the two "heavy" experiment image sets on the EC2 CPU box.
Needs the raw GADGET snapshots, so it can only run where they live.

  EXPERIMENT A — evolution: every available snapshot, face-on (1 view each).
      -> lets us plot CNN-predicted bar strength across simulation time
         against the true physical A2/A0 curve.

  EXPERIMENT B — tilt sweep: the 27 held-out TEST snapshots rendered from
      0 to 90 degrees in 10-degree steps (the model was only trained to 40).
      -> shows exactly where the model breaks beyond its training range.

Each snapshot is loaded ONCE and used for both experiments (loading is the
expensive part). Uses the exact training render recipe.

    python3 render_experiments.py            # single process
    python3 render_experiments.py 0 4        # shard 0 of 4 (run 4 in parallel)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph
import os, sys, gc, glob
import time as timer

IMG_SIZE, WIDTH = 224, 40.0
VMIN, VMAX = 1e7, 1e9
SWEEP_TILTS = range(0, 91, 10)          # 0..90 degrees
OUT_EVO, OUT_SWEEP = "exp_evolution", "exp_tiltsweep"

SHARD, NSHARDS = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
os.makedirs(OUT_EVO, exist_ok=True)
os.makedirs(OUT_SWEEP, exist_ok=True)

test_snaps = set(l.strip() for l in open("test_snaps.txt") if l.strip())
snaps = sorted(os.path.basename(p) for p in glob.glob("snapshot_[0-9]*")
               if os.path.getsize(p) == 96000312)
mine = [s for i, s in enumerate(snaps) if i % NSHARDS == SHARD]
print(f"shard {SHARD}/{NSHARDS}: {len(mine)} snapshots "
      f"({sum(1 for s in mine if s in test_snaps)} of them are test snaps)", flush=True)


def render(sim, path):
    im = sph.image(sim.star, qty="rho", width=WIDTH, resolution=IMG_SIZE,
                   log=False, noplot=True, show_cbar=False, approximate_fast=False)
    im = np.log10(np.clip(im, VMIN, VMAX))
    im = (im - np.log10(VMIN)) / (np.log10(VMAX) - np.log10(VMIN))
    plt.imsave(path, im, cmap="gray", vmin=0.0, vmax=1.0, origin="lower")


t0 = timer.time()
for n, snap in enumerate(mine, 1):
    try:
        sim = pynbody.load(snap)
        sim.physical_units()
        sim["pos"] *= 1.0 / 1000.0
        halo.center(sim)
        angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
        aligned = sim["pos"].copy()

        # EXPERIMENT A: face-on view of every snapshot
        render(sim, f"{OUT_EVO}/{snap}_X00_Y00.png")

        # EXPERIMENT B: tilt sweep, test snapshots only
        if snap in test_snaps:
            for t in SWEEP_TILTS:
                sim["pos"][:] = aligned
                if t:
                    sim.rotate_x(t)
                render(sim, f"{OUT_SWEEP}/{snap}_X{t:02d}_Y00.png")

        del sim; gc.collect()
        el = timer.time() - t0
        print(f"[{n}/{len(mine)}] {snap}  ({el/n:.0f}s/snap, "
              f"~{el/n*(len(mine)-n)/60:.0f} min left)", flush=True)
    except Exception as e:
        print(f"ERROR on {snap}: {e}", flush=True)

print(f"shard {SHARD} done in {(timer.time()-t0)/60:.1f} min", flush=True)
