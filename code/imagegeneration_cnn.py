
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph
import os
import time as timer
import gc

# --- Config ---
START = 0
END = 501
WIDTH = 50.0          # tighter crop — bar region is ~10-15 kpc
RESOLUTION = 400
DPI = 150
APPROXIMATE_FAST = True
OUTPUT_DIR = "cnn_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(START, END + 1):
    snap_file = f"snapshot_{i:03d}"
    if not os.path.exists(snap_file):
        print(f"File {snap_file} does not exist. Skipping.")
        continue

    t0 = timer.time()

    sim = pynbody.load(snap_file)
    sim.physical_units()
    sim['pos'] *= 1 / 1000.0
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)

    # Clean image: no colorbar, no title, no axes
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sph.image(
        sim.star,
        qty="rho",
        width=WIDTH,
        cmap="gray",          # grayscale — simpler for CNN
        log=True,
        resolution=RESOLUTION,
        approximate_fast=APPROXIMATE_FAST,
        show_cbar=False,      # no colorbar
        subplot=ax,
    )

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no margins
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"snapshot_{i:03d}.png"),
        dpi=DPI,
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close('all')
    del sim
    gc.collect()
    elapsed = timer.time() - t0
    print(f"[{i:03d}/{END}] done in {elapsed:.1f}s")
