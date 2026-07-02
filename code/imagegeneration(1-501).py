
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — faster rendering, no GUI overhead
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
from matplotlib.colors import LinearSegmentedColormap
import pynbody.plot.sph as sph
import os
import time as timer
import gc

# --- Build colormap once (was being recreated every iteration) ---
colors = [
    (0.0, 0.0, 1.0),   # blue
    (0.0, 1.0, 0.0),   # green
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 0.0),   # red
    (0.4, 0.0, 0.4),   # purple
    (1.0, 1.0, 1.0),   # white
]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors)

# --- Config (tune these knobs) ---
START = 500
END = 501
WIDTH = 54.0
RESOLUTION = 400      # 500 vs 1000 = 4x fewer pixels, visually nearly identical
DPI = 225             # 150 vs 500 = much smaller PNGs, still sharp
APPROXIMATE_FAST = True  # grid interpolation instead of exact SPH kernel — huge speedup

for i in range(START, END + 1):
    snap_file = f"snapshot_{i:03d}"
    if not os.path.exists(snap_file):
        print(f"File {snap_file} does not exist. Skipping.")
        continue

    t0 = timer.time()

    sim = pynbody.load(snap_file)
    sim.physical_units()
    sim['pos']*=1/1000.0  # convert from kpc/h to kpc
    halo.center(sim)
    angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)

    sph.image(
        sim.star,
        qty="rho",
        width=WIDTH,
        cmap= mycmap,
        log=True,
        resolution=RESOLUTION,
        approximate_fast=APPROXIMATE_FAST,
        colorbar_label=r"$\rho$ (M$_\odot$/kpc$^3$)",
        
    )

    plt.title(f"Face-on Stellar Density {i:03d}")
    plt.tight_layout()
    plt.savefig(f"snapshot_{i:03d}.png", dpi=DPI)
    plt.close('all')  # free memory — prevents figure accumulation leak
    del sim
    gc.collect()
    elapsed = timer.time() - t0
    print(f"[{i:03d}/{END}] done in {elapsed:.1f}s")
    
