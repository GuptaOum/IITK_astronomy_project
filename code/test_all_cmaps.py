import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph
import numpy as np
import os
import time as timer

SNAP_FILE = "snapshot_501"
WIDTH = 40.0
RESOLUTION = 400
DPI = 200
OUTPUT_DIR = "cmap_samples"

CMAPS = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
    'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
    'Grays', 'Grays_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
    'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
    'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn',
    'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r',
    'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
    'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
    'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
    'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
    'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'berlin', 'berlin_r', 'binary',
    'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
    'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
    'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
    'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r',
    'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
    'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot',
    'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot',
    'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r',
    'managua', 'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
    'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis',
    'viridis_r', 'winter', 'winter_r',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load simulation once
print(f"Loading {SNAP_FILE}...")
t0 = timer.time()

sim = pynbody.load(SNAP_FILE)
sim.physical_units()
sim['pos'] *= 1.0 / 1000.0

halo.center(sim)
angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
print(f"Loaded and aligned in {timer.time() - t0:.1f}s")

# Pre-render the SPH image data once, then just re-color it
print("Rendering SPH image data...")
rendered = sph.image(
    sim.star,
    qty="rho",
    width=WIDTH,
    resolution=RESOLUTION,
    log=True,
    noplot=True,
    approximate_fast=False,
)
print(f"SPH render done in {timer.time() - t0:.1f}s")

# Normalize for consistent vmin/vmax
log_data = np.log10(np.clip(rendered, 1e6, 3e9))
norm_data = (log_data - np.log10(1e6)) / (np.log10(3e9) - np.log10(1e6))

# Loop through all cmaps and save
print(f"Generating {len(CMAPS)} colormap images...")
for i, cmap_name in enumerate(CMAPS):
    try:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(norm_data, cmap=cmap_name, origin='lower', vmin=0, vmax=1)
        ax.set_title(cmap_name, fontsize=10, fontweight='bold')
        ax.set_axis_off()
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

        fname = os.path.join(OUTPUT_DIR, f"{cmap_name}.png")
        fig.savefig(fname, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  [{i+1}/{len(CMAPS)}] {cmap_name}")
    except Exception as e:
        print(f"  [{i+1}/{len(CMAPS)}] FAILED {cmap_name}: {e}")
        plt.close('all')

print(f"\nDone! {len(CMAPS)} images saved to '{OUTPUT_DIR}/' in {timer.time() - t0:.1f}s")
