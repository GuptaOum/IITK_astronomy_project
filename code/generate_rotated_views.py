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


# -------------------------------
# USER PARAMETERS
# -------------------------------

SNAP_FILE = "snapshot_501"

WIDTH = 40.0          # kpc — R90 is ~14 kpc, show ~2x that
RESOLUTION = 500
DPI = 200

APPROXIMATE_FAST = False

OUTPUT_DIR = "rotated_views_501"

ANGLES = range(0, 100, 10)   # 0..90 degrees


# -------------------------------
# SETUP
# -------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {SNAP_FILE}...")

t0 = timer.time()

sim = pynbody.load(SNAP_FILE)
sim.physical_units()
sim['pos'] *= 1.0 / 1000.0
# center galaxy
halo.center(sim)

# align galaxy face-on
angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
print(f"Loaded and aligned in {timer.time() - t0:.1f}s")


# Save aligned state
aligned_pos = sim['pos'].copy()



# -------------------------------
# IMAGE COUNT
# -------------------------------

total = len(ANGLES) ** 3
count = 0
t_start = timer.time()


# -------------------------------
# ROTATION LOOP
# -------------------------------

for rz in ANGLES:
    for ry in ANGLES:
        for rx in ANGLES:

            count += 1
            t_img = timer.time()

            # Reset galaxy to aligned state
            sim['pos'][:] = aligned_pos
            

            # Apply rotations
            if rx != 0:
                sim.rotate_x(rx)

            if ry != 0:
                sim.rotate_y(ry)

            if rz != 0:
                sim.rotate_z(rz)

            # Create image
            fig, ax = plt.subplots(figsize=(4,4))

            sph.image(
                sim.star,
                qty="rho",
                width=WIDTH,
                resolution=RESOLUTION,
                cmap="berlin",
                vmin=1e6,
                vmax=3e9,
                log=True,
                approximate_fast=False,
                show_cbar=False,
                axes=ax,
            )
            

            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Save image
            fname = f"rot_X{rx:02d}_Y{ry:02d}_Z{rz:02d}.png"

            fig.savefig(
                os.path.join(OUTPUT_DIR, fname),
                dpi=DPI,
                bbox_inches='tight',
                pad_inches=0
            )

            plt.close(fig)

            # Progress info
            img_time = timer.time() - t_img
            elapsed = timer.time() - t_start
            avg = elapsed / count
            remaining = avg * (total - count)

            print(f"[{count:4d}/{total}] {fname}  {img_time:.2f}s  (total {elapsed:.0f}s, ~{remaining:.0f}s remaining)")


# cleanup
del sim

print(f"\nDone! {count} images saved to {OUTPUT_DIR}/")
print(f"Total time: {timer.time() - t_start:.1f}s")