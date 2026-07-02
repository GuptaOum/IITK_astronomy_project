import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph

from matplotlib.colors import LinearSegmentedColormap
import time as timer

SNAP_FILE = "snapshot_501"
WIDTH = 40.0
RESOLUTION = 600
DPI = 200

# Test with the same angle that looked bad
RX, RY, RZ = 0, 30, 0

print(f"Loading {SNAP_FILE}...")
t0 = timer.time()

sim = pynbody.load(SNAP_FILE)
sim.physical_units()
sim['pos'] *= 1.0 / 1000.0

halo.center(sim)
angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
print(f"Loaded and aligned in {timer.time() - t0:.1f}s")
colors = [
    (0.0, 0.0, 1.0),   # blue
    (0.0, 1.0, 0.0),   # green
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 0.0),   # red
    (0.4, 0.0, 0.4),   # purple
    (1.0, 1.0, 1.0),   # white
]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Apply rotation
if RX != 0:
    sim.rotate_x(RX)
if RY != 0:
    sim.rotate_y(RY)
if RZ != 0:
    sim.rotate_z(RZ)

# Render
fig, ax = plt.subplots(figsize=(4, 4))

sph.image(
    sim.star,
    qty="rho",
    width=WIDTH,
    resolution=RESOLUTION,
    cmap='berlin',
    vmin=1e6,
    vmax=3e9,
    log=True,
    approximate_fast=False,
    show_cbar=False,
    axes=ax
)

ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

fname = f"test_rot_X{RX:02d}_Y{RY:02d}_Z{RZ:02d}.png"
fig.savefig(fname, dpi=DPI, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Saved {fname} in {timer.time() - t0:.1f}s")
