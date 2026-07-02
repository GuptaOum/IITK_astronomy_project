import numpy as np
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.profile as profile
import pynbody.analysis.angmom as angmom
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

start_snap = 0
end_snap = 501
snap_prefix = "snapshot_"
Rmax = (2.0, 12.0)
#radial_bin_width = 1.0
#radial_bins = np.arange(0, Rmax + radial_bin_width, radial_bin_width)
BUCKLE_THRESHOLD = 0.02
outdir = "buckling_plots"
os.makedirs(outdir, exist_ok=True)

def compute_Buckle_for_particles(pos, z, m):
    x, y = pos[:, 0], pos[:, 1]
    phi = np.arctan2(y, x)
    numerator = np.sum(m * z * np.exp(2j * phi))
    denom = np.sum(m)
    print(f"buckle aplitude {np.abs(float(numerator / denom))}")
    try:
        return np.abs(float(numerator / denom))
    except ZeroDivisionError:
        return 0.0

'''def radial_Buckle_map(pos, z, m, r_bins):
    R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    nbins = len(r_bins) - 1
    arr = np.zeros(nbins)
    for i in range(nbins):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.count_nonzero(mask) < 5:
            arr[i] = np.nan
            continue
        arr[i] = compute_Buckle_for_particles(pos[mask], z[mask], m[mask])
    centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    return centers, arr'''

times, buckle_ts = [], []
#radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
radial_map = []

for i in range(start_snap, end_snap + 1):
    print(f"Processing snapshot {i:03d} ...")
    snap_file = f"{snap_prefix}{i:03d}"
    if not os.path.exists(snap_file):
        print(f"Snapshot {i:03d} not found, skipping.")
        continue
    s = pynbody.load(snap_file)
    s.physical_units()
    s=s.star
    s['pos'] *= 1.0 / 1000.0
    halo.center(s, mode='ssc')
    angmom.sideon(s)
    
    pos = np.array(s['pos'])[:, :2]
    z = np.array(s['pos'])[:, 2]
    m = np.array(s['mass'])
    R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    mask = (R >= Rmax[0]) & (R <= Rmax[1])
    buckle = compute_Buckle_for_particles(pos[mask], z[mask], m[mask])
    '''centers, buckle_radial = radial_Buckle_map(pos[mask], z[mask], m[mask], radial_bins)'''
    
    time = float(s.properties['time']/ 1000)
    times.append(time)
    buckle_ts.append(buckle)
    ''' radial_map.append(buckle_radial)'''# (502, 24) array its diagnostic and not always presentable its noisy  

times = np.array(times)
buckle_ts = np.array(buckle_ts)
'''radial_map = np.vstack(radial_map)
print(radial_map.shape)     # should be (502, 24)
print(np.nanmin(radial_map), np.nanmax(radial_map))'''
# Plotting the buckling time series with peaks and threshold

fig, ax = plt.subplots(figsize=(9,5))

# Global buckling curve (thin line)
ax.plot(
    times, buckle_ts,
    linewidth=1.0,
    label=' Buckle amplitude'
)


# Threshold line (thin dashed)
ax.axhline(
    BUCKLE_THRESHOLD,
    color='k',
    linestyle='--',
    linewidth=0.8,
    label='Threshold'
)

# Peaks (small dots)
peaks, _ = find_peaks(buckle_ts, height=0.02) #here height is actually the min height of peaks
ax.plot(
    times[peaks], buckle_ts[peaks],
    'o',
    markersize=3,
    color='red',
    label='Buckling Peaks'
)

ax.set_xlabel('Time (Gyr)')
ax.set_ylabel('Global Buckle')

ax.grid(True, which='both', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(outdir, "buckling_timeseries.png"), dpi=200)
plt.show()
'''fig, ax = plt.subplots(figsize=(10,5))

im = ax.imshow(
    radial_map.T,
    origin='lower',
    aspect='auto',
    extent=[times.min(), times.max(),
            radial_centers.min(), radial_centers.max()],
    cmap='viridis'
)

ax.set_xlabel('Time (Gyr)')
ax.set_ylabel('Radius (kpc)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Radial Buckle Strength')

plt.tight_layout()
plt.show()'''

