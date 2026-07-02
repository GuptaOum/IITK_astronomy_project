"""
Combined plot script:
  Panel 1: σ_z vs σ_r (scatter/time-colored)
  Panel 2: A_buckle vs time
  Panel 3: BPX strength vs time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.analysis.profile as profile
from scipy.signal import find_peaks
import os
import gc

# ============================================================
# Configuration
# ============================================================
N_SNAPS = 502
SNAP_PREFIX = "snapshot_"
R_BUCKLE = (2.0, 12.0)      # radial range for buckling (kpc)
R_BPX = (2.0, 7.0)          # radial range for BPX strength (kpc)
PROFILE_MIN = 1              # min radius for dispersion profile (kpc)
PROFILE_MAX = 11             # max radius for dispersion profile (kpc)
PROFILE_NBINS = 24
BUCKLE_THRESHOLD = 0.02

# ============================================================
# Allocate arrays
# ============================================================
times = np.full(N_SNAPS, np.nan)
sigma_r = np.zeros(N_SNAPS)
sigma_z = np.zeros(N_SNAPS)
buckle_amp = np.zeros(N_SNAPS)
bpx_strength = np.zeros(N_SNAPS)


def compute_buckle(pos_xy, z, mass):
    """Compute A_buckle = |sum(m * z * exp(2i*phi)) / sum(m)|"""
    phi = np.arctan2(pos_xy[:, 1], pos_xy[:, 0])
    numerator = np.sum(mass * z * np.exp(2j * phi))
    denom = np.sum(mass)
    if denom == 0:
        return 0.0
    return np.abs(numerator / denom)


# ============================================================
# Main snapshot loop
# ============================================================
for i in range(N_SNAPS):
    snap_file = f"{SNAP_PREFIX}{i:03d}"
    if not os.path.exists(snap_file):
        print(f"Skipping {snap_file} (not found)")
        continue

    try:
        print(f"Processing {snap_file} ...")
        s = pynbody.load(snap_file)
        s.physical_units()
        s['pos'] *= 1.0 / 1000.0

        # Center and align
        halo.center(s, mode='ssc')
        angmom.faceon(s.star, disk_size=4.0, move_all=True,
                      already_centered=True)

        times[i] = s.properties['time'].in_units('Gyr') / 1000.0
        stars = s.star

        # ----------------------------------------------------------
        # 1. σ_r and σ_z (mass-weighted profile-based)
        # ----------------------------------------------------------
        p = profile.Profile(stars, min=PROFILE_MIN, max=PROFILE_MAX,
                            nbins=PROFILE_NBINS)
        vr_disp_bins = p['vr_disp']
        vz_disp_bins = p['vz_disp']
        mass_bins = p['mass']

        sigma_r[i] = np.sqrt(np.sum(vr_disp_bins**2 * mass_bins) /
                             np.sum(mass_bins))
        sigma_z[i] = np.sqrt(np.sum(vz_disp_bins**2 * mass_bins) /
                             np.sum(mass_bins))

        # ----------------------------------------------------------
        # 2. BPX strength: sqrt(sum(m * z^2) / sum(m)) in annulus
        # ----------------------------------------------------------
        pos = np.array(stars['pos'])
        r_cyl = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        z_arr = pos[:, 2]
        mass_arr = np.array(stars['mass'])

        bpx_mask = (r_cyl >= R_BPX[0]) & (r_cyl < R_BPX[1])
        if np.sum(bpx_mask) > 0:
            m_px = mass_arr[bpx_mask]
            z_px = z_arr[bpx_mask]
            bpx_strength[i] = np.sqrt(np.sum(m_px * z_px**2) /
                                      np.sum(m_px))

        # ----------------------------------------------------------
        # 3. Buckling amplitude (side-on needed)
        #    Re-align to side-on for buckling calculation
        # ----------------------------------------------------------
        angmom.sideon(s)
        pos_so = np.array(stars['pos'])
        z_so = pos_so[:, 2]
        r_so = np.sqrt(pos_so[:, 0]**2 + pos_so[:, 1]**2)

        buck_mask = (r_so >= R_BUCKLE[0]) & (r_so <= R_BUCKLE[1])
        if np.sum(buck_mask) > 0:
            buckle_amp[i] = compute_buckle(pos_so[buck_mask, :2],
                                           z_so[buck_mask],
                                           mass_arr[buck_mask])

        print(f"  T={times[i]:.3f} Gyr | σ_r={sigma_r[i]:.1f} | "
              f"σ_z={sigma_z[i]:.1f} | A_buckle={buckle_amp[i]:.4f} | "
              f"BPX={bpx_strength[i]:.4f}")

        del s
        gc.collect()

    except Exception as e:
        print(f"Error on snapshot {i}: {e}")

# ============================================================
# Filter out NaN time entries
# ============================================================
valid = ~np.isnan(times)
t = times[valid]
sr = sigma_r[valid]
sz = sigma_z[valid]
ab = buckle_amp[valid]
bpx = bpx_strength[valid]

# ============================================================
# PLOT 1: σ_z and σ_r vs time
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t, sr, linewidth=1.0, color='green', label=r'$\sigma_R$ (radial)')
ax1.plot(t, sz, linewidth=1.0, color='blue', label=r'$\sigma_z$ (vertical)')

ax1.set_xlabel('Time (Gyr)', fontsize=13)
ax1.set_ylabel('Velocity Dispersion (km/s)', fontsize=13)
ax1.set_title(r'$\sigma_z$ and $\sigma_R$ vs Time', fontsize=14)
ax1.legend(fontsize=11)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax1.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax1.tick_params(which='both', direction='inout', top=True, right=True)
plt.tight_layout()
plt.savefig('plot_sigma_z_sigma_r_vs_time.png', dpi=200)
plt.show()

# ============================================================
# PLOT 2: A_buckle vs time
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(t, ab, linewidth=1.0, color='tab:red', label=r'$A_{\mathrm{buckle}}$')
ax2.axhline(BUCKLE_THRESHOLD, color='k', linestyle='--', linewidth=0.8,
            label=f'Threshold = {BUCKLE_THRESHOLD}')

# Mark peaks
peaks, _ = find_peaks(ab, height=BUCKLE_THRESHOLD)
if len(peaks) > 0:
    ax2.plot(t[peaks], ab[peaks], 'o', markersize=5, color='darkred',
             label='Buckling peaks')

ax2.set_xlabel('Time (Gyr)', fontsize=13)
ax2.set_ylabel(r'$A_{\mathrm{buckle}}$', fontsize=13)
ax2.set_title('Buckling Strength vs Time', fontsize=14)
ax2.legend(fontsize=11)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax2.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.5)
ax2.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax2.tick_params(which='both', direction='inout', top=True, right=True)
plt.tight_layout()
plt.savefig('plot_buckle_vs_time.png', dpi=200)
plt.show()

# ============================================================
# PLOT 3: BPX strength vs time
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(t, bpx, linewidth=1.0, color='tab:blue',
         label=r'$A_{\mathrm{BPX}}$ (2–8 kpc)')

ax3.set_xlabel('Time (Gyr)', fontsize=13)
ax3.set_ylabel(r'$A_{\mathrm{BPX}}$ (kpc)', fontsize=13)
ax3.set_title('Boxy/Peanut Strength vs Time', fontsize=14)
ax3.legend(fontsize=11)

ax3.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax3.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.5)
ax3.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
ax3.tick_params(which='both', direction='inout', top=True, right=True)
plt.tight_layout()
plt.savefig('plot_bpx_vs_time.png', dpi=200)
plt.show()

print("\nAll plots saved: plot_sigma_z_vs_sigma_r.png, "
      "plot_buckle_vs_time.png, plot_bpx_vs_time.png")