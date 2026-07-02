import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.analysis.profile as profile

# -----------------------------------------------------
# Number of snapshotsa
# -----------------------------------------------------
N_snaps = 502

# -----------------------------------------------------
# Allocate arrays
# -----------------------------------------------------
time = np.zeros(N_snaps)
sigma_r_10kpc = np.zeros(N_snaps)
sigma_z_10kpc = np.zeros(N_snaps)   # vertical = sigma_e in your code

# -----------------------------------------------------
# MAIN SNAPSHOT LOOP
# -----------------------------------------------------
for i in range(N_snaps):
    snap_file = f"snapshot_{i:03d}"
    print(f"Loading {snap_file} ...")

    s = pynbody.load(snap_file)
    s.physical_units()

    # convert pc → kpc (if required by your sim)
    s['pos'] *= 1.0 / 1000.0

    # center and align face-on
   # halo.center(s, mode='ssc')
   # angmom.faceon(s.star, disk_size=4.0, move_all=True, already_centered=True)

    # record time
    time[i] = s.properties['time'].in_units('Gyr') / 1000.0

    # compute dispersions using radial profile
    p = profile.Profile(s.star, min=1, max=11, nbins=24)

    vr_disp_bins = p['vr_disp']
    vz_disp_bins = p['vz_disp']
    mass_bins = p['mass']

    # mass-weighted RMS dispersions
    sigma_r_10kpc[i] = np.sqrt(np.sum((vr_disp_bins**2) * mass_bins) /
                               np.sum(mass_bins))

    sigma_z_10kpc[i] = np.sqrt(np.sum((vz_disp_bins**2) * mass_bins) /
                               np.sum(mass_bins))

    print(f"Snapshot {i:03d} | Time = {time[i]:.3f} Gyr | "
          f"σ_r = {sigma_r_10kpc[i]:.2f} km/s | σ_z = {sigma_z_10kpc[i]:.2f} km/s")

# -----------------------------------------------------
# CREATE DATAFRAME & SAVE CSV
# -----------------------------------------------------
'''df = pd.DataFrame({
    'time_Gyr': time,
    'sigma_r_10kpc': sigma_r_10kpc,
    'sigma_z_10kpc': sigma_z_10kpc
})'''
#df.to_csv("radial_vertical_dispersion_vs_time.csv", index=False)
#print("Saved radial_vertical_dispersion_vs_time.csv")

# -----------------------------------------------------
# Compute ratio σz / σr
# -----------------------------------------------------
sigma_ratio = sigma_z_10kpc / sigma_r_10kpc

'''df_ratio = pd.DataFrame({
    'time_Gyr': time,
    'sigma_r_10kpc': sigma_r_10kpc,
    'sigma_z_10kpc': sigma_z_10kpc,
    'sigma_z_over_sigma_r': sigma_ratio
})
df_ratio.to_csv("sigma_z_over_sigma_r_vs_time.csv", index=False)

print("Saved sigma_z_over_sigma_r_vs_time.csv")'''

# -----------------------------------------------------
# PLOTS
# -----------------------------------------------------

# ---------- Plot 1: Radial & Vertical Dispersions ----------
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(time, sigma_r_10kpc, marker='o', color='green',
        label=r'$\sigma_R$ (radial)', markersize=2)
ax.plot(time, sigma_z_10kpc, marker='o', color='blue',
        label=r'$\sigma_z$ (vertical)', markersize=2)

ax.set_xlabel("Time (Gyr)")
ax.set_ylabel("Velocity Dispersion (km/s)")
ax.set_title("Radial & Vertical Velocity Dispersion vs Time")
ax.legend()

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.9)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
ax.tick_params(which='both', direction='inout', top=True, right=True)

plt.tight_layout()
plt.show()

# ---------- Plot 2: σz / σr ----------
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(time, sigma_ratio, marker='o', color='purple', markersize=2,
        label=r'$\sigma_z / \sigma_R$ (within 10 kpc)')

ax.set_xlabel("Time (Gyr)")
ax.set_ylabel(r"$\sigma_z / \sigma_R$")
ax.set_title("Vertical-to-Radial Dispersion Ratio")
ax.legend()

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.9)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
ax.tick_params(which='both', direction='inout', top=True, right=True)

plt.tight_layout()
plt.show()

print("All plots complete.")
