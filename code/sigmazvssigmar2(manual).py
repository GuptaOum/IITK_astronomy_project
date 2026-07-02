import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom

# --- Configuration based on papers ---
N_snaps = 502
Rd = 2.9  # Scale length from Paper [cite: 90, 894]
R_MAX_LIMIT = 2.0 * Rd  # Calculation region (5.8 kpc) [cite: 249, 996]

time = np.zeros(N_snaps)
sigma_ratio = np.zeros(N_snaps)

for i in range(N_snaps):
    snap_file = f"snapshot_{i:03d}"
    try:
        s = pynbody.load(snap_file)
        s.physical_units()
        
        # User-specified position shift
        s['pos'] *= 1.0 / 1000.0
        
        # Center and Align
        halo.center(s, mode='ssc')
        angmom.faceon(s.star, disk_size=4.0, move_all=True, already_centered=True)
        
        # Record Time
        time[i] = s.properties['time'].in_units('Gyr') / 1000.0
        
        # --- Built-in Particle Statistics (Avoiding Profile Gradient Error) ---
        # Select stars within the 2*Rd region [cite: 996]
        stars_in_region = s.star[pynbody.filt.LowPass('rxy', R_MAX_LIMIT)]
        
        if len(stars_in_region) > 1:
            # Use pynbody's built-in derived arrays for dispersion
            # vr_disp and vz_disp calculate the standard deviation of velocities
            # in the selected group [cite: 151, 990]
            sig_R = stars_in_region['vr'].std() #'vr' is the radial velocity component of the snapshot
            sig_z = stars_in_region['vz'].std()  #'vz' is the vertical velocity component of the snapshot
            
            sigma_ratio[i] = sig_z / sig_R
            
        print(f"Snap {i:03d}: Time={time[i]:.3f}, Ratio={sigma_ratio[i]:.4f}")
        
    except Exception as e:
        print(f"Error snap {i}: {e}")
        time[i], sigma_ratio[i] = np.nan, np.nan

# --- Plotting to Match Figure 1 (2024 Paper) ---

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(time, sigma_ratio, color='black', linewidth=1.5)

ax.set_xlabel("Time (Gyr)", fontsize=12)
ax.set_ylabel(r"$\sigma_z / \sigma_R$", fontsize=12)
ax.set_title(r"Velocity Dispersion Ratio ($R < 2R_d$)")

# Expected range based on Figure 1 of 2024 paper [cite: 978, 982]
ax.set_ylim(0.6, 0.85) 
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(direction='in', top=True, right=True)

plt.tight_layout()
plt.show()