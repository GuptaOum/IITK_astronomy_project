import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
n_snaps = 501
snap_base = "snapshot"
time = np.zeros(n_snaps)
bar_phase = np.zeros(n_snaps) # Stores the raw phase angle of the bar
valid_snaps = np.zeros(n_snaps, dtype=bool) # Keeps track of successful loads

# --- Analysis Loop ---
for i in range(n_snaps):
    try:
        snap_file = f"{snap_base}_{i:03d}"
        
        # Load snapshot
        try:
            s = pynbody.load(snap_file)
        except OSError:
            print(f"Skipping {snap_file} (not found)")
            valid_snaps[i] = False
            continue

        # ==============================================================================
        # PRE-PROCESSING
        # ==============================================================================
        s.physical_units()
        s['pos'] *= 1.0 / 1000.0 
        
        halo.center(s, mode='ssc')
        
        # Align disk face-on so we measure rotation in the x-y plane
        angmom.faceon(s.star, disk_size=4.0, move_all=True, already_centered=True)
        
        # Store time
        time[i] = s.properties['time'].in_units('Gyr') / 1000.0
        # ==============================================================================

        # --- Prepare Star Data ---
        stars = s.star
        mass = stars['mass']
        pos = stars['pos']
        
        # Cylindrical coordinates
        r_cyl = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        # Using arctan2 is CRITICAL for correct 4-quadrant angles
        phi_particles = np.arctan2(pos[:, 1], pos[:, 0]) 

        # --- Find Radius of Peak Bar Strength (A2/A0) ---
        # "measured using the Fourier mode of the annular region corresponding 
        # to the peak in maximum value of A2/A0" [cite: 1258]
        
        bin_edges = np.arange(0, 15.1, 1.0) # 1 kpc bins up to 15 kpc
        a2_strength_profile = []
        phase_profile = []
        
        for j in range(len(bin_edges)-1):
            r_min = bin_edges[j]
            r_max = bin_edges[j+1]
            
            mask = (r_cyl >= r_min) & (r_cyl < r_max)
            if np.sum(mask) > 0:
                m_bin = mass[mask]
                phi_bin = phi_particles[mask]
                
                # Fourier Coefficients
                a2 = np.sum(m_bin * np.cos(2 * phi_bin))
                b2 = np.sum(m_bin * np.sin(2 * phi_bin))
                a0 = np.sum(m_bin)
                
                # Strength magnitude
                strength = np.sqrt(a2**2 + b2**2) / a0
                a2_strength_profile.append(strength)
                
                # Phase angle: 0.5 * arctan(b2/a2)
                # We use arctan2(b2, a2) to preserve quadrant info, then halve it
                phase_val = 0.5 * np.arctan2(b2, a2)
                phase_profile.append(phase_val)
            else:
                a2_strength_profile.append(0.0)
                phase_profile.append(0.0)
        
        # Identify the index of the Peak Strength
        peak_idx = np.argmax(a2_strength_profile)
        
        # Extract the bar phase at that specific radius
        current_phase = phase_profile[peak_idx]
        bar_phase[i] = current_phase
        valid_snaps[i] = True

        print(f"Snap {i}: T={time[i]:.3f}, Peak Radius={bin_edges[peak_idx]}-{bin_edges[peak_idx+1]} kpc, Phase={current_phase:.3f} rad")

    except Exception as e:
        print(f"Error processing snapshot {i}: {e}")
        valid_snaps[i] = False

# ==============================================================================
# POST-PROCESSING: Calculate Pattern Speed (Omega_p)
# ==============================================================================

# Filter only valid snapshots
t_clean = time[valid_snaps]
phi_clean = bar_phase[valid_snaps]

# 1. Unwrap the Phase
# The bar has m=2 symmetry, meaning it looks the same after 180 deg (pi radians).
# The calculated phase is in (-pi/2, pi/2).
# We multiply by 2 to map it to (-pi, pi), unwrap to handle 360 jumps, then divide by 2.
phi_unwrapped = np.unwrap(2 * phi_clean) / 2.0

# 2. Calculate Derivative (d_phi / d_t)
# Using central differences for smoothness
omega_p = np.gradient(phi_unwrapped, t_clean)

# 3. Convert Units
# Current units: Radians / Gyr
# Desired units (from paper): km s^-1 kpc^-1
# Conversion factor: 1 (km/s)/kpc = 1.0227 rad/Gyr
# So: (rad/Gyr) * 0.9778 = (km/s)/kpc
CONVERSION_FACTOR = 0.9778 
omega_p_kms_kpc = np.abs(omega_p) * CONVERSION_FACTOR

# ==============================================================================
# PLOTTING
# ==============================================================================




fig, ax = plt.subplots(figsize=(9, 6))

# Line + dots
ax.plot(t_clean, omega_p_kms_kpc, color='black', linewidth=1.2, label='Omega_p')
#ax.scatter(t_clean, omega_p_kms_kpc, color='blue', s=20)

# Mean line
mean_speed = np.mean(omega_p_kms_kpc)
ax.axhline(mean_speed, color='red', linestyle='--', linewidth=1, label='Mean')

# Peak point
peak_idx = np.argmax(omega_p_kms_kpc)
peak_val = omega_p_kms_kpc[peak_idx]
peak_time = t_clean[peak_idx]

ax.plot(peak_time, peak_val, marker='o', color='green',
        markersize=3, label='Peak')

# Labels & title
ax.set_xlabel('Time (Gyr)')
ax.set_ylabel(r'$\Omega_p$ (km s$^{-1}$ kpc$^{-1}$)')
ax.set_title('Bar Pattern Speed Evolution')

# Grid
# Turn on minor ticks (this creates extra tick locations)
ax.minorticks_on()

# Major grid (based on major ticks)
ax.grid(True, which='major', linestyle='--', alpha=0.5)

# Minor grid (based on minor ticks)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)


# 🔹 Text at top-middle
ax.text(
    0.5, 0.98,
    f"Peak: {peak_val:.1f} km s$^{{-1}}$ kpc$^{{-1}}$   |   "
    f"Mean: {mean_speed:.1f} km s$^{{-1}}$ kpc$^{{-1}}$   |   "
    f"Peak time: {peak_time:.3f} Gyr",
    transform=ax.transAxes,
    fontsize=10,
    ha='center',
    va='top'
)

# Legend
ax.legend()

fig.tight_layout()
fig.savefig('PatternSpeed_Evolution_simple_top_middle.png', dpi=150)
plt.show()
