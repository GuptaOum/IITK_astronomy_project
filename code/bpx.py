import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
n_snaps = 501
snap_base = "snapshot"
time = np.zeros(n_snaps)

# Arrays for the three key metrics
bar_strength = np.zeros(n_snaps)      # A2/A0
buckling_strength = np.zeros(n_snaps) # A_Buckle
bpx_strength = np.zeros(n_snaps)      # A_BPX (Corrected to RMS)

# --- Analysis Loop ---
for i in range(n_snaps):
    try:
        snap_file = f"{snap_base}_{i:03d}"
        
        # Load snapshot
        try:
            s = pynbody.load(snap_file)
        except OSError:
            print(f"Skipping {snap_file} (not found)")
            time[i] = np.nan
            continue

        # ==============================================================================
        # PRE-PROCESSING (Units, Centering, Alignment)
        # ==============================================================================
        s.physical_units()
        s['pos']*= 1.0 /1000.0
        
        halo.center(s, mode='ssc')
        
        # Align the disk face-on based on the inner region
        angmom.faceon(s, disk_size=12,move_all=True, already_centered=True)
        
        s=s.star
        # Store time
        time[i] = s.properties['time'].in_units('Gyr') / 1000.0
        # ==============================================================================

        # --- Prepare Star Data ---

        
        # Cylindrical coordinates
        r_cyl = np.sqrt(s['pos'][:, 0]**2 + s['pos'][:, 1]**2)
        #phi = np.arctan2(s['pos'][:, 1], s['pos'][:, 0])
        z = s['pos'][:, 2]

        # ---------------------------------------------------------
        # 1. Bar Strength (A2/A0)
        # ---------------------------------------------------------


        # ---------------------------------------------------------
        # 2. Buckling Strength (A_Buckle)
        # ---------------------------------------------------------
       

        # ---------------------------------------------------------
        # 3. BPX Strength (A_BPX) - CORRECTED FORMULA (Eq 8 / Snippet 249)
        # Definition: RMS of the vertical height within annular region (2-8 kpc)
        # Formula: sqrt( sum(m * z^2) / sum(m) )
        # ---------------------------------------------------------
        bpx_mask = (r_cyl >= 2.0) & (r_cyl < 8.0)
        if np.sum(bpx_mask) > 0:
            m_px = s['mass'][bpx_mask]
            z_px = z[bpx_mask]
            
            # Corrected to RMS
            weighted_sq_z = np.sum(m_px * (z_px**2))
            total_mass_px = np.sum(m_px)
            
            a_bpx = np.sqrt(weighted_sq_z / total_mass_px)
            bpx_strength[i] = a_bpx
        else:
            bpx_strength[i] = 0.0

        print(f"Snap {i}: T={time[i]:.3f}, BPX={bpx_strength[i]:.3f}")

    except Exception as e:
        print(f"Error processing snapshot {i}: {e}")
        time[i] = np.nan

# ==============================================================================
# PLOTTING - Separate Canvas for Each Metric
# ==============================================================================




# Canvas 3: BPX Strength
plt.figure(figsize=(8, 5))
plt.plot(time, bpx_strength, color='blue', linewidth=1.0)
plt.ylabel(r'$A_{BPX}$ (kpc)')
plt.xlabel('Time (Gyr)')
plt.title('Boxy/Peanut Strength')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Canvas3_BPXStrength.png')
plt.show()