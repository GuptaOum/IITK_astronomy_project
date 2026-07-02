"""
Compute peak A2/A0 bar strength for each snapshot and save labels.
Label: 1 = barred (A2/A0 > threshold), 0 = not barred.

Output: bar_labels.csv with columns [snapshot, time_gyr, peak_a2a0, label]
"""

import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import numpy as np
import pandas as pd
import os
import gc

# --- Config ---
START = 0
END = 501
BAR_THRESHOLD = 0.19  # A2/A0 above this = "barred"; adjust after inspecting the plot
OUTPUT_CSV = "bar_labels.csv"

results = []

for i in range(START, END + 1):
    snap_file = f"snapshot_{i:03d}"
    if not os.path.exists(snap_file):
        print(f"Skipping {snap_file} (not found)")
        continue

    try:
        s = pynbody.load(snap_file)
        s.physical_units()
        s['pos'] *= 1.0 / 1000.0

        halo.center(s, mode='ssc')
        angmom.faceon(s.star, disk_size=4.0, move_all=True, already_centered=True)

        time_gyr = s.properties['time'].in_units('Gyr') / 1000.0

        # Star particle data
        stars = s.star
        mass = stars['mass']
        pos = stars['pos']
        r_cyl = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        phi = np.arctan2(pos[:, 1], pos[:, 0])

        # Compute A2/A0 in radial bins, find peak
        bin_edges = np.arange(0, 16, 1.0)
        peak_strength = 0.0

        for j in range(len(bin_edges) - 1):
            mask = (r_cyl >= bin_edges[j]) & (r_cyl < bin_edges[j + 1])
            if np.sum(mask) > 0:
                m_bin = mass[mask]
                phi_bin = phi[mask]
                a2 = np.sum(m_bin * np.cos(2 * phi_bin))
                b2 = np.sum(m_bin * np.sin(2 * phi_bin))
                a0 = np.sum(m_bin)
                strength = np.sqrt(a2**2 + b2**2) / a0
                peak_strength = max(peak_strength, strength)

        label = 1 if peak_strength > BAR_THRESHOLD else 0
        results.append({
            'snapshot': f"snapshot_{i:03d}",
            'time_gyr': round(time_gyr, 4),
            'peak_a2a0': round(float(peak_strength), 4),
            'label': label,
        })

        print(f"[{i:03d}] T={time_gyr:.3f} Gyr  A2/A0={peak_strength:.4f}  label={label}")

        del s
        gc.collect()

    except Exception as e:
        print(f"Error on snapshot {i}: {e}")

# Save
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

# Summary
n_barred = df['label'].sum()
n_total = len(df)
print(f"\nSaved {OUTPUT_CSV}: {n_total} snapshots, {n_barred} barred, {n_total - n_barred} not barred")
print(f"Threshold used: A2/A0 > {BAR_THRESHOLD}")
print(f"\nA2/A0 stats: min={df['peak_a2a0'].min():.4f}, max={df['peak_a2a0'].max():.4f}, mean={df['peak_a2a0'].mean():.4f}")
