import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
N_snaps = 502
snap_nums = np.arange(N_snaps)

Lz_disk = np.zeros(N_snaps)
Lz_halo = np.zeros(N_snaps)
time = np.zeros(N_snaps)
for i in range(N_snaps):
    snap_file = f"snapshot_{i:03d}"
    s = pynbody.load(snap_file)
    s.physical_units()
    s['pos'] *= 1.0 / 1000.0
    # --- Center galaxy ---
    halo.center(s, mode='ssc')

    # --- Align disk (face-on) ---
    angmom.faceon(
        s.star,
        disk_size=3.0,
        move_all=True,
        already_centered=True
    )

    # --- Time ---
    time[i] = s.properties['time'].in_units('Gyr')

    # --- Angular momentum vectors ---
    L_disk = angmom.ang_mom_vec(s.star)
    L_halo = angmom.ang_mom_vec(s.dm)

    # --- z-components ---
    Lz_disk[i] = L_disk[2]
    Lz_halo[i] = L_halo[2]

    print(
        f"Snap {i:03d} | "
        f"Lz_disk = {Lz_disk[i]:.3e},"
        f"Lz_halo = {Lz_halo[i]:.3e}"
    )
Delta_Lz_disk = Lz_disk - Lz_disk[0]  # Why we choose snapshot 0 as reference It defines:the initial angular momentum budget before the bar forms / grows You could choose another reference snapshot:
Delta_Lz_halo = Lz_halo - Lz_halo[0]
plt.figure(figsize=(9,6))

plt.plot(time, Delta_Lz_disk, label="ΔLz Disk", color="blue")
plt.plot(time, Delta_Lz_halo, label="ΔLz Halo", color="red")

plt.axhline(0, color='k', linestyle='--', linewidth=0.8)

plt.xlabel("Time (Gyr)")
plt.ylabel("Δ Angular Momentum")
plt.title("Angular Momentum Exchange Between Disk and Halo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
