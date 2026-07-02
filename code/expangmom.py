
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom

N_snaps = 502

disk_L = np.zeros(N_snaps)
halo_L = np.zeros(N_snaps)
time= np.zeros(N_snaps)  
snap_nums = np.arange(N_snaps)
for i in range(N_snaps):
    
    snap_file = f"snapshot_{i:03d}"  
    s = pynbody.load(snap_file)
    s.physical_units()
    
    
    s['pos'] *= 1.0 / 1000.0
    

    halo.center(s, mode='ssc')
    

    angmom.faceon(s.star, disk_size=4.0, move_all=True, already_centered=True)
    time[i] = s.properties['time'].in_units('Gyr')/1000  
    
    
    disk_L_vec = angmom.ang_mom_vec(s.star)
    halo_L_vec = angmom.ang_mom_vec(s.dm)
    
    
    disk_L[i] = np.linalg.norm(disk_L_vec)# it is important to take the norm to get the magnitude of the angular momentum vector
    halo_L[i] = np.linalg.norm(halo_L_vec)

    print(f"Snapshot {i:03d}: Disk L = {disk_L[i]:.2e}, Halo L = {halo_L[i]:.2e}")

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(time, disk_L, label="Disk Angular Momentum", color='blue')
ax.plot(time, halo_L, label="Halo Angular Momentum", color='red')
ax.set_xlabel("Time (Gyr)")
ax.set_ylabel(f"Angular Momentum [{disk_L_vec.units}]")
ax.set_title("Angular Momentum Exchange in Disk and Halo")
ax.legend()


ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))   
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))    

ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.9)
ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
ax.tick_params(which='both', direction='inout', top=True, right=True)

plt.tight_layout()
plt.show()
