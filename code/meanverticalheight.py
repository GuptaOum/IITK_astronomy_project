
import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pynbody.analysis.halo as halo

N_snaps = 502
r_max = 10.0   
mean_z = np.zeros(N_snaps)
time = np.zeros(N_snaps)  

for i in range(N_snaps):
    snap_file = f"snapshot_{i:03d}"  
    s = pynbody.load(snap_file)
    s.physical_units()


    time[i] = s.properties['time'].in_units('Gyr')/1000.0


    s['pos'] *= 1.0 / 1000.0
    
    

    
    
    r = np.sqrt(s.star['x']**2 + s.star['y']**2)
    stars_in_disk = s.star[r <= r_max]
    
    
    mean_z[i] = np.mean(np.abs(stars_in_disk['z']))
    
    print(f"Snapshot {i:03d}, Time = {time[i]:.2f} Gyr: Mean |z| = {mean_z[i]:.2f} kpc")


plt.figure(figsize=(10,6))
plt.plot(time, mean_z, color='green')
plt.xlabel("Time (Gyr)")
plt.ylabel("Mean |z| (kpc)")
plt.title(f"Disk Mean Vertical Height within {r_max} kpc over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("mean_vertical_height_vs_time.png")