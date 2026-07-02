import pynbody
import numpy as np
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import pynbody.plot.sph as sph

sim=pynbody.load("snapshot_501")

sim.physical_units()
sim['pos'] *= 1.0 / 1000.0
halo.center(sim)
angmom.faceon(sim, move_all=True, already_centered=True, disk_size=15.0)
#lets calculate the density of the stars in the disk plane. We can do this by calculating the distance of each star from the center of the galaxy and then binning the stars into annuli. We can then calculate the density by using pynbody deafult functions
print(np.max(sim['rho']))
print(np.min(sim['rho']))
print(np.mean(sim['rho']))
print(np.median(sim['rho']))
print(sim['rho'].shape
      )
#lets calcuate the max radius
print(np.max(sim['r']))
