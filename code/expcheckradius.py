import pynbody
import pynbody.analysis.halo as halo
import pynbody.analysis.angmom as angmom
import numpy as np
sim = pynbody.load("snapshot_501")
sim.physical_units()
sim['pos'] *= 1.0 / 1000.0
halo.center(sim)

angmom.faceon(sim,  move_all=True, already_centered=True)


stars = sim.star
r = stars['r']           # distance from center
m = stars['mass']

# sort by radius
idx = r.argsort()
r_sorted = r[idx]
m_sorted = m[idx]

cum_mass = m_sorted.cumsum()
total_mass = cum_mass[-1]

# R90
R90 = r_sorted[cum_mass >= 0.9 * total_mass][0]
diameter = 2 * R90
print(f"R90: {R90:.3f} kpc, Diameter: {diameter:.3f} kpc")