import time
import numpy as np
from numpy.random import rand
import scipy.integrate as integ
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scattering, dsd
from constants import cm_per_m, g_per_kg

ref_adjust = 180.

lam = 0.0321 # m
temp = 10.0 # oC
d = np.linspace(0.01, 1.0, 200).reshape(-1, 1) / cm_per_m
qr = np.linspace(0.0001, 25.0, 400).reshape(1,-1) / g_per_kg

dist = dsd.mp_from_lwc(d, qr)

ts = time.time()

ref = []
#temps = np.arange(-10, 35, 5)
temps = np.array([9.9, 10.0, 10.1])
for temp in temps:
    model = scattering.scatterer(lam, temp, 'water', diameters = d)
    model.set_scattering_model('tmatrix')
    ref.append(10.0 * np.log10(model.get_reflectivity_factor(dist)) + ref_adjust)
ref = np.array(ref)
Q,T = np.meshgrid(qr.squeeze(), temps)
#ref_lut = interp.interp2d(Q, T, ref)
tck = interp.bisplrep(Q, T, ref)

temp = np.array([10.0]*1000)
#qs = np.random.rand(1000)*0.005 + 0.01
#qs = np.array([0.01]*1000)
qs = np.linspace(0.0001, 0.025, 1000)
print 'Performing lookup...'
z = interp.bisplev(qs, temp, tck)

print time.time() - ts

plt.subplot(1, 2, 1)
for r,T0 in zip(ref, temps):
    plt.plot(qr.squeeze() * g_per_kg, r, label='%d degC' % T0)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(qs, z, 'o')
plt.show()
