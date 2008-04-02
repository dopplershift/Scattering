import matplotlib.pyplot as P
import numpy as N
import scattering
from constants import cm_per_m, mm_per_m

d = N.linspace(0, 100, 500) / mm_per_m
T = 0.0
wavelengths = N.array([10.0, 5.5, 3.21]) / cm_per_m
lines = ['r--','b:','g-']

m_water = N.array([N.sqrt(80.255+24.313j), N.sqrt(65.476+37.026j),
    N.sqrt(44.593+41.449j)])
m_ice = N.array([N.sqrt(3.16835+0.02492j), N.sqrt(3.16835+0.01068j),
    N.sqrt(3.16835+0.0089j)])

P.figure()
for mw, mi, lam, line in zip(m_water, m_ice, wavelengths, lines):
    scat = scattering.scatterer(lam, T, 'water', diameters=d, ref_index=mw)
    scat.set_scattering_model('tmatrix')
    P.subplot(1,2,1)
    P.semilogy(d * mm_per_m, scat.sigma_b * cm_per_m**2, line,
        label='%5.2fcm Tmat' % (lam * cm_per_m))
    scat = scattering.scatterer(lam, T, 'ice', diameters=d, ref_index=mi)
    scat.set_scattering_model('tmatrix')
    P.subplot(1,2,2)
    P.semilogy(d * mm_per_m, scat.sigma_b * cm_per_m**2, line,
        label='%5.2fcm Tmat' % (lam * cm_per_m))

P.subplot(1,2,1)
P.xlabel('Diameter (mm)')
P.ylabel(r'Backscatter Cross-Section (cm$^{2}$)')
P.xlim(0,100.0)
P.ylim(1.0e-2,1e3)

P.subplot(1,2,2)
P.xlabel('Diameter (mm)')
P.ylabel(r'Backscatter Cross-Section (cm$^{2}$)')
P.xlim(0,100.0)
P.ylim(1.0e-2,1e3)

P.legend(loc='lower right')
P.show()
