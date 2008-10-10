import matplotlib.pyplot as plt
import numpy as np
import scattering
import scipy.constants as consts

d = np.linspace(0, 100, 500) * consts.milli
T = 0.0
wavelengths = np.array([10.0, 5.5, 3.21]) * consts.centi
lines = ['r--','b:','g-']

m_water = np.array([np.sqrt(80.255+24.313j), np.sqrt(65.476+37.026j),
    np.sqrt(44.593+41.449j)])
m_ice = np.array([np.sqrt(3.16835+0.02492j), np.sqrt(3.16835+0.01068j),
    np.sqrt(3.16835+0.0089j)])

plt.figure()
for mw, mi, lam, line in zip(m_water, m_ice, wavelengths, lines):
    scat = scattering.scatterer(lam, T, 'water', diameters=d, ref_index=mw)
    scat.set_scattering_model('tmatrix')
    plt.subplot(1,2,1)
    plt.semilogy(d / consts.milli, scat.sigma_b / (consts.centi)**2, line,
        label='%5.2fcm Tmat' % (lam / consts.centi))
    scat = scattering.scatterer(lam, T, 'ice', diameters=d, ref_index=mi)
    scat.set_scattering_model('tmatrix')
    plt.subplot(1,2,2)
    plt.semilogy(d / consts.milli, scat.sigma_b / (consts.centi)**2, line,
        label='%5.2fcm Tmat' % (lam / consts.centi))

plt.subplot(1,2,1)
plt.xlabel('Diameter (mm)')
plt.ylabel(r'Backscatter Cross-Section (cm$^{2}$)')
plt.xlim(0,100.0)
plt.ylim(1.0e-2,1e3)

plt.subplot(1,2,2)
plt.xlabel('Diameter (mm)')
plt.ylabel(r'Backscatter Cross-Section (cm$^{2}$)')
plt.xlim(0,100.0)
plt.ylim(1.0e-2,1e3)

plt.legend(loc='lower right')
plt.show()
