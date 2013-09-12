import matplotlib.pyplot as plt
import numpy as np
import scattering
import scipy.constants as consts

nd = np.logspace(-2, 1, 200)
T = 0.0
wavelengths = np.array([0.86, 3.21, 5.5, 10.0]) * consts.centi
colors = ['r','b','g','c']

plt.figure()
for lam,color in zip(wavelengths,colors):
    d = nd * lam / np.pi
    scat = scattering.scatterer(lam, T, 'water', diameters=d)
    scat.set_scattering_model('tmatrix')
    plt.loglog(nd, scat.sigma_e * 4/(np.pi * d**2), color+'-',
        label='%5.2fcm Tmat' % (lam / consts.centi))
#    scat.set_scattering_model('mie')
#    plt.loglog(nd, scat.sigma_e * 4/(np.pi * d**2), color+'x',
#        label='%5.2fcm Mie'%(lam / consts.centi))

plt.legend(loc='lower right')
plt.xlabel('Normalized Drop Diameter')
plt.ylabel('Normalized Extinction Cross-Section')
plt.xlim(0.01,10.0)
plt.ylim(1.0e-3,10)
plt.show()
