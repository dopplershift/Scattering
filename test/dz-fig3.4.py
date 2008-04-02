import pylab as P
import numpy as N
import scattering

nd = N.logspace(-2, 1, 200)
T = 0.0
wavelengths = [0.86, 3.21, 5.5, 10.0]
colors = ['r','b','g','c']

P.figure()
for lam,color in zip(wavelengths,colors):
    d = nd * lam / N.pi
    scat = scattering.scatterer(lam, T, 'water', diameters = d)
    scat.set_scattering_model('tmatrix')
    P.loglog(nd, scat.sigma_e * 4/(N.pi * d**2), color+'-', label='%5.2fcm Tmat'%lam)
##    scat.set_scattering_model('mie')
##    P.loglog(nd, scat.sigma_e * 4/(N.pi * d**2), color+'x',
##        label='%5.2fcm Mie'%lam)

P.legend(loc='lower right')
P.xlabel('Normalized Drop Diameter')
P.ylabel('Normalized Extinction Cross-Section')
P.xlim(0.01,10.0)
P.ylim(1.0e-3,10)
P.show()