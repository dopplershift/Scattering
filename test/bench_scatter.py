import numpy as N
from numpy.random import rand
import scipy.integrate as integ
import scipy.interpolate as interp
import scattering, dsd

def plot_csec(scatterer, csec, name):
  P.semilogy(scatterer.diameters * 10.0, csec,
    label = '%.1f cm (%s)' % (scatterer.wavelength, scatterer.model))
  P.xlabel('Diameter (mm)')
  P.ylabel(r'$%s (cm^2)$' % name)
  P.legend(loc = 'lower right')

def plot_csecs(scatterers):
  for s in scatterers:
    P.subplot(2,2,1)
    plot_csec(s, s.sigma_s, '\sigma_{s}')
    P.subplot(2,2,2)
    plot_csec(s, s.sigma_a, '\sigma_{a}')
    P.subplot(2,2,3)
    plot_csec(s, s.sigma_e, '\sigma_{e}')
    P.subplot(2,2,4)
    plot_csec(s, s.sigma_b, '\sigma_{b}')
  
d = N.linspace(0.01, 2.0, 200)
dmm = d*10.
qr = rand(963,963,83)*20.
for q in qr:
    mp = dsd.mp_from_lwc(dmm, q)
    s_tmat = scattering.scatterer(sband, 10.0, 'water', diameters = d)
    s_tmat.set_scattering_model('tmatrix')
    s_tmat_ref = 10.0 * N.log10(s_tmat.get_reflectivity(mp))

