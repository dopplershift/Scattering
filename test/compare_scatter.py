import matplotlib.pyplot as P
import numpy as N
import scattering, dsd
from constants import cm_per_m, mm_per_m

def plot_csec(scatterer, csec, name):
  P.semilogy(scatterer.diameters * mm_per_m, csec * cm_per_m**2,
    label = '%.1f cm (%s)' % (scatterer.wavelength * cm_per_m, scatterer.model))
  P.xlabel('Diameter (mm)')
  P.ylabel(r'$%s (cm^2)$' % name)

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
  
d = N.linspace(0.01, 2.0, 200).reshape(200,1) / cm_per_m
sband = .1 
xband = .0321

temp = 0.0

x_mie = scattering.scatterer(xband, temp, 'water', diameters=d)
x_mie.set_scattering_model('mie')
x_ray = scattering.scatterer(xband, temp, 'water', diameters=d)
x_ray.set_scattering_model('rayleigh')

s_mie = scattering.scatterer(sband, temp, 'water', diameters=d)
s_mie.set_scattering_model('mie')
s_ray = scattering.scatterer(sband, temp, 'water', diameters=d)
s_ray.set_scattering_model('rayleigh')

plot_csecs([x_mie, x_ray, s_mie, s_ray])
P.legend(loc = 'lower right')
P.show()


