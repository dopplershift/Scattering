import matplotlib.pyplot as plt
import numpy as np
import scattering, dsd
import scipy.constants as consts

def plot_csec(scatterer, csec, name):
    plt.semilogy(scatterer.diameters / consts.milli, csec / consts.centi**2,
        label = '%.1f cm (%s)' % (scatterer.wavelength / consts.centi,
        scatterer.model))
    plt.xlabel('Diameter (mm)')
    plt.ylabel(r'$\rm{%s (cm^2)}$' % name)

def plot_csecs(scatterers):
    for s in scatterers:
        plt.subplot(2,2,1)
        plot_csec(s, s.sigma_s, '\sigma_{s}')
        plt.subplot(2,2,2)
        plot_csec(s, s.sigma_a, '\sigma_{a}')
        plt.subplot(2,2,3)
        plot_csec(s, s.sigma_e, '\sigma_{e}')
        plt.subplot(2,2,4)
        plot_csec(s, s.sigma_b, '\sigma_{b}')
  
d = np.linspace(0.01, 2.0, 200).reshape(200,1) * consts.centi
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
plt.legend(loc = 'lower right')
plt.show()
