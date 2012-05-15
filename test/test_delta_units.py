import matplotlib.pyplot as plt
import numpy as np
import scattering
import scipy.constants as consts
import quantities as pq

def plot_csec(scatterer, d, var, name):
    lam = scatterer.wavelength.rescale('cm')
    plt.plot(d, var,
            label='%.1f %s' % (lam, lam.dimensionality))
    plt.xlabel('Diameter (%s)' % d.dimensionality)
    plt.ylabel(name)

def plot_csecs(d, scatterers):
    for s in scatterers:
        plt.subplot(1,1,1)
        plot_csec(s, d, np.rad2deg(np.unwrap(-np.angle(-s.S_bkwd[0,0].conj() *
            s.S_bkwd[1,1]).squeeze())), 'delta')
        plt.gca().set_ylim(-4, 20)

d = np.linspace(0.01, 0.7, 200).reshape(200, 1) * pq.cm
sband = pq.c / (2.8 * pq.GHz)
cband = pq.c / (5.4 * pq.GHz)
xband = pq.c / (9.4 * pq.GHz)

temp = 10.0

x_fixed = scattering.scatterer(xband, temp, 'water', diameters=d, shape='oblate')
x_fixed.set_scattering_model('tmatrix')

c_fixed = scattering.scatterer(cband, temp, 'water', diameters=d, shape='oblate')
c_fixed.set_scattering_model('tmatrix')

s_fixed = scattering.scatterer(sband, temp, 'water', diameters=d, shape='oblate')
s_fixed.set_scattering_model('tmatrix')

plot_csecs(d, [x_fixed, c_fixed, s_fixed])
plt.legend(loc = 'upper left')
plt.show()
