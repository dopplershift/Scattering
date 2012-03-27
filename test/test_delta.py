import matplotlib.pyplot as plt
import numpy as np
import scattering
import scipy.constants as consts

def plot_csec(scatterer, d, var, name):
    plt.plot(d / consts.centi, var,
            label='%.1f cm' % (scatterer.wavelength / consts.centi))
    plt.xlabel('Diameter (cm)')
    plt.ylabel(name)

def plot_csecs(d, scatterers):
    for s in scatterers:
        plt.subplot(1,1,1)
        plot_csec(s, d, np.rad2deg(np.unwrap(-np.angle(-s.S_bkwd[0,0].conj() *
            s.S_bkwd[1,1]).squeeze())), 'delta')
        plt.gca().set_ylim(-4, 20)

d = np.linspace(0.01, 0.7, 200).reshape(200, 1) * consts.centi
sband = 3e8 / 2.8e9
cband = 3e8 / 5.4e9
xband = 3e8 / 9.4e9

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
