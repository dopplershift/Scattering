import matplotlib.pyplot as plt
import numpy as np
import scattering, dsd
import scipy.constants as consts

db_factor = 10.0 * np.log10(np.e)
ref_adjust = 180

def plot_csec(scatterer, l, var, name):
    plt.plot(l, var, label = '%.1f cm' % (scatterer.wavelength / consts.centi))
    plt.xlabel('Rain Content (g m^-3)')
    plt.ylabel(name)

def plot_csecs(l, scatterers):
    for s in scatterers:
        plt.subplot(1,2,1)
        plot_csec(s, l, s.get_copolar_cross_correlation(mp), 'rho(1)')
        plt.subplot(1,2,2)
        plot_csec(s, l, np.unwrap(s.get_backscatter_differential_phase(mp)), 'delta')

d = np.linspace(0.01, 2.0, 200).reshape(200, 1) * consts.centi
l = np.linspace(0.01, 25.0, 100).reshape(1,100) / consts.kilo
mp = dsd.mp_from_lwc(d, l)
sband = 0.1
cband = 0.05
xband = 0.0321

temp = 10.0
angleWidth = 10

x_fixed = scattering.scatterer(xband, temp, 'water', diameters=d, shape='oblate')
x_fixed.set_scattering_model('tmatrix')

x_spread = scattering.scatterer(xband, temp, 'water', diameters=d, shape='oblate')
x_spread.angle_width = angleWidth
x_spread.set_scattering_model('tmatrix')

c_fixed = scattering.scatterer(cband, temp, 'water', diameters=d, shape='oblate')
c_fixed.set_scattering_model('tmatrix')

c_spread = scattering.scatterer(cband, temp, 'water', diameters=d, shape='oblate')
c_spread.angle_width = angleWidth
c_spread.set_scattering_model('tmatrix')

s_fixed = scattering.scatterer(sband, temp, 'water', diameters=d, shape='oblate')
s_fixed.set_scattering_model('tmatrix')

s_spread = scattering.scatterer(sband, temp, 'water', diameters=d, shape='oblate')
s_spread.angle_width = angleWidth
s_spread.set_scattering_model('tmatrix')

l = l.squeeze() * consts.kilo
plot_csecs(l, [x_fixed, x_spread, c_fixed, c_spread, s_fixed, s_spread])
#plot_csecs(l, [x_spread])
plt.legend(loc = 'upper left')
plt.show()
