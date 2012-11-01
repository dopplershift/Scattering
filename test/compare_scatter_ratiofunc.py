import matplotlib.pyplot as plt
import numpy as np
import scattering, dsd
import scipy.constants as consts

d = np.linspace(0.01, 1.0, 200).reshape(200,1) * consts.centi
l = np.linspace(0.01, 15.0, 100).reshape(1,100) / consts.kilo
dist = dsd.mp_from_lwc(d, l)
#lam = 0.1
lam = 0.0321
temp = 10.0

db_factor = 10.0 * np.log10(np.e)
ref_adjust = 180

rg_brandes = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate', axis_ratio_func=scattering.brandes_axis_ratios)
rg_brandes.set_scattering_model('gans')

rg_prup = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate', axis_ratio_func=scattering.pruppacher_axis_ratios)
rg_prup.set_scattering_model('gans')

tmat_brandes = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate', axis_ratio_func=scattering.brandes_axis_ratios)
tmat_brandes.set_scattering_model('tmatrix')

tmat_prup = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate', axis_ratio_func=scattering.pruppacher_axis_ratios)
tmat_prup.set_scattering_model('tmatrix')

d = d.squeeze() / consts.centi
l = l.squeeze() * consts.kilo

lines = ['ro', 'g-', 'bo', 'k-']
names = ['RG+Brandes', 'RG+Pruppacher', 'Tmatrix+Brandes', 'Tmatrix+Pruppacher']
models = [rg_brandes, rg_prup, tmat_brandes, tmat_prup]

for model, line, name in zip(models, lines, names):
    ref = 10.0 * np.log10(model.get_reflectivity_factor(dist)) + ref_adjust
    refv = (10.0 * np.log10(model.get_reflectivity_factor(dist, polar='v')) +
            ref_adjust)
    atten = model.get_attenuation(dist) * consts.kilo * db_factor
    attenv = model.get_attenuation(dist, polar='v') * consts.kilo * db_factor
    zdr = ref - refv
    diff_atten = atten - attenv

    plt.subplot(2, 2, 1)
    plt.plot(l, zdr, line, label=name)
    plt.subplot(2, 2, 2)
    plt.plot(l, ref, line, label=name)
    plt.subplot(2, 2, 3)
    plt.plot(l, diff_atten, line, label=name)
    plt.subplot(2, 2, 4)
    plt.plot(l, atten, line, label=name)

plt.subplot(2,2,1)
plt.legend(loc = 'lower right')
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel(r'Z$_{DR}$ (dB)')

plt.subplot(2,2,2)
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel(r'Z$_{e}$ (dBZ)')

plt.subplot(2,2,3)
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel('1-way Differential Attenuation (db/km)')

plt.subplot(2,2,4)
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel('1-way Attenuation (db/km)')
plt.gcf().text(0.5,0.95,'Comparison of Various Scattering models',
  horizontalalignment='center',fontsize=16)
plt.show()

