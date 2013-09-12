import matplotlib.pyplot as plt
import numpy as np
import scattering, dsd
import scipy.constants as consts
  
d = np.linspace(0.01, 1.0, 200).reshape(200,1) * consts.centi
l = np.linspace(0.01, 25.0, 100).reshape(1,100) / consts.kilo
dist = dsd.mp_from_lwc(d, l)
#lam = 0.1
lam = 0.0321
temp = 10.0

db_factor = 10.0 * np.log10(np.e)
ref_adjust = 180

ray = scattering.scatterer(lam, temp, 'water', diameters=d)
ray.set_scattering_model('rayleigh')

oblate_rg = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate')
oblate_rg.set_scattering_model('gans')

oblate = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate')
oblate.set_scattering_model('tmatrix')

d = d.squeeze() / consts.centi
l = l.squeeze() * consts.kilo

lines = ['r-', 'g-', 'g--', 'b-', 'b--']
names = ['Rayleigh', 'Rayleigh-Gans (H)', 'Rayleigh-Gans (V)',
    'T-Matrix (H)', 'T-Matrix (V)']

sigs = [ray.sigma_b, oblate_rg.sigma_bh, oblate_rg.sigma_bv, oblate.sigma_bh,
    oblate.sigma_bv]
plt.subplot(2, 2, 1)
for i, sig in enumerate(sigs):
    plt.semilogy(d, sig, lines[i], label=names[i])

refs = [ray.get_reflectivity_factor(dist),
    oblate_rg.get_reflectivity_factor(dist, polar='h'),
    oblate_rg.get_reflectivity_factor(dist, polar='v'),
    oblate.get_reflectivity_factor(dist, polar='h'),
    oblate.get_reflectivity_factor(dist, polar='v')]
plt.subplot(2, 2, 2)
for i, ref in enumerate(refs):
    ref = 10.0 * np.log10(ref) + ref_adjust
    plt.plot(l, ref, lines[i], label=names[i])

sigs = [ray.sigma_e, oblate_rg.sigma_eh, oblate_rg.sigma_ev, oblate.sigma_eh,
    oblate.sigma_ev]
plt.subplot(2, 2, 3)
for i, sig in enumerate(sigs):
    plt.semilogy(d, sig, lines[i], label=names[i])

attens = [ray.get_attenuation(dist),
    oblate_rg.get_attenuation(dist, polar='h'),
    oblate_rg.get_attenuation(dist, polar='v'),
    oblate.get_attenuation(dist, polar='h'),
    oblate.get_attenuation(dist, polar='v')]
plt.subplot(2, 2, 4)
for i, atten in enumerate(attens):
    atten = atten * consts.kilo * db_factor
    plt.plot(l, atten, lines[i], label=names[i])

plt.subplot(2,2,1)
plt.legend(loc = 'lower right')
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_b \rm{(m^2)}$')

plt.subplot(2,2,2)
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel(r'Z$_{e}$ (dBZ)')

plt.subplot(2,2,3)
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_s \rm{(m^2)}$')

plt.subplot(2,2,4)
plt.xlabel('Rain Content (g/m^3)')
plt.ylabel('1-way Attenuation (db/km)')

plt.gcf().text(0.5,0.95,'Comparison of Various Scattering models',
  horizontalalignment='center',fontsize=16)
plt.show()

