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

mie = scattering.scatterer(lam, temp, 'water', diameters=d)
mie.set_scattering_model('mie')

ray = scattering.scatterer(lam, temp, 'water', diameters=d)
ray.set_scattering_model('rayleigh')

oblate_rg = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate')
oblate_rg.set_scattering_model('gans')

sphere_rg = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='sphere')
sphere_rg.set_scattering_model('gans')

oblate = scattering.scatterer(lam, temp, 'water', diameters=d,
    shape='oblate')
oblate.set_scattering_model('tmatrix')

d = d.squeeze() / consts.centi
l = l.squeeze() * consts.kilo

lines = ['r-', 'g-', 'b-', 'k-', 'k--']
names = ['Rayleigh', 'Rayleigh-Gans (oblate)', 'Rayleigh-Gans (sphere)',
    'Mie', 'T-Matrix (oblate)']
#models = [ray, oblate_rg, sphere_rg, mie, oblate]
models = [ray, oblate_rg, sphere_rg]

for model, line, name in zip(models, lines, names):
#    ref = 10.0 * np.log10(model.get_reflectivity_factor(dist)) + ref_adjust
#    atten = model.get_attenuation(dist) * consts.kilo * db_factor

    plt.subplot(2, 2, 1)
    plt.semilogy(d, model.sigma_b, line, label=name)
    plt.subplot(2, 2, 2)
    plt.semilogy(d, model.sigma_a, line, label=name)
#    plt.plot(l, ref, line, label=name)
    plt.subplot(2, 2, 3)
    plt.semilogy(d, model.sigma_s, line, label=name)
    plt.subplot(2, 2, 4)
    plt.semilogy(d, model.sigma_e, line, label=name)
#    plt.plot(l, atten, line, label=name)

plt.subplot(2,2,1)
plt.legend(loc = 'lower right')
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_b \rm{(m^2)}$')
#plt.semilogy(d, 4 * np.pi * np.abs(oblate_rg.bmat[1,1].reshape(
#                self.diameters.shape))**2

plt.subplot(2,2,2)
#plt.xlabel('Rain Content (g/m^3)')
#plt.ylabel(r'Z$_{e}$ (dBZ)')
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_a \rm{(m^2)}$')

plt.subplot(2,2,3)
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_s \rm{(m^2)}$')

plt.subplot(2,2,4)
#plt.xlabel('Rain Content (g/m^3)')
#plt.ylabel('1-way Attenuation (db/km)')
plt.xlabel('Diameter (cm)')
plt.ylabel(r'$\sigma_e \rm{(m^2)}$')
plt.gcf().text(0.5,0.95,'Comparison of Various Scattering models',
  horizontalalignment='center',fontsize=16)
plt.show()

