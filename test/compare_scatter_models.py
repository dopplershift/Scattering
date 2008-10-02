import matplotlib.pyplot as P
import numpy as N
import scattering, dsd
from constants import cm_per_m, g_per_kg, mm_per_m, m_per_km
  
d = N.linspace(0.01, 1.0, 200).reshape(200,1) / cm_per_m
l = N.linspace(0.01, 25.0, 100).reshape(1,100) / g_per_kg
dist = dsd.mp_from_lwc(d, l)
#lam = 0.1
lam = 0.0321
temp = 10.0

db_factor = 10.0 * N.log10(N.e)
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

d = d.squeeze() * cm_per_m
l = l.squeeze() * g_per_kg

lines = ['r-', 'g-', 'b-', 'k-', 'k--']
names = ['Rayleigh', 'Rayleigh-Gans (oblate)', 'Rayleigh-Gans (sphere)',
    'Mie', 'T-Matrix (oblate)']
#models = [ray, oblate_rg, sphere_rg, mie, oblate]
models = [ray, oblate_rg, sphere_rg]

for model, line, name in zip(models, lines, names):
    ref = 10.0 * N.log10(model.get_reflectivity_factor(dist)) + ref_adjust
    atten = model.get_attenuation(dist) * m_per_km * db_factor

    P.subplot(2, 2, 1)
    P.semilogy(d, model.sigma_b, line, label=name)
    P.subplot(2, 2, 2)
    P.plot(l, ref, line, label=name)
    P.subplot(2, 2, 3)
    P.semilogy(d, model.sigma_e, line, label=name)
    P.subplot(2, 2, 4)
    P.plot(l, atten, line, label=name)

P.subplot(2,2,1)
P.legend(loc = 'lower right')
P.xlabel('Diameter (cm)')
P.ylabel(r'$\sigma_b \rm{(m^2)}$')
#P.semilogy(d, 4 * N.pi * N.abs(oblate_rg.bmat[1,1].reshape(
#                self.diameters.shape))**2

P.subplot(2,2,2)
P.xlabel('Rain Content (g/m^3)')
P.ylabel(r'Z$_{e}$ (dBZ)')

P.subplot(2,2,3)
P.xlabel('Diameter (cm)')
P.ylabel(r'$\sigma_e \rm{(m^2)}$')

P.subplot(2,2,4)
P.xlabel('Rain Content (g/m^3)')
P.ylabel('1-way Attenuation (db/km)')
P.gcf().text(0.5,0.95,'Comparison of Various Scattering models',
  horizontalalignment='center',fontsize=16)
P.show()

