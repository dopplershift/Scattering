import matplotlib.pyplot as P
import numpy as N
import scattering, dsd
from constants import cm_per_m, g_per_kg, mm_per_m, m_per_km
 
d = N.linspace(0.01, 2.0, 200).reshape(200,1) / cm_per_m
l = N.linspace(0.01, 25.0, 100).reshape(1,100) / g_per_kg
mp = dsd.mp_from_lwc(d, l)
sband = .1 
xband = .0321

T = 10.0

db_factor = 10.0 * N.log10(N.e)
ref_adjust = 180

s_tmat = scattering.scatterer(sband, T, 'water', diameters=d)
s_tmat.set_scattering_model('tmatrix')
s_tmat_ref = 10.0 * N.log10(s_tmat.get_reflectivity_factor(mp)) + ref_adjust
s_tmat_atten = s_tmat.get_attenuation(mp) * m_per_km * db_factor

x_tmat = scattering.scatterer(xband, T, 'water', diameters=d)
x_tmat.set_scattering_model('tmatrix')
x_tmat_ref = 10.0 * N.log10(x_tmat.get_reflectivity_factor(mp)) + ref_adjust
x_tmat_atten = x_tmat.get_attenuation(mp) * m_per_km * db_factor

s_ray = scattering.scatterer(sband, T, 'water', diameters=d)
s_ray.set_scattering_model('rayleigh')
s_ray_ref = 10.0 * N.log10(s_ray.get_reflectivity_factor(mp)) + ref_adjust
s_ray_atten = s_ray.get_attenuation(mp) * m_per_km * db_factor

s_mie = scattering.scatterer(sband, T, 'water', diameters=d)
s_mie.set_scattering_model('mie')
s_mie_ref = 10.0 * N.log10(s_mie.get_reflectivity_factor(mp)) + ref_adjust
s_mie_atten = s_mie.get_attenuation(mp) * m_per_km * db_factor

x_ray = scattering.scatterer(xband, T, 'water', diameters=d)
x_ray.set_scattering_model('rayleigh')
x_ray_ref = 10.0 * N.log10(x_ray.get_reflectivity_factor(mp)) + ref_adjust
x_ray_atten = x_ray.get_attenuation(mp) * m_per_km * db_factor

x_mie = scattering.scatterer(xband, T, 'water', diameters=d)
x_mie.set_scattering_model('mie')
x_mie_ref = 10.0 * N.log10(x_mie.get_reflectivity_factor(mp)) + ref_adjust
x_mie_atten = x_mie.get_attenuation(mp) * m_per_km * db_factor

d = d.squeeze() * cm_per_m
l = l.squeeze() * g_per_kg

P.subplot(2, 2, 1)
P.semilogy(d, s_ray.sigma_b, 'b--', label = 'Rayleigh (10cm)')
P.semilogy(d, s_mie.sigma_b, 'b-', label = 'Mie (10cm)')
P.semilogy(d[::5], s_tmat.sigma_b[::5], 'bx', label = 'T-matrix (10cm)')
P.semilogy(d, x_ray.sigma_b, 'r--', label = 'Rayleigh (3.21cm)')
P.semilogy(d, x_mie.sigma_b, 'r-', label = 'Mie (3.21cm)')
P.semilogy(d[::5], x_tmat.sigma_b[::5], 'rx', label = 'T-matrix (3.21cm)')
P.legend(loc = 'lower right')
P.xlabel('Diameter (cm)')
P.ylabel(r'$\sigma_b \rm{(m^2)}$')

P.subplot(2, 2, 2)
P.plot(l, s_ray_ref, 'b--', label = 'Rayleigh (10cm)')
P.plot(l, s_mie_ref, 'b-', label = 'Mie (10cm)')
P.plot(l[::5], s_tmat_ref[::5], 'bx', label = 'T-matrix (10cm)')
P.plot(l, x_ray_ref, 'r--', label = 'Rayleigh (3.21cm)')
P.plot(l, x_mie_ref, 'r-', label = 'Mie (3.21cm)')
P.plot(l[::5], x_tmat_ref[::5], 'rx', label = 'T-matrix (3.21cm)')
P.xlabel('Rain Content (g/m^3)')
P.ylabel(r'Z$_{e}$ (dBZ)')

P.subplot(2, 2, 3)
P.semilogy(d, s_ray.sigma_e, 'b--', label = 'Rayleigh (10cm)')
P.semilogy(d, s_mie.sigma_e, 'b-', label = 'Mie (10cm)')
P.semilogy(d[::5], s_tmat.sigma_e[::5], 'bx', label = 'T-matrix (10cm)')
P.semilogy(d, x_ray.sigma_e, 'r--', label = 'Rayleigh (3.21cm)')
P.semilogy(d, x_mie.sigma_e, 'r-', label = 'Mie (3.21cm)')
P.semilogy(d[::5], x_tmat.sigma_e[::5], 'rx', label = 'T-matrix (3.21cm)')
P.xlabel('Diameter (cm)')
P.ylabel(r'$\sigma_e \rm{(m^2)}$')

P.subplot(2, 2, 4)
P.plot(l, s_ray_atten, 'b--', label = 'Rayleigh (10cm)')
P.plot(l, s_mie_atten, 'b-', label = 'Mie (10cm)')
P.plot(l[::5], s_tmat_atten[::5], 'bx', label = 'T-matrix (10cm)')
P.plot(l, x_ray_atten, 'r--', label = 'Rayleigh (3.21cm)')
P.plot(l, x_mie_atten, 'r-', label = 'Mie (3.21cm)')
P.plot(l[::5], x_tmat_atten[::5], 'rx', label = 'T-matrix (3.21cm)')
P.xlabel('Rain Content (g/m^3)')
P.ylabel('1-way Attenuation (db/km)')
P.gcf().text(0.5,0.95,'Comparison of Rayleigh and Mie Scattering models',
  horizontalalignment='center',fontsize=16)
P.show()

