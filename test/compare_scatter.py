import pylab as P
import matplotlib as M
import numpy as N
import scipy.integrate as si
import scattering, dsd
from matplotlib.font_manager import FontProperties

def plot_csec(scatterer, csec, name):
  P.semilogy(scatterer.diameters * 10.0, csec,
    label = '%.1f cm (%s)' % (scatterer.wavelength, scatterer.model))
  P.xlabel('Diameter (mm)')
  P.ylabel(r'$%s (cm^2)$' % name)
  P.legend(loc = 'lower right')

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
  
d = N.linspace(0.01, 2.0, 200)
l = N.linspace(0.01, 25.0, 100) / 1000.0
mp = dsd.mp_from_lwc(d*10.0, l)
sband = 10.0
xband = 3.21

s_tmat = scattering.scatterer(sband, 10.0, 'water', diameters = d)
s_tmat.set_scattering_model('tmatrix')
s_tmat_ref = 10.0 * N.log10(s_tmat.get_reflectivity(mp))
s_tmat_atten = s_tmat.get_attenuation(mp) * (10.0 * N.log10(N.e))

x_tmat = scattering.scatterer(xband, 10.0, 'water', diameters = d)
x_tmat.set_scattering_model('tmatrix')
x_tmat_ref = 10.0 * N.log10(x_tmat.get_reflectivity(mp))
x_tmat_atten = x_tmat.get_attenuation(mp) * (10.0 * N.log10(N.e))

##s_tmatd = scattering.scatterer(sband, 10.0, 'water', diameters = d,
##  shape = 'raindrop')
##s_tmatd.set_scattering_model('tmatrix')
##s_tmat_refd = 10.0 * N.log10(s_tmatd.get_reflectivity(mp))
##s_tmat_attend = s_tmatd.get_attenuation(mp) * (10.0 * N.log10(N.e))

##x_tmatd = scattering.scatterer(xband, 10.0, 'water', diameters = d,
##  shape = 'raindrop')
##x_tmatd.set_scattering_model('tmatrix')
##x_tmat_refd = 10.0 * N.log10(x_tmatd.get_reflectivity(mp))
##x_tmat_attend = x_tmatd.get_attenuation(mp) * (10.0 * N.log10(N.e))

s_ray = scattering.scatterer(sband, 10.0, 'water', diameters = d)
s_ray.set_scattering_model('rayleigh')
s_ray_ref = 10.0 * N.log10(s_ray.get_reflectivity(mp))
s_ray_atten = s_ray.get_attenuation(mp) * (10.0 * N.log10(N.e))
#ref_ray_a = 10.0 * N.log10(720 * dsd.mp_N0 / (dsd.mp_slope_3rd(l) ** 7))

s_mie = scattering.scatterer(sband, 10.0, 'water', diameters = d)
s_mie.set_scattering_model('mie')
s_mie_ref = 10.0 * N.log10(s_mie.get_reflectivity(mp))
s_mie_atten = s_mie.get_attenuation(mp) * (10.0 * N.log10(N.e))

x_ray = scattering.scatterer(xband, 10.0, 'water', diameters = d)
x_ray.set_scattering_model('rayleigh')
x_ray_ref = 10.0 * N.log10(x_ray.get_reflectivity(mp))
x_ray_atten = x_ray.get_attenuation(mp) * (10.0 * N.log10(N.e))

x_mie = scattering.scatterer(xband, 10.0, 'water', diameters = d)
x_mie.set_scattering_model('mie')
x_mie_ref = 10.0 * N.log10(x_mie.get_reflectivity(mp))
x_mie_atten = x_mie.get_attenuation(mp) * (10.0 * N.log10(N.e))

##s_mie_0 = scattering.scatterer(sband, 0.0, 'water', diameters = d)
##s_mie_0.set_scattering_model('mie')
##ref_mie_0 = 10.0 * N.log10(s_mie_0.get_reflectivity(mp))
##atten_mie_0 = s_mie_0.get_attenuation(mp) * (10.0 * N.log10(N.e))
##
##s_mie_20 = scattering.scatterer(sband, 20.0, 'water', diameters = d)
##s_mie_20.set_scattering_model('mie')
##ref_mie_20 = 10.0 * N.log10(s_mie_20.get_reflectivity(mp))
##atten_mie_20 = s_mie_20.get_attenuation(mp) * (10.0 * N.log10(N.e))
##
##s_mie_n20 = scattering.scatterer(sband, -20.0, 'water', diameters = d)
##s_mie_n20.set_scattering_model('mie')
##ref_mie_n20 = 10.0 * N.log10(s_mie_n20.get_reflectivity(mp))
##atten_mie_n20 = s_mie_n20.get_attenuation(mp) * (10.0 * N.log10(N.e))

P.subplot(2, 2, 1)
P.semilogy(d, s_ray.sigma_b, 'b--', label = 'Rayleigh (10cm)')
P.semilogy(d, s_mie.sigma_b, 'b-', label = 'Mie (10cm)')
P.semilogy(d[::5], s_tmat.sigma_b[::5], 'bx', label = 'T-matrix (10cm)')
P.semilogy(d, x_ray.sigma_b, 'r--', label = 'Rayleigh (3.21cm)')
P.semilogy(d, x_mie.sigma_b, 'r-', label = 'Mie (3.21cm)')
P.semilogy(d[::5], x_tmat.sigma_b[::5], 'rx', label = 'T-matrix (3.21cm)')
##P.semilogy(d, s_tmatd.sigma_b, 'g-', label = 'Raindrop (10cm)')
##P.semilogy(d, x_tmatd.sigma_b, 'g--', label = 'Raindrop (3.21cm)')
##P.semilogy(d, s_mie_0.sigma_b, label = 'Mie (0oC)')
##P.semilogy(d, s_mie_20.sigma_b, label = 'Mie (20oC)')
##P.semilogy(d, s_mie_n20.sigma_b, label = 'Mie (-20oC)')
P.legend(loc = 'lower right', prop = FontProperties(size = 'smaller'))
P.xlabel('Diameter (cm)', fontsize = 'smaller')
#P.ylabel('Backscatter Cross-section (cm^2)')
P.ylabel(r'$\sigma_b \rm{(cm^2)}$')
P.setp(P.getp(P.gca(), 'yticklabels'), fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'xticklabels'), fontsize = 'smaller')
#P.title('Backscatter for Mie and Rayleigh Scattering')

P.subplot(2, 2, 2)
P.plot(l, s_ray_ref, 'b--', label = 'Rayleigh (10cm)')
P.plot(l, s_mie_ref, 'b-', label = 'Mie (10cm)')
P.plot(l[::5], s_tmat_ref[::5], 'bx', label = 'T-matrix (10cm)')
P.plot(l, x_ray_ref, 'r--', label = 'Rayleigh (3.21cm)')
P.plot(l, x_mie_ref, 'r-', label = 'Mie (3.21cm)')
P.plot(l[::5], x_tmat_ref[::5], 'rx', label = 'T-matrix (3.21cm)')
##P.plot(l, s_tmat_refd, 'g-', label = 'Raindrop (10cm)')
##P.plot(l, x_tmat_refd, 'g--', label = 'Raindrop (3.21cm)')
P.axis([0, 0.025, 0, 70])
##P.plot(l, ref_mie_0, label = 'Mie (0oC)')
##P.plot(l, ref_mie_20, label = 'Mie (20oC)')
##P.plot(l, ref_mie_n20, label = 'Mie (-20oC)')
##P.plot(l, ref_ray_a, label = 'Analytical MP')
#P.legend(loc = 'lower right')
P.xlabel('Rain Content (kg/m^3)', fontsize = 'smaller')
P.ylabel('Ze (dBZ)', fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'yticklabels'), fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'xticklabels'), fontsize = 'smaller')
#P.title('Reflectivity for Mie and Rayleigh Scattering')

P.subplot(2, 2, 3)
P.semilogy(d, s_ray.sigma_e, 'b--', label = 'Rayleigh (10cm)')
P.semilogy(d, s_mie.sigma_e, 'b-', label = 'Mie (10cm)')
P.semilogy(d[::5], s_tmat.sigma_e[::5], 'bx', label = 'T-matrix (10cm)')
P.semilogy(d, x_ray.sigma_e, 'r--', label = 'Rayleigh (3.21cm)')
P.semilogy(d, x_mie.sigma_e, 'r-', label = 'Mie (3.21cm)')
P.semilogy(d[::5], x_tmat.sigma_e[::5], 'rx', label = 'T-matrix (3.21cm)')
##P.semilogy(d, s_tmatd.sigma_e, 'g-', label = 'Raindrop (10cm)')
##P.semilogy(d, x_tmatd.sigma_e, 'g--', label = 'Raindrop (3.21cm)')
##P.semilogy(d, s_mie_0.sigma_e, label = 'Mie (0oC)')
##P.semilogy(d, s_mie_20.sigma_e, label = 'Mie (20oC)')
##P.semilogy(d, s_mie_n20.sigma_e, label = 'Mie (-20oC)')
#P.legend(loc = 'lower right')
P.xlabel('Diameter (cm)', fontsize = 'smaller')
#P.ylabel('Extinction Cross-section (cm^2)')
P.ylabel(r'$\sigma_e \rm{(cm^2)}$')
P.setp(P.getp(P.gca(), 'yticklabels'), fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'xticklabels'), fontsize = 'smaller')
#P.title('Backscatter for Mie and Rayleigh Scattering')

P.subplot(2, 2, 4)
P.plot(l, s_ray_atten, 'b--', label = 'Rayleigh (10cm)')
P.plot(l, s_mie_atten, 'b-', label = 'Mie (10cm)')
P.plot(l[::5], s_tmat_atten[::5], 'bx', label = 'T-matrix (10cm)')
P.plot(l, x_ray_atten, 'r--', label = 'Rayleigh (3.21cm)')
P.plot(l, x_mie_atten, 'r-', label = 'Mie (3.21cm)')
P.plot(l[::5], x_tmat_atten[::5], 'rx', label = 'T-matrix (3.21cm)')
##P.plot(l, s_tmat_attend, 'g-', label = 'Raindrop (10cm)')
##P.plot(l, x_tmat_attend, 'g--', label = 'Raindrop (3.21cm)')
P.axis([0, 0.025, 0, 25])
##P.plot(l, atten_mie_0, label = 'Mie (0oC)')
##P.plot(l, atten_mie_20, label = 'Mie (20oC)')
##P.plot(l, atten_mie_n20, label = 'Mie (-20oC)')
#P.legend(loc = 'upper left')
P.xlabel('Rain Content (kg/m^3)', fontsize = 'smaller')
P.ylabel('1-way Attenuation (db/km)', fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'yticklabels'), fontsize = 'smaller')
P.setp(P.getp(P.gca(), 'xticklabels'), fontsize = 'smaller')
#P.title('Reflectivity for Mie and Rayleigh Scattering')
#P.figlegend((lsr, lsm, lxr, lxm), ('Rayleigh (10cm)','Mie (10cm)',\
#  'Rayleigh (3.21cm)', 'Mie (3.21cm)'), 'lower left')
P.gcf().text(0.5,0.95,'Comparison of Rayleigh and Mie Scattering models',
  horizontalalignment='center',fontsize=16)
P.show()
#P.savefig('ray_v_mie.png', dpi=150)

##xband = 3.21
##sband = 10.0
##temp = 0.0
##
##
##x_mie = scattering.scatterer(xband, temp, 'water', diameters = d)
##x_mie.set_scattering_model('mie')
##x_ray = scattering.scatterer(xband, temp, 'water', diameters = d)
##x_ray.set_scattering_model('rayleigh')
##
##s_mie = scattering.scatterer(sband, temp, 'water', diameters = d)
##s_mie.set_scattering_model('mie')
##s_ray = scattering.scatterer(sband, temp, 'water', diameters = d)
##s_ray.set_scattering_model('rayleigh')
##
##plot_csecs([x_mie, x_ray, s_mie, s_ray])
##P.show()
