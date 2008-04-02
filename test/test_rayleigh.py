import numpy as N
import matplotlib.pyplot as P
import scattering, dsd

d = N.linspace(0.01, 2.0, 200).reshape(200,1)
l = N.linspace(0.01, 25.0, 100).reshape(1,100)
mp = dsd.mp_from_lwc(d*10.0, l)
sband = 10.0

s_ray = scattering.scatterer(sband, 10.0, 'water', diameters = d)
s_ray.set_scattering_model('rayleigh')
s_ray_ref = 10.0 * N.log10(s_ray.get_reflectivity(mp))
s_ray_atten = s_ray.get_attenuation(mp) * (10.0 * N.log10(N.e))
m = scattering.refractive_index('water', sband, 10.0)
Kw = (m**2 - 1)/(m**2 + 2)
lam = dsd.mp_slope_3rd(l)
s_ray_atten_mp = ((2/3.) * (N.pi**5 * N.abs(Kw)**2 / (sband**4) * 720./lam**7
    * dsd.mp_N0) + 6 * N.pi / (1e6 * sband) * N.imag(Kw) * l).squeeze()

d = d.squeeze()
l = l.squeeze()

f = P.figure()

ax = f.add_subplot(1,2,2)
ax.semilogy(l, s_ray_atten_mp, 'b--')
ax.semilogy(l, s_ray_atten, 'b-')
ax.set_xlabel('LWC (g/kg)')
ax.set_ylabel('Attenuation')
ax.set_title('Comparison of Attenuation')
ax.grid()

P.show()
