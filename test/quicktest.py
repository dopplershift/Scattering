import numpy as np
from scattering import tmatrix, mie, rayleigh, rayleigh_gans
from scattering import refractive_index

lam = .1 #meters
ds = np.array([.001])#meters
m = refractive_index('water', lam)

fmat,bmat,qs = tmatrix(m, ds, lam * 10.0, 'sphere')
print 'T-matrix'
print qs
print fmat
print bmat

fmat,bmat,qs = mie(m, ds, lam * 10.0, 'sphere')
print '\nMie'
print qs
print fmat
print bmat

fmat,bmat,qs = rayleigh(m, ds, lam * 10.0, 'sphere')
print '\nRayleigh'
print qs
print fmat
print bmat

fmat,bmat,qs = rayleigh_gans(m, ds, lam * 10.0, 'sphere')
print '\nRayleigh-Gans'
print qs
print fmat
print bmat

