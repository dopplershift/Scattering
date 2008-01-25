import numpy as N
import tmatrix as tmat
from scattering import refractive_index
m = refractive_index('water',3.0)
ds = 5.0
qs,fmat,bmat= tmat.tmatrix(ds/2.0,1.0,30.0,m,1.0,-1)
