#Various scattering functions
import numpy as N
import scipy.special as ss
import scipy.integrate as si
from scipy.constants import milli, centi
import _tmatrix as _tmat

__all__ = ['ice', 'mie', 'raindrop_axis_ratios', 'rayleigh', 'rayleigh2',
    'rayleigh_gans', 'ref_rs', 'refractive_index', 'refractive_index0',
    'scatterer', 'tmatrix', 'water']

def refractive_index(material, wavelength, temp = 20.0):
    '''Calculates the complex refractive index using an expand Debye formula. 
    The argument to the function gives another function which will return the
    necessary constants.  Temperature is in Celsius, Wavelength in m'''
    (eps_s, eps_inf, alpha, lam_s, sigma) = _material_dict[material](temp)
    wavelength /= centi
    lam_ratio = (lam_s / wavelength) ** (1 - alpha)
    sin_alpha = N.sin(N.pi * alpha / 2.0)
    denom = 1 + 2 * lam_ratio * sin_alpha + lam_ratio * lam_ratio
    eps_real = eps_inf + (eps_s - eps_inf) * ((1 + lam_ratio * sin_alpha)
        / denom)
    eps_imag = (eps_s - eps_inf) * lam_ratio * (N.cos(N.pi * alpha / 2.0)
        / denom) + sigma * wavelength / 18.8496e10

    return N.sqrt(eps_real + 1.0j * eps_imag)

def raindrop_axis_ratios(d):
    '''Calculates the axis ratio for an oblate spheroid approximating a raindrop
    given the (equi-volume) diameter of a spherical drop.  Diameter is in m.
    The original polynomial is documented in Brandes et al. (2002), but the
    coefficients listed don't agree with the graph in Brandes et al. (2004).
    The coefficients here yield the correct graph (and more sensible values)'''
    d = d / milli
    ab = 0.995 + d * (0.0251 + d * (-0.0364 + d * (0.005303 - 0.0002492 * d)))
    ab[d>8] = ab[d<=8].min()
    return ab

def mie(m, d, lam, shape=None):
    '''Computation of Mie Efficiencies for given
    complex refractive-index ratio m=m'+im"
    and diameter and wavelength (which should have the same units), using
    complex Mie Coefficients an and bn for n=1 to nmax, calculated using 
    mie_abcd.
    s. Bohren and Huffman (1983) BEWI:TDD122, p. 103,119-122,477.
    C. Matzler, May 2002.
    Returns: Backward and forward scattering matrices (in the same units as
    lambda) and the scattering efficiency.
    '''

    xs = N.pi * d / lam
    if(m.imag < 0):
        m = N.conj(m)

    #Want to make sure we can accept single arguments or arrays
    try:
        xs.size
        xlist = xs
    except:
        xlist = N.array(xs)

    S_frwd = N.zeros((2, 2, d.size), dtype=N.complex64)
    S_bkwd = N.zeros((2, 2, d.size), dtype=N.complex64)
    qsca = N.zeros(xlist.size)
##    qext = N.zeros(xlist.size)
##    qabs = N.zeros(xlist.size)
    for i,x in enumerate(xlist.flat):
        if float(x)==0.0:               # To avoid a singularity at x=0
            qsca[i] = 0.0
##            qext[i] = 0.0
##            qabs[i] = 0.0
##            qb[i] = 0.0
        else:
            an,bn = _mie_abcd(m,x)
            n = N.arange(1, an.size + 1)
            c = 2 * n + 1
            gn = (-1)**n
            x2 = x * x
##            an_mag = N.abs(an)
##            bn_mag = N.abs(bn)
            n_factor = c / 2.0
            qsca[i] = (2.0 / x2) * (c * (N.abs(an)**2 + N.abs(bn)**2)).sum()
##            qext[i] = (2.0 / x2) * (c * (an.real + bn.real)).sum()
##            qabs[i] = qext[i] - qsca[i]
            Sf = (1.0j * lam / (2 * N.pi)) * (n_factor * (an + bn)).sum()
            S_frwd[...,i] = N.array([[Sf, 0],[0, Sf]])
            Sb = (1.0j * lam / (2 * N.pi)) * (n_factor * gn * (an - bn)).sum()
            S_bkwd[...,i] = N.array([[-Sb, 0],[0, Sb]])
##            q = N.abs((c * gn * (an - bn)).sum())
##            qb[i] = q**2 / x2
    return S_frwd, S_bkwd, qsca

def _mie_abcd(m, x):
    '''Computes a matrix of Mie coefficients, a_n, b_n, c_n, d_n,
    of orders n=1 to nmax, complex refractive index m=m'+im",
    and size parameter x=k0*a, where k0= wave number
    in the ambient medium, a=sphere radius;
    p. 100, 477 in Bohren and Huffman (1983) BEWI:TDD122
    C. Matzler, June 2002'''

    nmax = N.round(2+x+4*x**(1.0/3.0))
    mx = m * x

    # Get the spherical bessel functions of the first (j) and second (y) kind,
    # and their derivatives evaluated at x at order up to nmax
    j_x,jd_x,y_x,yd_x = ss.sph_jnyn(nmax, x)

    # The above function includes the 0 order bessel functions, which aren't used
    j_x = j_x[1:]
    jd_x = jd_x[1:]
    y_x = y_x[1:]
    yd_x = yd_x[1:]

    # Get the spherical Hankel function of the first type (and it's derivative)
    # from the combination of the bessel functions
    h1_x = j_x + 1.0j*y_x
    h1d_x = jd_x + 1.0j*yd_x

    # Get the spherical bessel function of the first kind and it's derivative
    # evaluated at mx
    j_mx,jd_mx = ss.sph_jn(nmax, mx)
    j_mx = j_mx[1:]
    jd_mx = jd_mx[1:]

    #Get primes (d/dx [x*f(x)]) using derivative product rule
    j_xp = j_x + x*jd_x
    j_mxp = j_mx + mx*jd_mx
    h1_xp = h1_x + x*h1d_x

    m2 = m * m
    an = (m2 * j_mx * j_xp - j_x * j_mxp)/(m2 * j_mx * h1_xp - h1_x * j_mxp)
    bn = (j_mx * j_xp - j_x * j_mxp)/(j_mx * h1_xp - h1_x * j_mxp)
##    cn = (j_x * h1_xp - h1_x * j_xp)/(j_mx * h1_xp - h1_x * j_mxp)
##    dn = m * (j_x * h1_xp - h1_x * j_xp)/(m2 * j_mx * h1_xp -h1_x * j_mxp)
    return an, bn

def rayleigh2(m, d, lam, shape):
    x = N.pi * d / lam
    Kw = (m**2 - 1.0)/(m**2 + 2.0)
    qb = 4.0 * abs(Kw)**2 * x ** 4
    qsca = (2.0/3.0) * qb
    qabs = 4.0 * Kw.imag * x
    qext = qsca + qabs
    return qext, qsca, qabs

def rayleigh(m, d, lam, shape):
    empty = N.zeros(d.shape, dtype=N.complex64)
    Kw = (m**2 - 1.0)/(m**2 + 2.0)
    S = Kw * N.pi**2 / (2 * lam**2) * d**3
    qsca = (32.0/3.0) * (N.abs(S)/d)**2
    qabs = 4.0 * Kw.imag * N.pi * d / lam
    qext = qsca + qabs
    #Hack here so that extinction cross section can be correctly retrieved from
    #the forward scattering matrix
    S_frwd = S.real + 1.0j * qext * N.pi * d**2 / (8.0 * lam)
    fmat = N.array([[S_frwd, empty], [empty, S_frwd]])
    bmat = N.array([[S, empty], [empty, -S]])
    return fmat, bmat, qsca

def rayleigh_gans(m, d, lam, shape):
    #Get the lambda_z parameter that is a function of the shape of the drop
    if shape == 'sphere':
        lz = 1./3. * N.ones(d.shape)
    elif shape == 'oblate':
        rat = raindrop_axis_ratios(d)
        f2 = rat**-2 - 1
        f = N.sqrt(f2)
        lz = ((1 + f2) / f2) * (1 - (1. / f) * N.arctan(f))
    elif shape == 'prolate':
        #TODO: Need to finish this
        raise NotImplementedError
        lz = (1 - e2) / e2 * (N.log((1 + e) / (1 - e)) / (2 * e)  - 1)
    else:
        raise NotImplementedError, 'Unimplemented shape: %s' % shape

    #Calculate the constants outside of the matrix
    eps_r = m**2
    empty = N.zeros(d.shape, dtype=N.complex64)
    l = (1 - lz) / 2.
    Sfact = N.pi**2 * d**3 * (eps_r - 1) / (6. * lam**2)
    
    #Calculate a scattering efficiency using the rayleigh approximation
    qsca = (32.0/3.0) * (N.abs(Sfact) / d)**2
    qabs = (4./3.) * N.imag((eps_r - 1) / ((eps_r - 1) * l + 1)) * N.pi * d / lam
    qext = qsca + qabs

    #Hack here so that extinction cross section can be correctly retrieved from
    #the forward scattering matrix
    S_frwd = Sfact.real + 1.0j * qext * N.pi * d**2 / (8.0 * lam)

    #Get the forward and backward scattering matrices
    fmat = Sfact * N.array([[S_frwd, empty],
        [empty, S_frwd]], dtype=N.complex64)
    bmat = Sfact * N.array([[1. / ((eps_r - 1) * l + 1), empty],
        [empty, -1. / ((eps_r - 1) * lz + 1)]], dtype=N.complex64)
    
    
    return fmat, bmat, qsca

def tmatrix(m, d, lam, shape):
    equal_volume = 1.0
    d = N.atleast_1d(d)
    
    #Set up parameters that depend on what shape model we use for the scatterer
    if shape == 'sphere':
        np = -1
        eccen = N.ones(d.shape)
        eccen.fill(1.000001) #According to Mischenko, using 1.0 can overflow
    elif shape == 'oblate':
        np = -1
        eccen = 1. / raindrop_axis_ratios(d)
    elif shape == 'prolate':
        raise NotImplementedError
        np = -1
    elif shape == 'raindrop':
        np = -3
        eccen = N.ones(d.shape)
    else:
        raise NotImplementedError, 'Unimplemented shape: %s' % shape
  
    #Initialize arrays
    S_frwd = N.zeros((2, 2, d.size), dtype=N.complex64)
    S_bkwd = N.zeros((2, 2, d.size), dtype=N.complex64)
    qsca = N.zeros(d.shape)

    #Loop over each diameter in the list and perform the T-matrix computation
    #using the wrapped fortran routine by Mischenko.  This gives us the
    #forward and backward scattering matrices, as well as the scattering
    #efficiency
    for i,ds in enumerate(d):
        if ds == 0:
            continue
        qs,fmat,bmat = _tmat.tmatrix(ds/2.0,equal_volume,lam,m,eccen[i],np)
        sigma_g = (N.pi / 4.0) * ds ** 2
        qsca[i] = qs * (lam ** 2 / (2 * N.pi)) / sigma_g
        S_frwd[...,i] = fmat
        S_bkwd[...,i] = bmat
    
    return S_frwd, S_bkwd, qsca

def refractive_index0(material, wavelength, temp = 20.0):
    material_dict = dict(water=water, ice=ice)
    (eps_s, eps_inf, alpha, lam_s, sigma) = material_dict[material](temp)
    wavelength /= centi

    lam_ratio = lam_s / wavelength
    denom = 1 + lam_ratio * lam_ratio
    eps_real = eps_inf + (eps_s - eps_inf) / denom
    eps_imag = (eps_s - eps_inf) * lam_ratio / denom

    return N.sqrt(eps_real + 1.0j * eps_imag)

def water(temp):
    eps_s = 78.54*(1.0 - 4.579e-3 * (temp-25.0) + 1.19e-5 * (temp-25.0)**2 \
        - 2.8e-8 * (temp-25.0)**3)
    eps_inf = 5.27137 + 0.0216474*temp + 0.00131198*temp*temp
    alpha = -16.8129/(temp + 273) + 0.0609265
    lam_s = 0.00033836 * N.exp(2513.98/(temp + 273))
    sigma = 12.5664e8
    return eps_s, eps_inf, alpha, lam_s, sigma

def ice(temp):
    eps_s = 203.168 + 2.5 * temp + 0.15 * temp**2
    eps_inf = 3.168
    alpha = 0.288 + 0.0052 * temp + 0.00023 * temp**2
    lam_s = 9.990288e-4 * N.exp(13200.0/(1.9869*(temp + 273)))
    sigma = 1.26 * N.exp(-12500.0/(1.9869*(temp + 273)))
    return eps_s, eps_inf, alpha, lam_s, sigma

#Used to lookup functions that specify parameters given the material
_material_dict = dict(water=water, ice=ice)
  
def ref_rs(wavelength, temp):
    eps_inf = 5.5
    eps_0_slope = -0.3759468439
    eps_0_int = 190.4835017

    C_TO_KELVIN = 273.15
    LAMBDA_TEMP_INC = 10.0
    MAX_LAMBDA_INDEX = 4

    delta_lambda_slope = N.array([-.00135,-.00071,-.001418,-.0000261,-.0000261])
    delta_lambda_int = N.array([ .0359, .0224, .0153, .00112, .000859 ])

    lambda_index = N.floor(temp / LAMBDA_TEMP_INC)
    if lambda_index < 0:
        lambda_index = 0
    elif lambda_index > MAX_LAMBDA_INDEX:
        lambda_index = MAX_LAMBDA_INDEX

    delta_lambda = delta_lambda_slope[lambda_index]\
        * (temp - lambda_index * LAMBDA_TEMP_INC)\
        + delta_lambda_int[lambda_index]

    eps_0 = eps_0_slope * (temp + C_TO_KELVIN) + eps_0_int

    lambda_ratio = delta_lambda / wavelength
    eps_denom = 1 + lambda_ratio * lambda_ratio
    eps_real = (eps_0 - eps_inf) / eps_denom + eps_inf
    eps_imag = (eps_inf - eps_0) * lambda_ratio / eps_denom

    return N.sqrt(eps_real - 1.0j*eps_imag)

class scatterer(object):
    type_map = dict(mie=mie, rayleigh=rayleigh, gans=rayleigh_gans,
      tmatrix=tmatrix)
    def __init__(self, wavelength, temperature, type='water', shape='sphere',
      diameters=None, ref_index=None):
        self.wavelength = wavelength
        self.temperature = temperature
        self.type = type
        if ref_index is None:
            self.m = refractive_index(self.type, self.wavelength,
                self.temperature)
        else:
            self.m = ref_index
        self.shape = shape
        if diameters == None:
          self.diameters = N.linspace(0, .01, 100) # in meters
        else:
          self.diameters = diameters
        self.x = N.pi * self.diameters / self.wavelength
        self.sigma_g = (N.pi / 4.0) * self.diameters ** 2
        self.model = 'None'
    def set_scattering_model(self, model):
        try:
            fmat, bmat, qsca = scatterer.type_map[model](self.m, self.diameters,
                self.wavelength, shape=self.shape)
            self.sigma_e = (2 * self.wavelength
                * fmat[0,0].imag).reshape(self.diameters.shape)
            self.sigma_s = qsca.reshape(self.diameters.shape) * self.sigma_g
            self.sigma_a = self.sigma_e - self.sigma_s
            self.sigma_b = 4 * N.pi * N.abs(bmat[0,0].reshape(
                self.diameters.shape))**2
            self.S_frwd = fmat
            self.S_bkwd = bmat
            self.model = model
        except KeyError:
            msg = 'Invalid scattering model: %s\n' % model
            msg += 'Valid choices are: %s' % str(scatterer.type_map.keys())
            raise ValueError(msg)
    def get_reflectivity(self, dsd_weights):
        return si.trapz(self.sigma_b * dsd_weights, x=self.diameters, axis=0)

    def get_reflectivity_factor(self, dsd_weights):
        return self.get_reflectivity(dsd_weights) * self.wavelength**4 / (
            N.pi**5 * 0.93)

    def get_attenuation(self, dsd_weights):
        return si.trapz(self.sigma_e * dsd_weights, x=self.diameters, axis=0)

if __name__ == '__main__':
    import pylab as P
    lam = .1
    print 'm for Water, %.1fcm, and 10 oC: %f'\
        % (lam / centi, refractive_index('water', lam, 10.0))
    T = N.arange(-25.0,25.0,1.0)
    m = refractive_index('water', lam, T)
    m_old = refractive_index0('water', lam, T)
    m_rs = N.array([ref_rs(lam, temp) for temp in T])

    P.subplot(2,2,1)
    P.plot(T, m.real, T, m_old.real, T, m_rs.real)
    P.grid()
    P.xlabel(r'Temperature ($^{o}$C)')
    P.title('Real Part')
    P.legend(["Mod. Form", "Orig. Form", 'RS Form'])

    P.subplot(2,2,2)
    P.plot(T, m.imag, T, m_old.imag, T, m_rs.imag)
    P.grid()
    P.xlabel(r'Temperature ($^{o}$C)')
    P.title('Imaginary Part')
    P.legend(["Mod. Form", "Orig. Form", 'RS Form'], loc = 'upper right')

    P.subplot(2,2,3)
    P.plot(T, abs((m**2-1)/(m**2+2))**2, T, abs((m_old**2-1)/(m_old**2+2))**2,
        T, abs((m_rs**2-1)/(m_rs**2+2))**2)
    P.grid()
    P.xlabel(r'Temperature ($^{o}$C)')
    P.title(r'|Kw|$^2$')
    P.legend(["Mod. Form", "Orig. Form", 'RS Form'])

    P.subplot(2,2,4)
    P.plot(T, ((m**2-1)/(m**2+2)).imag, T, ((m_old**2-1)/(m_old**2+2)).imag,
        T, ((m_rs**2-1)/(m_rs**2+2)).imag)
    P.grid()
    P.xlabel(r'Temperature ($^{o}$C)')
    P.title('Imaginary Part of Kw')
    P.legend(["Mod. Form", "Orig. Form", 'RS Form'], loc = 'upper right')
    P.show()
