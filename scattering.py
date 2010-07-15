'''
This module contains several useful functions for doing electromagnetic
scattering calculations for liquid water and ice. This code was developed
to facilitate doing radar calculations.
'''
import numpy as np
import scipy.special as ss
from scipy.constants import milli, centi
import _tmatrix as _tmat

__all__ = ['scatterer', 'tmatrix', 'mie', 'rayleigh', 'rayleigh_gans',
    'raindrop_axis_ratios', 'refractive_index', 'ice', 'water']

def refractive_index(material, wavelength, temp=20.0):
    '''
    Calculates the complex refractive index using an expand Debye formula.
    The argument to the function gives another function which will return the
    necessary constants.  Temperature is in Celsius, Wavelength in m.
    '''
    (eps_s, eps_inf, alpha, lam_s, sigma) = _material_dict[material](temp)
    wavelength /= centi
    lam_ratio = (lam_s / wavelength) ** (1 - alpha)
    sin_alpha = np.sin(np.pi * alpha / 2.0)
    denom = 1 + 2 * lam_ratio * sin_alpha + lam_ratio * lam_ratio
    eps_real = eps_inf + (eps_s - eps_inf) * ((1 + lam_ratio * sin_alpha)
        / denom)
    eps_imag = (eps_s - eps_inf) * lam_ratio * (np.cos(np.pi * alpha / 2.0)
        / denom) + sigma * wavelength / 18.8496e10

    return np.sqrt(eps_real + 1.0j * eps_imag)

def raindrop_axis_ratios(d):
    '''
    Calculates the axis ratio for an oblate spheroid approximating a raindrop
    given the (equi-volume) diameter of a spherical drop.  Diameter is in m.
    The original polynomial is documented in Brandes et al. (2002), but the
    coefficients listed don't agree with the graph in Brandes et al. (2004)
    (or with the same coefficients listedi n Brandes et al. (2004). The change
    here is to use 0.005303 (which is taken from lecture notes and assignments
    from a class with G. Zhang, one of the papers' co-authors) instead of the
    published value of 0.005030. The coefficients here yield the correct graph
    (and more sensible values).
    '''
    d = d / milli
    ab = 0.9951 + d * (0.0251 + d * (-0.03644 + d * (0.005303 - 0.0002492 * d)))
    ab[d>8] = ab[d<=8].min()
    return ab

def mie(m, d, lam, shape=None):
    '''
    Computation of Mie Efficiencies for given
    complex refractive-index ratio m=m'+im"
    and diameter and wavelength (which should have the same units), using
    complex Mie Coefficients an and bn for n=1 to nmax, calculated using
    mie_abcd.
    s. Bohren and Huffman (1983) BEWI:TDD122, p. 103,119-122,477.
    C. Matzler, May 2002.
    Returns: Backward and forward scattering matrices (in the same units as
    lambda) and the scattering efficiency.
    '''

    xs = np.pi * d / lam
    if(m.imag < 0):
        m = np.conj(m)

    #Want to make sure we can accept single arguments or arrays
    try:
        xs.size
        xlist = xs
    except:
        xlist = np.array(xs)

    S_frwd = np.zeros((2, 2, d.size), dtype=np.complex64)
    S_bkwd = np.zeros((2, 2, d.size), dtype=np.complex64)
    qsca = np.zeros(xlist.size)
##    qext = np.zeros(xlist.size)
##    qabs = np.zeros(xlist.size)
    for i,x in enumerate(xlist.flat):
        if float(x)==0.0:               # To avoid a singularity at x=0
            qsca[i] = 0.0
##            qext[i] = 0.0
##            qabs[i] = 0.0
##            qb[i] = 0.0
        else:
            an,bn = _mie_abcd(m,x)
            n = np.arange(1, an.size + 1)
            c = 2 * n + 1
            gn = (-1)**n
            x2 = x * x
##            an_mag = np.abs(an)
##            bn_mag = np.abs(bn)
            n_factor = c / 2.0
            qsca[i] = (2.0 / x2) * (c * (np.abs(an)**2 + np.abs(bn)**2)).sum()
##            qext[i] = (2.0 / x2) * (c * (anp.real + bnp.real)).sum()
##            qabs[i] = qext[i] - qsca[i]
            Sf = (1.0j * lam / (2 * np.pi)) * (n_factor * (an + bn)).sum()
            S_frwd[...,i] = np.array([[Sf, 0],[0, Sf]])
            Sb = (-1.0j * lam / (2 * np.pi)) * (n_factor * gn * (an - bn)).sum()
            S_bkwd[...,i] = np.array([[-Sb, 0],[0, Sb]])
##            q = np.abs((c * gn * (an - bn)).sum())
##            qb[i] = q**2 / x2
    return S_frwd, S_bkwd, qsca

def _mie_abcd(m, x):
    '''Computes a matrix of Mie coefficients, a_n, b_n, c_n, d_n,
    of orders n=1 to nmax, complex refractive index m=m'+im",
    and size parameter x=k0*a, where k0= wave number
    in the ambient medium, a=sphere radius;
    p. 100, 477 in Bohren and Huffman (1983) BEWI:TDD122
    C. Matzler, June 2002'''

    nmax = np.round(2+x+4*x**(1.0/3.0))
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

def rayleigh(m, d, lam, shape):
    empty = np.zeros(d.shape, dtype=np.complex64)
    Kw = (m**2 - 1.0)/(m**2 + 2.0)
    S = Kw * np.pi**2 / (2 * lam**2) * d**3
    qsca = (32.0/3.0) * (np.abs(S)/d)**2
    qabs = 4.0 * Kw.imag * np.pi * d / lam
    qext = qsca + qabs
    #Hack here so that extinction cross section can be correctly retrieved from
    #the forward scattering matrix
    S_frwd = S.real + 1.0j * qext * np.pi * d**2 / (8.0 * lam)
    fmat = np.array([[S_frwd, empty], [empty, S_frwd]])
    bmat = np.array([[-S, empty], [empty, S]])
    return fmat, bmat, qsca

def rayleigh_gans(m, d, lam, shape):
    #Get the lambda_z parameter that is a function of the shape of the drop
    if shape == 'sphere':
        lz = 1./3. * np.ones(d.shape)
    elif shape == 'oblate':
        rat = raindrop_axis_ratios(d)
        f2 = rat**-2 - 1
        f = np.sqrt(f2)
        lz = ((1 + f2) / f2) * (1 - (1. / f) * np.arctan(f))
    elif shape == 'prolate':
        #TODO: Need to finish this
        raise NotImplementedError('Prolate not implemented. Yet.')
        lz = (1 - e2) / e2 * (np.log((1 + e) / (1 - e)) / (2 * e)  - 1)
    else:
        raise NotImplementedError('Unimplemented shape: %s' % shape)

    #Calculate the constants outside of the matrix
    eps_r = m**2
    empty = np.zeros(d.shape, dtype=np.complex64)
    l = (1 - lz) / 2.
    Sfact = np.pi**2 * d**3 * (eps_r - 1) / (6. * lam**2)
    polar_h = 1. / ((eps_r - 1) * l + 1)
    polar_v = 1. / ((eps_r - 1) * lz + 1)

    #Calculate a scattering efficiency using the rayleigh approximation
    qsca_h = (32.0/3.0) * (np.abs(Sfact * polar_h) / d)**2
    qabs_h = (4./3.) * np.imag((eps_r - 1) * polar_h) * np.pi * d / lam
    qext_h = qsca_h + qabs_h

    #Calculate a scattering efficiency using the rayleigh approximation
    qsca_v = (32.0/3.0) * (np.abs(Sfact * polar_v) / d)**2
    qabs_v = (4./3.) * np.imag((eps_r - 1) * polar_v) * np.pi * d / lam
    qext_v = qsca_v + qabs_v

    #Get the forward and backward scattering matrices
    fmat = Sfact * np.array([[polar_h, empty],
                            [empty, polar_v]], dtype=np.complex64)

    #Hack here so that extinction cross section can be correctly retrieved from
    #the forward scattering matrix
    fmat[0,0].imag = qext_h * np.pi * d**2 / (8.0 * lam)
    fmat[1,1].imag = qext_v * np.pi * d**2 / (8.0 * lam)

    bmat = Sfact * np.array([[-polar_h, empty],
                            [empty, polar_v]], dtype=np.complex64)

    return fmat, bmat, qsca_h

def tmatrix(m, d, lam, shape):
    equal_volume = 1.0
    d = np.atleast_1d(d)

    #Set up parameters that depend on what shape model we use for the scatterer
    if shape == 'sphere':
        shp_code = -1
        eccen = np.ones(d.shape)
        eccen.fill(1.00000001) #According to Mischenko, using 1.0 can overflow
    elif shape == 'oblate':
        shp_code = -1
        eccen = 1. / raindrop_axis_ratios(d)
    elif shape == 'prolate':
        raise NotImplementedError
        shp_code = -1
    elif shape == 'raindrop':
        shp_code = -3
        eccen = np.ones(d.shape)
    else:
        raise NotImplementedError, 'Unimplemented shape: %s' % shape

    #Initialize arrays
    S_frwd = np.zeros((2, 2, d.size), dtype=np.complex64)
    S_bkwd = np.zeros((2, 2, d.size), dtype=np.complex64)
    qsca = np.zeros(d.shape)

    #Loop over each diameter in the list and perform the T-matrix computation
    #using the wrapped fortran routine by Mischenko.  This gives us the
    #forward and backward scattering matrices, as well as the scattering
    #efficiency
    for i,ds in enumerate(d):
        if ds == 0.:
            continue
        qs,fmat,bmat =_tmat.tmatrix(ds/2.0,equal_volume,lam,m,eccen[i],shp_code)
        sigma_g = (np.pi / 4.0) * ds ** 2
        qsca[i] = qs * (lam ** 2 / (2 * np.pi)) / sigma_g
        S_frwd[...,i] = fmat
        S_bkwd[...,i] = bmat

    return S_frwd, S_bkwd, qsca

def water(temp):
    '''
    Calculate various parameters for the calculation of the dielectric constant
    of liquid water using the extended Debye formula. Temp is in Celsius.
    '''
    eps_s = 78.54*(1.0 - 4.579e-3 * (temp-25.0) + 1.19e-5 * (temp-25.0)**2 \
        - 2.8e-8 * (temp-25.0)**3)
    eps_inf = 5.27137 + 0.0216474*temp + 0.00131198*temp*temp
    alpha = -16.8129/(temp + 273) + 0.0609265
    lam_s = 0.00033836 * np.exp(2513.98/(temp + 273))
    sigma = 12.5664e8
    return eps_s, eps_inf, alpha, lam_s, sigma

def ice(temp):
    '''
    Calculate various parameters for the calculation of the dielectric constant
    of ice using the extended Debye formula. Temp is in Celsius.
    '''
    eps_s = 203.168 + 2.5 * temp + 0.15 * temp**2
    eps_inf = 3.168
    alpha = 0.288 + 0.0052 * temp + 0.00023 * temp**2
    lam_s = 9.990288e-4 * np.exp(13200.0/(1.9869*(temp + 273)))
    sigma = 1.26 * np.exp(-12500.0/(1.9869*(temp + 273)))
    return eps_s, eps_inf, alpha, lam_s, sigma

#Used to lookup functions that specify parameters given the material
_material_dict = dict(water=water, ice=ice)

class scatterer(object):
    #All scattering matrices returned here are in the Forward Scattering
    #Aligned (FSA) convention
    type_map = dict(mie=mie, rayleigh=rayleigh, gans=rayleigh_gans,
      tmatrix=tmatrix)

    def __init__(self, wavelength, temperature, type='water', shape='sphere',
      diameters=None, ref_index=None):
        '''
        Construct a scatterer for *wavelength* and *temperature*.

        Required arguments:
            wavelength : Wavelength for the incident radiation in meters.

            temperature : Temperature for scattering, used to calculate the
                index of refraction. Given in degrees C.

        Optional arguments:
            type : Type of scatterer, either 'water' or 'ice'.
                Default is 'water'.

            shape : Shape of the scatterer. One of:
                ('sphere', 'oblate', 'prolate', 'raindrop'). These represent
                the assumed drop shape. Oblate uses the relationship of
                Brandes et al. (2002). Raindrop uses a distorted ellipsoid
                shape as given by Beard and Chuang 1987. Defaults to sphere.

            diameters : Array (or scalar) of volume equivalent diameters to
                use to calculate scattering, in meters. Defaults to an array
                of 100 diameters linearly spaced between 0 and 0.01 meters.

            ref_index : Used to give an explicit assumed value for the
                refractive index. If none is given, the coeficients of
                Ray (1972) are used to calculate the dielectric constant
                of water or ice, given the *temperature*.
        '''
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
          self.diameters = np.linspace(0, .01, 100) # in meters
        else:
          self.diameters = diameters
        self.x = np.pi * self.diameters / self.wavelength
        self.sigma_g = (np.pi / 4.0) * self.diameters ** 2
        self.model = 'None'

    def set_scattering_model(self, model):
        '''
        Actually performs the scattering calculation, using the scattering model
        given by *model*, which is one of:
            ('tmatrix', 'mie', 'gans', 'rayleigh')
        '''
        try:
            fmat, bmat, qsca = scatterer.type_map[model](self.m, self.diameters,
                self.wavelength, shape=self.shape)

            self.model = model
            self.S_frwd = fmat.reshape((2,2) + self.diameters.shape)
            self.S_bkwd = bmat.reshape((2,2) + self.diameters.shape)

            #Calculate extinction cross-section
            self.sigma_eh = 2 * self.wavelength * self.S_frwd[0,0].imag
            self.sigma_ev = 2 * self.wavelength * self.S_frwd[1,1].imag
            self.sigma_e = self.sigma_eh

            self.sigma_s = qsca.reshape(self.diameters.shape) * self.sigma_g
            self.sigma_a = self.sigma_e - self.sigma_s

            #Calculate back-scatter cross-sectionp. Negative sign on S_bkwd[0,0]
            #accounts for negative in FSA
            self.sigma_bh = 4 * np.pi * np.abs(-self.S_bkwd[0,0])**2
            self.sigma_bv = 4 * np.pi * np.abs(self.S_bkwd[1,1])**2
            self.sigma_bhv = 4 * np.pi * np.abs(-self.S_bkwd[0,1])**2
            self.sigma_b = self.sigma_bh
        except KeyError:
            msg = 'Invalid scattering model: %s\n' % model
            msg += 'Valid choices are: %s' % str(scatterer.type_map.keys())
            raise ValueError(msg)

    def get_reflectivity(self, dsd_weights, polar='h'):
        '''
        Calculates the reflectivity, in m^-1, given the drop size
        distribution, which should be in units of # m^-4. Polar is used
        to specifiy the polarization assumed, which can be 'h' for horizontal,
        'v' for vertical, or 'vh' or 'hv' for cross-polarization calculation.
        '''
        if polar == 'h':
            return np.trapz(self.sigma_bh * dsd_weights, x=self.diameters,
                axis=0)
        elif polar == 'v':
            return np.trapz(self.sigma_bv * dsd_weights, x=self.diameters,
                axis=0)
        elif polar in ('hv', 'vh'):
            return np.trapz(self.sigma_bhv * dsd_weights, x=self.diameters,
                axis=0)
        else:
            raise ValueError('Invalid polarization specified: %s' % polar)

    def get_reflectivity_factor(self, dsd_weights, polar='h'):
        '''
        Calculates the reflectivity factor, in m^3, given the drop size
        distribution, which should be in units of # m^-4. Polar is used
        to specifiy the polarization assumed, which can be 'h' for horizontal,
        'v' for vertical, or 'vh' or 'hv' for cross-polarization calculation.
        '''
        return (self.get_reflectivity(dsd_weights, polar=polar)
            * self.wavelength**4 / (np.pi**5 * 0.93))

    def get_attenuation(self, dsd_weights, polar='h'):
        '''
        Calculates the attenuation factor, in m^-1, given the drop size
        distribution, which should be in units of # m^-4. Polar is used
        to specifiy the polarization assumed, which can be 'h' for horizontal or
        'v' for vertical.
        '''
        if polar == 'h':
            return np.trapz(self.sigma_eh * dsd_weights, x=self.diameters,
                axis=0)
        else:
            return np.trapz(self.sigma_ev * dsd_weights, x=self.diameters,
                axis=0)

    def get_propagation_wavenumber(self, dsd_weights, polar='h'):
        '''
        Calculates the effective propagation wavenumber, in m^-1, given the
        drop size distribution, which should be in units of # m^-4. Polar is
        used to specifiy the polarization assumed, which can be 'h' for
        horizontal or 'v' for vertical.
        '''
        # Phase doesn't multiply by a factor of two, unlike attenuation,
        # because attenuation represents a decrease in power.  Phase and
        # attenuation, are two parts of the effective wavenumber, which is
        # used for the propagation of the electric field (amplitude)
        if polar == 'h':
            return self.wavelength * np.trapz(
                self.S_frwd[0,0].real * dsd_weights, x=self.diameters, axis=0)
        else:
            return self.wavelength * np.trapz(
                self.S_frwd[1,1].real * dsd_weights, x=self.diameters, axis=0)

    def get_backscatter_phase(self, dsd_weights, polar='h'):
        '''
        Calculates the backscatter phase shift, in radians, given the
        drop size distribution, which should be in units of # m^-4. Polar is
        used to specifiy the polarization assumed, which can be 'h' for
        horizontal,'v' for vertical, or 'vh' or 'hv' for cross-polarization
        calculation.
        '''
        if polar == 'h':
            return np.angle(np.trapz(-self.S_bkwd[0,0] * dsd_weights,
                x=self.diameters, axis=0))
        elif polar == 'v':
            return np.angle(np.trapz(self.S_bkwd[1,1] * dsd_weights,
                x=self.diameters, axis=0))
        elif polar in ('hv', 'vh'):
            return np.angle(np.trapz(-self.S_bkwd[0,1] * dsd_weights,
                x=self.diameters, axis=0))
        else:
            raise ValueError('Invalid polarization specified: %s' % polar)
