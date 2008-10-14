from numpy.distutils.core import setup
import os.path, sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('scattering', parent_package, top_path,
        version='0.8',
        author = 'Ryan May',
        author_email = 'rmay31@gmail.com',
        platforms = ['Linux'],
        description = 'Software for simulating hydrometeor scattering.',
        url = 'http://weather.ou.edu/~rmay/research.html')
    config.add_sconscript('src/SConstruct')
    return config

setup(**configuration(top_path='').todict())

