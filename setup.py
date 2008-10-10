from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from os.path import join

flags = ['-W', '-Wall', '-march=opteron', '-O3']

def configuration(parent_package='', top_path=None):
    config = Configuration('scattering', parent_package, top_path,
        author = 'Ryan May',
        author_email = 'rmay31@gmail.com',
        platforms = ['Linux'],
        description = 'Software for simulating weather radar data.',
        url = 'http://weather.ou.edu/~rmay/research.html')
    
    lapack = get_info('lapack_opt')
    sources = ['ampld.lp.pyf', 'ampld.lp.f', 'modified_double_precision_drop.f']
    config.add_extension('_tmatrix', [join('src', f) for f in sources],
        extra_compile_args=flags, **lapack)
    return config

setup(**configuration(top_path='').todict())
