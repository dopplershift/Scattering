from scattering import scatterer
import numpy as np
from numpy.testing import assert_array_almost_equal, run_module_suite


def test_ones():
    scat = scatterer(.1, 0.0, 'water', diameters=np.array([0.04, 0.05]))
    scat.set_scattering_model('rayleigh')

    assert_array_almost_equal(scat.sigma_b,
                              np.array( [0.01170936,  0.04466766], np.float32 ))


if __name__ == "__main__":
    run_module_suite()
