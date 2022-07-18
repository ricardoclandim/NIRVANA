
from IPython import embed

import numpy

from matplotlib import pyplot

from scipy import stats, special
from nirvana.data import manga
from nirvana.data import util
from nirvana.data import scatter
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.models.oned import HyperbolicTangent, Exponential
from nirvana.models.axisym import AxisymmetricDisk
from nirvana.models.bisym import BisymmetricDisk
from nirvana.models.beam import gauss2d_kernel, ConvolveFFTW


def test_disk():
    disk = BisymmetricDisk()
    disk.par[:2] = 0.           # Ensure that the center is at 0,0

    disk.par[2] = 45.       # Position angle
    disk.par[3] = 60.       # Inclination
    disk.par[4] = 0.        # Systemic velocity
    disk.par[5] = 75.       # Position angle of the bisymmetric feature
                            # relative to the major axis position angle

    disk.par[6] = 150.      # *Projected* asymptotic rotation speed
    disk.par[7] = 5.        # Rotation scale
    disk.par[8] = 50.       # *Projected* 2nd order tangential amplitude
    disk.par[9] = 5.       # 2nd order tangential scale
    disk.par[10] = 2.       # 2nd order tangential power
    disk.par[11] =  0.      # *Projected* 2nd order tangential amplitude
    disk.par[12] = 5.      # 2nd order tangential scale
    disk.par[13] = 2.       # 2nd order tangential power

    n = 71
    x = (numpy.arange(n) - n//2).astype(float)[::-1]
    y = (numpy.arange(n) - n//2).astype(float)
    x, y = numpy.meshgrid(x, y)

    vel = disk.model(x=x, y=y)
    beam = gauss2d_kernel(n, 3.)
    _vel = disk.model(x=x, y=y, beam=beam)

    assert numpy.isclose(vel[n//2,n//2], _vel[n//2,n//2]), 'Smearing moved the center.'


@requires_remote
def test_lsq_nopsf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8078, 12703, cube_path=data_root,
                                                 maps_path=data_root, ignore_psf=True)

#    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
#    # Set the disk velocity field
#    axisym_disk = AxisymmetricDisk(rc=rc)
#    # Fit it with a non-linear least-squares optimizer
#    axisym_disk.lsq_fit(kin, verbose=2)

    # Set the disk velocity field
    bisym_disk = BisymmetricDisk()
    # Fit it with a non-linear least-squares optimizer
    bisym_disk.lsq_fit(kin, verbose=2, analytic_jac=False)

#    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
#    assert 165. < disk.par[2] < 167., 'PA changed'
#    assert 53. < disk.par[3] < 55., 'Inclination changed'
#    assert 243. < disk.par[5] < 245., 'Projected rotation changed'


#test_lsq_nopsf()

