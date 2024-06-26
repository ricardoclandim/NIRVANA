
from IPython import embed

import numpy

from scipy import stats, special
from nirvana.data import manga
from nirvana.data import util
from nirvana.data import scatter
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.models.oned import HyperbolicTangent, Exponential
from nirvana.models.axisym import AxisymmetricDisk
from nirvana.models.beam import gauss2d_kernel, ConvolveFFTW


def test_disk():
    disk = AxisymmetricDisk()
    disk.par[:2] = 0.       # Ensure that the center is at 0,0
    disk.par[-1] = 1.       # Put in a quickly rising RC

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    vel = disk.model(x=x, y=y)
    beam = gauss2d_kernel(n, 3.)
    _vel = disk.model(x=x, y=y, beam=beam)

    assert numpy.isclose(vel[n//2,n//2], _vel[n//2,n//2]), 'Smearing moved the center.'


def test_disk_derivative_nosig():
    disk = AxisymmetricDisk()
    
    # Ensure that center is offset from 0,0 because of derivative calculation when r==0.
    disk.par[:2] = 0.1
    # Use a slowly rising rotation curve.  More quickly rising rotation curves
    # show a greater difference between the finite-difference and direct
    # derivative calculations after the convolution.
    disk.par[-1] = 20.

    # Finite difference test steps
    #                 x0      y0      pa     inc    vsys   vinf   hv
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001])

    n = 101
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    v, dv = disk.deriv_model(disk.par, x=x, y=y)
    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x and y
        vp[...,i] = disk.model(par=_p)
    disk._set_par(p)

    fd_dv = (vp - v[...,None])/dp[None,:]
    for i in range(disk.par.size):
        assert numpy.allclose(dv[...,i], fd_dv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different derivative for parameter {i+1}!'

    # Now include the beam-smearing
    beam = gauss2d_kernel(n, 3.)
    try:
        cnvfftw = ConvolveFFTW(beam.shape)
    except:
        cnvfftw = None
    v, dv = disk.deriv_model(disk.par, x=x, y=y, beam=beam, cnvfftw=cnvfftw)
    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x, y, beam, and
        # cnvfftw
        vp[...,i] = disk.model(par=_p)
    disk._set_par(p)

    fd_dv = (vp - v[...,None])/dp[None,:]
    for i in range(disk.par.size):
        assert numpy.allclose(dv[...,i], fd_dv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different derivative for parameter {i+1}!'


def test_disk_derivative():
    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    
    # Ensure that center is offset from 0,0 because of derivative calculation when r==0.
    disk.par[:2] = 0.1
    # Use a slowly rising rotation curve.  More quickly rising rotation curves
    # show a greater difference between the finite-difference and direct
    # derivative calculations after the convolution.
    disk.par[-3] = 20.

    # Finite difference test steps
    #                 x0      y0      pa     inc    vsys   vinf   hv      sig0   hsig
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.001, 0.0001])

    n = 101
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    v, sig, dv, dsig = disk.deriv_model(disk.par, x=x, y=y)
    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    sigp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x and y
        vp[...,i], sigp[...,i] = disk.model(par=_p)
    disk._set_par(p)

    fd_dv = (vp - v[...,None])/dp[None,:]
    fd_dsig = (sigp - sig[...,None])/dp[None,:]
    for i in range(disk.par.size):
        assert numpy.allclose(dv[...,i], fd_dv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different velocity derivative for parameter {i+1}!'
        # The precision is worse for dsig/dx0 and dsig/dy0 at x=y=0.0.  Not sure
        # why.  The larger atol is to account for this.
        assert numpy.allclose(dsig[...,i], fd_dsig[...,i], rtol=0., atol=3e-3), \
                f'Finite difference produced different sigma derivative for parameter {i+1}!'

    # Now include the beam-smearing
    beam = gauss2d_kernel(n, 3.)
    try:
        cnvfftw = ConvolveFFTW(beam.shape)
    except:
        cnvfftw = None
    v, sig, dv, dsig = disk.deriv_model(disk.par, x=x, y=y, beam=beam, cnvfftw=cnvfftw)
    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    sigp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x, y, beam, and
        # cnvfftw
        vp[...,i], sigp[...,i] = disk.model(par=_p)
    disk._set_par(p)

    fd_dv = (vp - v[...,None])/dp[None,:]
    fd_dsig = (sigp - sig[...,None])/dp[None,:]
    for i in range(disk.par.size):
        assert numpy.allclose(dv[...,i], fd_dv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different derivative for parameter {i+1}!'
        # Apparently the convolution smooths out the difference seen in the test above
        assert numpy.allclose(dsig[...,i], fd_dsig[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different sigma derivative for parameter {i+1}!'


@requires_remote
def test_disk_derivative_bin():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    
    # Ensure that center is offset from 0,0 because of derivative calculation when r==0.
    disk.par[:2] = 0.1
    # Use a slowly rising rotation curve.  More quickly rising rotation curves
    # show a greater difference between the finite-difference and direct
    # derivative calculations after the convolution.
    disk.par[-3] = 20.

    # Finite difference test steps
    #                 x0      y0      pa     inc    vsys   vinf   hv      sig0   hsig
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.001, 0.0001])

    # Include the beam-smearing
    try:
        cnvfftw = ConvolveFFTW(kin.spatial_shape)
    except:
        cnvfftw = None
    v, sig, dv, dsig = disk.deriv_model(disk.par, x=kin.grid_x, y=kin.grid_y, #sb=kin.grid_sb,
                                        beam=kin.beam_fft, is_fft=True, cnvfftw=cnvfftw)
    # Now also include the binning
    bv, dbv = kin.deriv_bin(v, dv)
    bsig, dbsig = kin.deriv_bin(sig, dsig)

    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    sigp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    bvp = numpy.empty(bv.shape+(disk.par.size,), dtype=float)
    bsigp = numpy.empty(bv.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x, y, sb, beam,
        # and cnvfftw
        vp[...,i], sigp[...,i] = disk.model(par=_p)
        bvp[...,i] = kin.bin(vp[...,i])
        bsigp[...,i] = kin.bin(sigp[...,i])
    disk._set_par(p)

    fd_dbv = (bvp - bv[...,None])/dp[None,:]
    fd_dbsig = (bsigp - bsig[...,None])/dp[None,:]

    for i in range(disk.par.size):
        assert numpy.allclose(dbv[...,i], fd_dbv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different derivative for parameter {i+1}!'
        # The difference is relatively large (again) for the dispersion data
        assert numpy.allclose(dbsig[...,i], fd_dbsig[...,i], rtol=0., atol=1e-3), \
                f'Finite difference produced different sigma derivative for parameter {i+1}!'


@requires_remote
def test_disk_derivative_bin_moments():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    
    # Ensure that center is offset from 0,0 because of derivative calculation when r==0.
    disk.par[:2] = 0.1
    # Use a slowly rising rotation curve.  More quickly rising rotation curves
    # show a greater difference between the finite-difference and direct
    # derivative calculations after the convolution.
    disk.par[-3] = 20.

    # Finite difference test steps
    #                 x0      y0      pa     inc    vsys   vinf   hv      sig0   hsig
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.001, 0.0001])

    # Include the beam-smearing
    try:
        cnvfftw = ConvolveFFTW(kin.spatial_shape)
    except:
        cnvfftw = None
    v, sig, dv, dsig = disk.deriv_model(disk.par, x=kin.grid_x, y=kin.grid_y, #sb=kin.grid_sb, 
                                        beam=kin.beam_fft, is_fft=True, cnvfftw=cnvfftw)

    _, bv, bsig, _, dbv, dbsig = kin.deriv_bin_moments(None, v, sig, None, dv, dsig)

    vp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    sigp = numpy.empty(v.shape+(disk.par.size,), dtype=float)
    bvp = numpy.empty(bv.shape+(disk.par.size,), dtype=float)
    bsigp = numpy.empty(bv.shape+(disk.par.size,), dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        # These calls to `model` reuse the previously provided x, y, sb, beam,
        # and cnvfftw
        vp[...,i], sigp[...,i] = disk.model(par=_p)
        _, bvp[...,i], bsigp[...,i] = kin.bin_moments(None, vp[...,i], sigp[...,i])
    disk._set_par(p)

    fd_dbv = (bvp - bv[...,None])/dp[None,:]
    fd_dbsig = (bsigp - bsig[...,None])/dp[None,:]

    # TODO: Constrain the test to only test the bins that have multiple spaxels?

    for i in range(disk.par.size):
#        vdiff = numpy.absolute(dbv[...,i]-fd_dbv[...,i])
#        sdiff = numpy.absolute(dbsig[...,i]-fd_dbsig[...,i])
#        print(i, numpy.amax(vdiff), numpy.amin(vdiff), numpy.amax(sdiff), numpy.amin(sdiff))
#        print(i, numpy.amax(vdiff[kin.nspax > 1]), numpy.amin(vdiff[kin.nspax > 1]),
#                 numpy.amax(sdiff[kin.nspax > 1]), numpy.amin(sdiff[kin.nspax > 1]))
#        continue
        assert numpy.allclose(dbv[...,i], fd_dbv[...,i], rtol=0., atol=1e-4), \
                f'Finite difference produced different derivative for parameter {i+1}!'
        # The difference is relatively large (again) for the dispersion data
        assert numpy.allclose(dbsig[...,i], fd_dbsig[...,i], rtol=0., atol=1e-3), \
                f'Finite difference produced different sigma derivative for parameter {i+1}!'


@requires_remote
def test_disk_fit_derivative():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())

    # Set the parameters close to the best-fitting parameters from a previous
    # run
    p0 = numpy.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])
    # Finite difference test steps
    #                 x0      y0      pa     inc    vsys   vinf   hv      sig0   hsig
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.001, 0.0001])

    # Prepare
    disk._init_par(p0, None)
    disk._init_model(None, kin.grid_x, kin.grid_y, kin.grid_sb, kin.beam_fft, True, None, False)
    disk._init_data(kin, None, True, True)
#    disk._fit_prep(kin, p0, None, None, True, True, True, None)
    # Get the method used to generate the figure-of-merit and the jacobian
    fom = disk._get_fom()
    jac = disk._get_jac()
    # Get the fom and the jacobian
    chi = fom(p0)
    dchi = jac(p0)

    # Brute force it
    chip = numpy.empty(dchi.shape, dtype=float)
    p = disk.par.copy()
    for i in range(disk.par.size):
        _p = p.copy()
        _p[i] += dp[i]
        chip[...,i] = fom(_p)
    disk._set_par(p)

    # Compare them
    fd_dchi = (chip - chi[...,None])/dp[None,:]
    for i in range(disk.par.size):
#        diff = numpy.absolute(dchi[...,i]-fd_dchi[...,i])
#        print(i, numpy.amax(diff), numpy.amin(diff))
#        continue
        assert numpy.allclose(dchi[...,i], fd_dchi[...,i], rtol=0., atol=1e-3), \
                f'Finite difference produced different derivative for parameter {i+1}!'


@requires_remote
def test_lsq_nopsf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root, ignore_psf=True)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin) #, verbose=2)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 53. < disk.par[3] < 55., 'Inclination changed'
    assert 243. < disk.par[5] < 245., 'Projected rotation changed'


@requires_remote
def test_lsq_psf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin) #, verbose=2)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 55. < disk.par[3] < 59., 'Inclination changed'
    assert 252. < disk.par[5] < 255., 'Projected rotation changed'


@requires_remote
def test_lsq_with_sig():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the dispersion profile
    dc = Exponential(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc, dc=dc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin, sb_wgt=True) #, verbose=2)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 56. < disk.par[3] < 60., 'Inclination changed'
    assert 250. < disk.par[5] < 253., 'Projected rotation changed'
    assert 27. < disk.par[7] < 37., 'Central velocity dispersion changed'


@requires_remote
def test_lsq_with_covar():
    # NOTE: This only fits the velocity field....

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root, covar=True)

    print('Forcing covariance to be positive definite.')
    kin.vel_covar = util.impose_positive_definite(kin.vel_covar)

    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc) #, dc=dc)
    # Fit it with a non-linear least-squares optimizer
#    import time
#    t = time.perf_counter()
    disk.lsq_fit(kin, sb_wgt=True)
#    print(f'First fit (no covar): {time.perf_counter()-t} s')

    # Rejected based on error-weighted residuals, accounting for intrinsic scatter
    resid = kin.vel - kin.bin(disk.model())
    err = 1/numpy.sqrt(kin.vel_ivar)
    scat = scatter.IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm)
    sig, rej, gpm = scat.iter_fit(fititer=5) #, verbose=2)
    # Check
    assert sig > 8., 'Different intrinsic scatter'
    assert numpy.sum(rej) == 21, 'Different number of pixels were rejected'

    # Refit with new mask, include scatter and covariance
    kin.vel_mask = numpy.logical_not(gpm)
    p0 = disk.par
#    t = time.perf_counter()
    disk.lsq_fit(kin, scatter=sig, sb_wgt=True, p0=p0, ignore_covar=False,
                 assume_posdef_covar=True) #, verbose=2)
#    print(f'Second fit (w/ covar): {time.perf_counter()-t} s')

    # Reject
    resid = kin.vel - kin.bin(disk.model())
    scat = scatter.IntrinsicScatter(resid, covar=kin.vel_covar, gpm=disk.vel_gpm,
                                    assume_posdef_covar=True)
    sig, rej, gpm = scat.iter_fit(fititer=5) #, verbose=2)
    # Check
    assert sig > 5., 'Different intrinsic scatter'
    assert numpy.sum(rej) == 7, 'Different number of pixels were rejected'
    # Model parameters
    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 56. < disk.par[3] < 58., 'Inclination changed'
    assert 249. < disk.par[5] < 252., 'Projected rotation changed'


@requires_remote
def test_mock_noerr():

    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    # Set the parameters close to the best-fitting parameters from a previous
    # run
    p0 = numpy.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    v, s = disk.model(par=p0, x=kin.grid_x, y=kin.grid_y, sb=kin.grid_sb, beam=kin.beam_fft,
                      is_fft=True)

    _, bv, bs = kin.bin_moments(kin.grid_sb, v, s)
    vremap = kin.remap(bv, mask=kin.vel_mask)
    sremap = kin.remap(bs, mask=kin.sig_mask)

    mock_kin = disk.mock_observation(p0, kin=kin)
    mock_vremap = mock_kin.remap('vel')
    mock_sremap = mock_kin.remap(numpy.sqrt(mock_kin.sig_phys2), mask=kin.sig_mask)

    assert numpy.ma.allclose(mock_vremap, vremap), 'Bad mock velocity'
    assert numpy.ma.allclose(mock_sremap, sremap), 'Bad mock dispersion'


@requires_remote
def test_mock_err():

    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    # Set the parameters close to the best-fitting parameters from a previous
    # run
    p0 = numpy.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    v, s = disk.model(par=p0, x=kin.grid_x, y=kin.grid_y, sb=kin.grid_sb, beam=kin.beam_fft,
                      is_fft=True)
    _, bv, bs = kin.bin_moments(kin.grid_sb, v, s)
    vremap = kin.remap(bv, mask=kin.vel_mask)
    sremap = kin.remap(bs, mask=kin.sig_mask)

    rng = numpy.random.default_rng(seed=909)
    mock_kin = disk.mock_observation(p0, kin=kin, add_err=True, rng=rng)
    mock_vremap = mock_kin.remap('vel')
    mock_sremap = mock_kin.remap(numpy.sqrt(mock_kin.sig_phys2), mask=kin.sig_mask)

    assert numpy.ma.std(mock_vremap-vremap) > 5, 'Velocity error changed'
    assert numpy.ma.std(mock_sremap-sremap) > 7, 'Dispersion error changed'


@requires_remote
def test_mock_covar():

    data_root = remote_data_file()
    kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root, covar=True)

    # Set the parameters close to the best-fitting parameters from a previous
    # run
    p0 = numpy.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])

    disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    v, s = disk.model(par=p0, x=kin.grid_x, y=kin.grid_y, sb=kin.grid_sb, beam=kin.beam_fft,
                      is_fft=True)
    vremap = kin.remap(kin.bin(v), mask=kin.vel_mask)
    sremap = kin.remap(kin.bin(s), mask=kin.sig_mask)

    # Fix the seed so that the result is deterministic
    # WARNING: Without this, there were instances where the deviate for the
    # dispersion would be entirely masked!  Need to understand how/why that can
    # happen.
    rng = numpy.random.default_rng(seed=909)
    mock_kin = disk.mock_observation(p0, kin=kin, add_err=True, rng=rng)
    mock_vremap = mock_kin.remap('vel')
    mock_sremap = mock_kin.remap(numpy.sqrt(mock_kin.sig_phys2), mask=kin.sig_mask)

    assert numpy.ma.std(mock_vremap-vremap) > 5, 'Velocity error changed'
    assert numpy.ma.std(mock_sremap-sremap) > 7, 'Dispersion error changed'


@requires_remote
def test_fisher():

    data_root = remote_data_file()
    for use_covar in [False, True]:
        kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                         maps_path=data_root, covar=use_covar)

        # Set the parameters close to the best-fitting parameters from a previous
        # run
        p0 = numpy.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])

        # Get the Fisher Information Matrix
        disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
        fim = disk.fisher_matrix(p0, kin, sb_wgt=True)

        # Use it to compute the correlation matrix
        covar = util.cinv(fim)
        var = numpy.diag(covar)
        rho = covar / numpy.sqrt(var[:,None]*var[None,:])

        # Get the upper triangle of the correlation matrix (without the main
        # diagonal)
        indx = numpy.triu_indices(rho.shape[0], k=1)

        # Get the indices of the parameters with the 4 strongest correlation coefficients
        srt = numpy.argsort(numpy.absolute(rho[indx]))[::-1][:4]

        # Check the result.  The strongest correlations should be between:
        #   (7,8) - The two sigma parameters
        #   (1,4) - The y coordinate and the systemic velocity
        #   (3,5) - The inclination and the asymptotic rotation speed
        #   (5,6) - The two rotation curve parameters
        #   (0,1) - The center coordinates
        for correlated_pair in zip(indx[0][srt], indx[1][srt]):
            assert correlated_pair in [(7,8), (1,4), (3,5), (5,6)], \
                    'Unexpected pair with strong correlation'


