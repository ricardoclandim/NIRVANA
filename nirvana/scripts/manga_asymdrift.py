"""
Script that simultaneously fits the ionized gas and stellar kinematic data to
measure asymmetric drift.
"""
import os
import argparse

from IPython import embed

import numpy as np
from matplotlib import pyplot

from astropy.io import fits

from ..data import manga
from ..util import fileio
from ..models import axisym
from ..models import multitrace

# TODO: Setup a logger

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plate', type=int, help='MaNGA plate identifier (e.g., 8138)')
    parser.add_argument('ifu', type=int, help='MaNGA ifu identifier (e.g., 12704)')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-11', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the directory to the MaNGA galaxy data (see also '
                             '--redux and --analysis), and it will be ignored if the root '
                             'directory is set directly (using --root).')
    parser.add_argument('--redux', default=None, type=str,
                        help='Top-level directory with the MaNGA DRP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_REDUX.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.')
    parser.add_argument('--root', default=None, type=str,
                        help='Path with *all* fits files required for the fit.  This includes ' \
                             'the DRPall file, the DRP LOGCUBE file, and the DAP MAPS file.  ' \
                             'The LOGCUBE file is only required if the beam-smearing is ' \
                             'included in the fit.')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity level.  0=only status output written to terminal; 1=show '
                             'fit result QA plot; 2=full output.')
    parser.add_argument('--odir', type=str, default=os.getcwd(), help='Directory for output files')
    parser.add_argument('--nodisp', dest='disp', default=True, action='store_false',
                        help='Only fit the velocity field (ignore velocity dispersion)')
    parser.add_argument('--nopsf', dest='smear', default=True, action='store_false',
                        help='Ignore the map PSF (i.e., ignore beam-smearing)')
    parser.add_argument('--covar', default=False, action='store_true',
                        help='Include the nominal covariance in the fit')
    parser.add_argument('--fix_cen', default=False, action='store_true',
                        help='Fix the dynamical center coordinate to the galaxy center')
    parser.add_argument('--fix_inc', default=False, action='store_true',
                        help='Fix the inclination to the guess inclination based on the '
                             'photometric ellipticity')
    parser.add_argument('--low_inc', default=None, type=float,
                        help='Best-fitting inclinations below this value are considered fitting '
                             'errors.  A flag is tripped indicating that the fit resulted in a '
                             'low inclination, but the actual returned fit fixes the inclination '
                             'to the photometric estimate (i.e., as if setting --fix_inc).  If '
                             'None, no lower limit is set.')
    parser.add_argument('--no_scatter', dest='fit_scatter', default=True, action='store_false',
                        help='Do not include any intrinsic scatter terms during the model fit.')
    parser.add_argument('--min_unmasked', default=None, type=int,
                        help='Minimum number of unmasked spaxels required to continue fit.')
    parser.add_argument('--sb_fill_sig', default=None, type=float,
                        help='Fill the surface-brightness map used for constructing the '
                             'beam-smeared model using an iterative Gaussian smoothing '
                             'operation.  This parameter both turns this on and sets the size of '
                             'the circular Gaussian smoothing kernel in spaxels.')
    parser.add_argument('--coherent', default=False, action='store_true',
                        help='After the initial rejection of S/N and error limits, find the '
                             'largest coherent region of adjacent spaxels and only fit that '
                             'region.')
    parser.add_argument('--screen', default=False, action='store_true',
                        help='Indicate that the script is being run behind a screen (used to set '
                             'matplotlib backend).')
    parser.add_argument('--skip_plots', default=False, action='store_true',
                        help='Skip the QA plots and just produce the main output file.')

    parser.add_argument('--gas_rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
    parser.add_argument('--gas_dc', default='Exponential', type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.')
    parser.add_argument('--gas_min_vel_snr', default=None, type=float,
                        help='Minimum S/N to include for velocity measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--gas_min_sig_snr', default=None, type=float,
                        help='Minimum S/N to include for dispersion measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--gas_max_vel_err', default=None, type=float,
                        help='Maximum velocity error to include in fit.')
    parser.add_argument('--gas_max_sig_err', default=None, type=float,
                        help='Maximum velocity dispersion error to include in fit '
                             '(ignored if dispersion not being fit).')
    parser.add_argument('--gas_max_flux', default=None, type=float,
                        help='Maximum flux to include in the surface-brightness weighting.')
    parser.add_argument('--gas_vel_rej', nargs='*', default=[15,10,10,10],
                        help='The velocity rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')
    parser.add_argument('--gas_sig_rej', nargs='*', default=[15,10,10,10],
                        help='The dispersion rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')

    parser.add_argument('--str_rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
    parser.add_argument('--str_dc', default='Exponential', type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.')
    parser.add_argument('--str_min_vel_snr', default=None, type=float,
                        help='Minimum S/N to include for velocity measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--str_min_sig_snr', default=None, type=float,
                        help='Minimum S/N to include for dispersion measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--str_max_vel_err', default=None, type=float,
                        help='Maximum velocity error to include in fit.')
    parser.add_argument('--str_max_sig_err', default=None, type=float,
                        help='Maximum velocity dispersion error to include in fit '
                             '(ignored if dispersion not being fit).')
#    parser.add_argument('--str_max_flux', default=None, type=float,
#                        help='Maximum flux to include in the surface-brightness weighting.')
    parser.add_argument('--str_vel_rej', nargs='*', default=[15,10,10,10],
                        help='The velocity rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')
    parser.add_argument('--str_sig_rej', nargs='*', default=[15,10,10,10],
                        help='The dispersion rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')
    # TODO: Other options:
    #   - Fit with least-squares vs. dynesty vs. emcee
    #   - Toggle the surface-brightness weighting

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    # Running the script behind a screen, so switch the matplotlib backend
    if args.screen:
        pyplot.switch_backend('agg')

    #---------------------------------------------------------------------------
    # Setup
    #  - Check that the output directory exists, and if not create it
    if not os.path.isdir(args.odir):
        os.makedirs(args.odir)
    #  - Set the output root name
    oroot = f'nirvana-manga-asymdrift-{args.plate}-{args.ifu}'

    #---------------------------------------------------------------------------
    # Read the metadata and the kinematic data
    galmeta = manga.MaNGAGlobalPar(args.plate, args.ifu, redux_path=args.redux, dr=args.dr,
                                   drpall_path=args.root)
    gas_kin = manga.MaNGAGasKinematics.from_plateifu(args.plate, args.ifu, daptype=args.daptype,
                                                     dr=args.dr, redux_path=args.redux,
                                                     cube_path=args.root, image_path=args.root,
                                                     analysis_path=args.analysis,
                                                     maps_path=args.root,
                                                     ignore_psf=not args.smear, covar=args.covar,
                                                     positive_definite=True,
                                                     flux_bound=(None, args.gas_max_flux),
                                                     sb_fill=args.sb_fill_sig)
    str_kin = manga.MaNGAStellarKinematics.from_plateifu(args.plate, args.ifu,
                                                         daptype=args.daptype, dr=args.dr,
                                                         redux_path=args.redux,
                                                         cube_path=args.root, image_path=args.root,
                                                         analysis_path=args.analysis,
                                                         maps_path=args.root,
                                                         ignore_psf=not args.smear,
                                                         covar=args.covar, positive_definite=True,
                                                         sb_fill=args.sb_fill_sig)

    #---------------------------------------------------------------------------
    # Run the iterative fits separately for the gas and stars
    gas_disk, gas_p0, gas_fix, gas_vel_mask, gas_sig_mask \
            = axisym.axisym_iter_fit(galmeta, gas_kin, rctype=args.gas_rc, dctype=args.gas_dc,
                                     fitdisp=args.disp, ignore_covar=not args.covar,
                                     max_vel_err=args.gas_max_vel_err,
                                     max_sig_err=args.gas_max_sig_err,
                                     min_vel_snr=args.gas_min_vel_snr,
                                     min_sig_snr=args.gas_min_sig_snr,
                                     vel_sigma_rej=args.gas_vel_rej,
                                     sig_sigma_rej=args.gas_sig_rej,
                                     fix_cen=args.fix_cen, fix_inc=args.fix_inc,
                                     low_inc=args.low_inc, min_unmasked=args.min_unmasked,
                                     select_coherent=args.coherent, fit_scatter=args.fit_scatter,
                                     verbose=args.verbose)
    gas_only_par = gas_disk.par.copy()

    str_disk, str_p0, str_fix, str_vel_mask, str_sig_mask \
            = axisym.axisym_iter_fit(galmeta, str_kin, rctype=args.str_rc, dctype=args.str_dc,
                                     fitdisp=args.disp, ignore_covar=not args.covar,
                                     max_vel_err=args.str_max_vel_err,
                                     max_sig_err=args.str_max_sig_err,
                                     min_vel_snr=args.str_min_vel_snr,
                                     min_sig_snr=args.str_min_sig_snr,
                                     vel_sigma_rej=args.str_vel_rej,
                                     sig_sigma_rej=args.str_sig_rej,
                                     fix_cen=args.fix_cen, fix_inc=args.fix_inc,
                                     low_inc=args.low_inc, min_unmasked=args.min_unmasked,
                                     select_coherent=args.coherent, fit_scatter=args.fit_scatter,
                                     verbose=args.verbose)
    str_only_par = str_disk.par.copy()

    #---------------------------------------------------------------------------
    # Run the combined fit using the independent fits as a starting point
    disk, p0, fix, gas_vel_mask, gas_sig_mask, str_vel_mask, str_sig_mask \
            = multitrace.asymdrift_iter_fit(galmeta, gas_kin, str_kin, gas_disk, str_disk, 
                                            gas_vel_mask=gas_vel_mask, gas_sig_mask=gas_sig_mask,
                                            str_vel_mask=str_vel_mask, str_sig_mask=str_sig_mask,
                                            ignore_covar=not args.covar, fix_cen=args.fix_cen,
                                            fix_inc=args.fix_inc, low_inc=args.low_inc,
                                            min_unmasked=args.min_unmasked,
                                            fit_scatter=args.fit_scatter,
                                            verbose=args.verbose)

    # Create the output data file
    # - Ensure the best-fitting parameters have been distributed to the disks
    disk.distribute_par()
    # - Get the output data for the gas
    gas_slice = disk.disk_slice(0)
    gas_hdu = axisym.axisym_fit_data(galmeta, gas_kin, p0[gas_slice], gas_disk,
                                     gas_vel_mask, gas_sig_mask)
    # - Get the output data for the stars
    str_slice = disk.disk_slice(1)
    str_hdu = axisym.axisym_fit_data(galmeta, str_kin, p0[str_slice], str_disk,
                                     str_vel_mask, str_sig_mask)
    # - Combine the data into a single fits file
    prihdr = gas_hdu[0].header.copy()
    prihdr.remove('MODELTYP')
    prihdr.remove('RCMODEL')
    prihdr.remove('DCMODEL')
    prihdr['QUAL'] = disk.global_mask
    resid = disk.fom(disk.par)
    prihdr['CHI2'] = (np.sum(resid**2), 'Total chi-square')
    prihdr['RCHI2'] = (prihdr['CHI2']/(resid.size - disk.nfree), 'Total reduced chi-square')
    for h in gas_hdu[1:]:
        h.name = 'GAS_'+h.name
    for h in str_hdu[1:]:
        h.name = 'STR_'+h.name
    ofile = os.path.join(args.odir, f'{oroot}.fits.gz')
    _ofile = ofile[:ofile.rfind('.')]
    fits.HDUList([fits.PrimaryHDU(header=prihdr)] + gas_hdu[1:] + str_hdu[1:]
                 ).writeto(_ofile, overwrite=True, checksum=True)
    fileio.compress_file(_ofile, overwrite=True)
    os.remove(_ofile)

    if args.skip_plots:
        return

    # Create the QA plots
    # - Get the gas fit residuals
    dv_plot = os.path.join(args.odir, f'{oroot}-Gas-vdist.png')
    ds_plot = os.path.join(args.odir, f'{oroot}-Gas-sdist.png')
    gas_disk.reject(vel_plot=dv_plot, sig_plot=ds_plot, plots_only=True) 
    # - Get the gas asymmetry plots
    sig_mask=None if gas_disk.dc is None else np.logical_not(gas_disk.sig_gpm)
    asym_plot = os.path.join(args.odir, f'{oroot}-Gas-asym.png')
    gas_kin.asymmetry_plot(galmeta=galmeta, xc=gas_disk.par[0], yc=gas_disk.par[1],
                           pa=gas_disk.par[2], vsys=gas_disk.par[4], fwhm=galmeta.psf_fwhm[1], 
                           vel_mask=np.logical_not(gas_disk.vel_gpm), sig_mask=sig_mask,
                           ofile=asym_plot)
    # - Make the gas fit plots
    fit_plot = os.path.join(args.odir, f'{oroot}-Gas-fit.png')
    axisym.axisym_fit_plot(galmeta, gas_kin, gas_disk, fix=fix[gas_slice], ofile=fit_plot)

    # - Get the stellar fit residuals
    dv_plot = os.path.join(args.odir, f'{oroot}-Stars-vdist.png')
    ds_plot = os.path.join(args.odir, f'{oroot}-Stars-sdist.png')
    str_disk.reject(vel_plot=dv_plot, sig_plot=ds_plot, plots_only=True) 
    # - Get the stellar asymmetry plots
    sig_mask=None if str_disk.dc is None else np.logical_not(str_disk.sig_gpm)
    asym_plot = os.path.join(args.odir, f'{oroot}-Stars-asym.png')
    str_kin.asymmetry_plot(galmeta=galmeta, xc=str_disk.par[0], yc=str_disk.par[1],
                           pa=str_disk.par[2], vsys=str_disk.par[4], fwhm=galmeta.psf_fwhm[1], 
                           vel_mask=np.logical_not(str_disk.vel_gpm), sig_mask=sig_mask,
                           ofile=asym_plot)
    # - Make the stellar fit plots
    fit_plot = os.path.join(args.odir, f'{oroot}-Stars-fit.png')
    axisym.axisym_fit_plot(galmeta, str_kin, str_disk, fix=fix[str_slice], ofile=fit_plot)
    # - Get the consolidated asymmetric drift plot
    # Create the final fit plots
    fit_plot = os.path.join(args.odir, f'{oroot}-asymdrift.png')
    multitrace.asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, ofile=fit_plot)


