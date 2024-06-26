"""
Script that runs the axisymmetric, least-squares fit for MaNGA data.
"""
import os
import argparse
import copy

from IPython import embed

import numpy as np
from matplotlib import pyplot

from ..data import manga
from ..models import axisym, oned
 


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
    parser.add_argument('-t', '--tracer', default='Gas', type=str,
                        help='The tracer to fit; must be either Gas or Stars.')
    parser.add_argument('--rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
#    parser.add_argument('--rc_bounds', default=None, nargs='*', type=str,
#                        help='Lower and upper bounds to use for each rotation curve parameter.  '
#                             'Function defaults are used if none provided by the user.  The '
#                             'number of provided bounds *must* be twice the number of '
#                             'parameters.  The expected order is: lower bound parameter 1, '
#                             'upper bound parameter 1, lower bound parameter 2, etc.  Any of the '
#                             'bounds can be set to None to indicate that the code should use the '
#                             'function default for that bound.')
    parser.add_argument('--dc', default='Exponential', type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.')
#    parser.add_argument('--dc_bounds', default=None, nargs='*', type=str,
#                        help='Lower and upper bounds to use for each dispersion profile '
#                             'parameter.  Function defaults are used if none provided by the '
#                             'user.  The number of provided bounds *must* be twice the number of '
#                             'parameters.  The expected order is: lower bound parameter 1, '
#                             'upper bound parameter 1, lower bound parameter 2, etc.  Any of the '
#                             'bounds can be set to None to indicate that the code should use the '
#                             'function default for that bound.')
    parser.add_argument('--min_vel_snr', default=None, type=float,
                        help='Minimum S/N to include for velocity measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--min_sig_snr', default=None, type=float,
                        help='Minimum S/N to include for dispersion measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--max_vel_err', default=None, type=float,
                        help='Maximum velocity error to include in fit.')
    parser.add_argument('--max_sig_err', default=None, type=float,
                        help='Maximum velocity dispersion error to include in fit '
                             '(ignored if dispersion not being fit).')
    parser.add_argument('--min_unmasked', default=None, type=int,
                        help='Minimum number of unmasked spaxels required to continue fit.')
    parser.add_argument('--max_flux', default=None, type=float,
                        help='Maximum flux to include in the surface-brightness weighting.')
    parser.add_argument('--sb_fill_sig', default=None, type=float,
                        help='Fill the surface-brightness map used for constructing the '
                             'beam-smeared model using an iterative Gaussian smoothing '
                             'operation.  This parameter both turns this on and sets the size of '
                             'the circular Gaussian smoothing kernel in spaxels.')
    parser.add_argument('--no_scatter', dest='fit_scatter', default=True, action='store_false',
                        help='Do not include any intrinsic scatter terms during the model fit.')
    parser.add_argument('--vel_rej', nargs='*', default=[15,10,10,10],
                        help='The velocity rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')
    parser.add_argument('--sig_rej', nargs='*', default=[15,10,10,10],
                        help='The dispersion rejection sigma.  Provide 1 or 4 numbers.  If 1 '
                             'number provided, the same sigma threshold is used for all fit '
                             'iterations.')
    parser.add_argument('--coherent', default=False, action='store_true',
                        help='After the initial rejection of S/N and error limits, find the '
                             'largest coherent region of adjacent spaxels and only fit that '
                             'region.')
    parser.add_argument('--screen', default=False, action='store_true',
                        help='Indicate that the script is being run behind a screen (used to set '
                             'matplotlib backend).')
    parser.add_argument('--skip_plots', default=False, action='store_true',
                        help='Skip the QA plots and just produce the main output file.')
    parser.add_argument('--save_vrot', default=False, action='store_true',
                        help='Save projected rotation velocity, dispersion velocity and uncertainties from data and parameters')
    parser.add_argument('--fisher', default=None, action='store_true',
                        help='Calculate covariance matrix of the estimated parameters')
                                            

    # TODO: Other options:
    #   - Fit with least-squares vs. dynesty
    #   - Type of rotation curve
    #   - Type of dispersion profile
    #   - Include the surface-brightness weighting

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    # Running the script behind a screen, so switch the matplotlib backend
    if args.screen:
        pyplot.switch_backend('agg')

    #---------------------------------------------------------------------------
    # Setup
    #  - Check the input
    if args.tracer not in ['Gas', 'Stars']:
        raise ValueError('Tracer to fit must be either Gas or Stars.')
    #  - Check that the output directory exists, and if not create it
    if not os.path.isdir(args.odir):
        os.makedirs(args.odir)
    #  - Set the output root name
    oroot = f'nirvana-manga-axisym-{args.plate}-{args.ifu}-{args.tracer}-{args.rc}-{args.dc}-{args.daptype}'

    flux_bound = (None, args.max_flux)

    #---------------------------------------------------------------------------
    # Read the data to fit
    if args.tracer == 'Gas':
        kin = manga.MaNGAGasKinematics.from_plateifu(args.plate, args.ifu, daptype=args.daptype,
                                                     dr=args.dr, redux_path=args.redux,
                                                     cube_path=args.root, image_path=args.root,
                                                     analysis_path=args.analysis,
                                                     maps_path=args.root,
                                                     ignore_psf=not args.smear, covar=args.covar,
                                                     positive_definite=True, flux_bound=flux_bound,
                                                     sb_fill=args.sb_fill_sig)
    elif args.tracer == 'Stars':
        kin = manga.MaNGAStellarKinematics.from_plateifu(args.plate, args.ifu,
                                                         daptype=args.daptype, dr=args.dr,
                                                         redux_path=args.redux,
                                                         cube_path=args.root, image_path=args.root,
                                                         analysis_path=args.analysis,
                                                         maps_path=args.root,
                                                         ignore_psf=not args.smear,
                                                         covar=args.covar, positive_definite=True,
                                                         sb_fill=args.sb_fill_sig)
    else:
        # NOTE: Should never get here given the check above.
        raise ValueError(f'Unknown tracer: {args.tracer}')

    # Setup the metadata
    galmeta = manga.MaNGAGlobalPar(args.plate, args.ifu, redux_path=args.redux, dr=args.dr,
                                   drpall_path=args.root, analysis_path=args.analysis)
    #---------------------------------------------------------------------------

    # Run the iterative fit
    disk, p0, lb, ub, fix, vel_mask, sig_mask \
            = axisym.axisym_iter_fit(galmeta, kin, rctype=args.rc, dctype=args.dc,
                                     fitdisp=args.disp, ignore_covar=not args.covar,
                                     max_vel_err=args.max_vel_err, max_sig_err=args.max_sig_err,
                                     min_vel_snr=args.min_vel_snr, min_sig_snr=args.min_sig_snr,
                                     vel_sigma_rej=args.vel_rej, sig_sigma_rej=args.sig_rej,
                                     fix_cen=args.fix_cen, fix_inc=args.fix_inc,
                                     low_inc=args.low_inc, min_unmasked=args.min_unmasked,
                                     select_coherent=args.coherent, fit_scatter=args.fit_scatter,
                                     verbose=args.verbose)

    # Write the output file
    data_file = os.path.join(args.odir, f'{oroot}.fits.gz')
    axisym.axisym_fit_data(galmeta, kin, p0, lb, ub, disk, vel_mask, sig_mask, ofile=data_file)

    if args.skip_plots:
        return

    # Plot the masking data
    mask_plot = os.path.join(args.odir, f'{oroot}-mask.png')
    axisym.axisym_fit_plot_masks(galmeta, kin, disk, vel_mask, sig_mask, ofile=mask_plot)

    # Plot the final residuals
    dv_plot = os.path.join(args.odir, f'{oroot}-vdist.png')
    ds_plot = os.path.join(args.odir, f'{oroot}-sdist.png')
    disk.reject(vel_plot=dv_plot, sig_plot=ds_plot, plots_only=True) 

    # Plot the fit asymmetry
    asym_plot = os.path.join(args.odir, f'{oroot}-asym.png')
    kin.asymmetry_plot(galmeta=galmeta, xc=disk.par[0], yc=disk.par[1], pa=disk.par[2],
                       inc=disk.par[3], vsys=disk.par[4], fwhm=galmeta.psf_fwhm[1], 
                       vel_mask=np.logical_not(disk.vel_gpm),
                       sig_mask=None if disk.dc is None else np.logical_not(disk.sig_gpm),
                       ofile=asym_plot)

    # Create the final fit plot
    fit_plot_err = os.path.join(args.odir, f'{oroot}-fit_err.png')
    
    
    
    #--------------------------------------------------------------------------------------------------
    # copy 'disk', because fisher_matrix changes the atributes
    disk_new = copy.deepcopy(disk)
    # calculate the covariance matrix (inverse of fisher matrix)
    if args.fisher == True: 
       fisher = disk.fisher_matrix(disk.par, kin, sb_wgt=True, scatter=disk.scatter, ignore_covar=True, fix=np.logical_not(disk.free), inverse = True)
    else:
       fisher = None
    
    
#    print(np.sqrt(fisher[0,0])*np.sqrt(fisher[1,1]) -np.absolute(fisher[0,1]), np.sqrt(fisher[0,0])*np.sqrt(fisher[2,2]) -np.absolute(fisher[0,2]), np.sqrt(fisher[0,0])*np.sqrt(fisher[3,3]) -np.absolute(fisher[0,3])\
#    	,np.sqrt(fisher[0,0])*np.sqrt(fisher[4,4]) -np.absolute(fisher[0,4]) )
    	
#    print(np.sqrt(fisher[1,1])*np.sqrt(fisher[2,2]) -np.absolute(fisher[1,2]), np.sqrt(fisher[1,1])*np.sqrt(fisher[3,3]) -np.absolute(fisher[1,3]), np.sqrt(fisher[1,1])*np.sqrt(fisher[4,4]) -np.absolute(fisher[1,4]) )
 #   print(np.sqrt(fisher[2,2])*np.sqrt(fisher[3,3]) -np.absolute(fisher[2,3]), np.sqrt(fisher[2,2])*np.sqrt(fisher[4,4]) -np.absolute(fisher[2,4]) )
 #   print(np.sqrt(fisher[3,3])*np.sqrt(fisher[4,4]) -np.absolute(fisher[3,4]) )
    
    # copy back 'disk' 
    disk = copy.deepcopy(disk_new)
    
    
    axisym.axisym_fit_plot(galmeta, kin, disk, fix=fix, fisher=fisher, ofile=fit_plot_err,save_vrot=args.save_vrot)
    
    
