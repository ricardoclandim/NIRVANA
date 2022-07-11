"""
Module with a class that fits multiple tracers to a single disk.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import warnings

from IPython import embed

import numpy as np
from scipy import optimize
from matplotlib import pyplot, rc, patches, ticker, colors

from .geometry import projected_polar
from ..data.util import select_kinematic_axis, bin_stats, growth_lim, atleast_one_decade
from .util import cov_err
from ..util import plot

class MultiTracerDisk:
    """
    Define a class that enables multiple kinematic datasets to be simultaneously
    fit with the ThinDisk models.

    .. todo::

        internals ...

    Args:
        disk (array-like):
            The :class:`~nirvana.models.thindisk.ThinDisk` models to use for
            each dataset; i.e., there must be one disk model per dataset even if
            the models are identical.  Importantly, these must be separate
            object instances regardless of whether or not they are identical
            models.  The :class:`~nirvana.models.thindisk.ThinDisk` classes are
            used to perform a number of operations that are specific to each
            dataset that expedite the fit; the results are saved to the
            internals of the class meaning that there must be one instance per
            kinematic database.
        tie_base (array-like, optional):
            A 5-element vector indicating which geometric projection parameters
            should tied between the disk models.  The parameters are the
            dynamical center coordinates (x, y), the position angle, the
            inclination, and the systemic velocity.
        tie_disk (array-like, optional):
            A vector indicating which model-specific parameters should be tied
            between disks.  Models to be tied must have the same type and the
            same number of parameters.  The length of this vector must match the
            number of disk model parameters *excluding* the base parameters.
    """
    def __init__(self, disk, tie_base=None, tie_disk=None):

        self.disk = np.atleast_1d(disk)
        self.ntracer = self.disk.size
        if self.ntracer == 1:
            raise ValueError('You are only fitting one dataset.  Use the disk class directly!')

        # Parameter checks
        if np.any([self.disk[i+1].nbp != self.disk[0].nbp for i in range(self.ntracer-1)]):
            raise ValueError('All of the disks should have the same number of base parameters.')
        self.nbp = self.disk[0].nbp
        self.np = np.sum([d.np for d in self.disk])

        # Setup the ties between base and disk parameters
        self.tie_base = None        # Vector indicating which base parameters to tie between disks
        self.tie_disk = None        # Vector indicating which disk parameters to tie
        self.update_tie_base(tie_base)  # Set the base tying strategy
        self.update_tie_disk(tie_disk)  # Set the disk tying strategy

        # Setup the parameters
        self.par = None             # Holds the *unique* (untied) parameters
        self.par_err = None         # Holds any estimated parameter errors
        # TODO: Set this to the bitmask dtype
        self.global_mask = 0        # Global bitmask value
        self.fit_status = None      # Detailed fit status
        self.fit_success = None     # Simple fit success flag

        # Workspace
        self.disk_fom = None        # Functions used to calculate the minimized figure-of-merit
        self.disk_jac = None        # Functions used to calculate the fit Jacobian
        self._wrkspc_parslc = None  # Fit workspace: parameter slices
        self._wrkspc_ndata = None   #   ... Number of data points per dataset
        self._wrkspc_sdata = None   #   ... Starting indices of each dataset in the concatenation
        self._wrkspc_jac = None     #   ... Full fit Jacobian

    def update_tie_base(self, tie_base):
        """
        Setup parameter tying between the base geometric projection parameters.

        Function sets :attr:`tie_base` and calls :func:`retie`.

        Args:
            tie_base (array-like):
                Boolean array indicating which base parameters should be tied
                between disks.
        """
        # Set the internal
        self.tie_base = np.zeros(self.nbp, dtype=bool) if tie_base is None \
                            else np.atleast_1d(tie_base)
        # Check the result
        if self.tie_base.size != self.nbp:
            raise ValueError('Number of parameters in base tying vector incorrect!  Expected '
                             f'{self.nbp}, found {selt.tie_base.size}.')
        if not np.issubdtype(self.tie_base.dtype, bool):
            raise TypeError('Base tying vector should hold booleans!')
        # Reset the tying vectors
        self.retie()

    @property
    def can_tie_disks(self):
        """
        Check if the disks can be tied.

        Current criteria are:

            - disks must have same class type
            - disks must have the same number of parameters

        Returns:
            :obj:`bool`: Flag that disk parameters can be tied.
        """
        types = np.array([d.__class__.__name__ for d in self.disk])
        npar = np.array([d.np for d in self.disk])
        return np.all(types == types[0]) and np.all(npar == npar[0])

    def update_tie_disk(self, tie_disk):
        """
        Setup parameter tying between the disk kinematic parameters

        Function sets :attr:`tie_disk` and calls :func:`retie`.

        Args:
            tie_disk (array-like):
                Boolean array indicating which kinematic parameters should be
                tied between disks.
        """
        # Set the internal
        ndp = self.disk[0].np - self.nbp
        self.tie_disk = np.zeros(ndp, dtype=bool) if tie_disk is None else np.atleast_1d(tie_disk)
        # Check that the disks *can* be tied
        if np.any(self.tie_disk) and not self.can_tie_disks:
            raise ValueError('Disk parameters cannot be tied together!')
        # Check the tying flags
        if self.tie_disk.size != ndp:
            raise ValueError('Number of parameters in kinematics tying vector incorrect!  '
                             f'Expected {self.disk[0].np - self.nbp}, found {self.tie_disk.size}.')
        if not np.issubdtype(self.tie_disk.dtype, bool):
            raise TypeError('Kinematics tying vector should hold booleans!')
        # Reset the tying vectors
        self.retie()

    def retie(self):
        """
        Construct the tying and untying vectors.

        Given the current :attr:`tie_base` and :attr:`tie_disk` arrays, this
        sets the :attr:`tie` and :attr:`untie` internals.

        If ``full_par`` provides the complete set of parameters, including
        redundant parameters that are tied and ``tied_par`` par only contains
        the unique, untied parameters, one can change between them using the
        :attr:`tie` and :attr:`untie` internal arrays as follows:

        .. code-block:: python

            full_par = tied_par[disk.untie]
            tied_par = full_par[disk.tie]

        where ``disk`` is an instance of this class.
        """
        # Find the starting index for the parameters of each disk
        s = [0 if i == 0 else np.sum([d.np for d in self.disk[:i]]) for i in range(self.ntracer)]

        # Instatiate as all parameters being untied
        self.untie = np.arange(self.np)

        # Tie base parameters by replacing parameter indices in disks with the
        # ones from the first disk.
        indx = np.where(self.tie_base)[0]
        if indx.size > 0: 
            for i in range(self.ntracer-1):
                self.untie[indx + s[i+1]] = indx
            
        # Tie disk parameters by replacing parameter indices in disks with the
        # ones from the first disk.
        # NOTE: This works when self.tie_disk is None, so there's no need to test for it
        indx = np.where(self.tie_disk)[0]
        if indx.size > 0:
            for i in range(self.ntracer-1):
                self.untie[indx + s[i+1] + self.nbp] = indx + self.nbp

        # The tying vector simply uses the unique indices in the untying vector
        self.tie, self.untie = np.unique(self.untie, return_inverse=True)

        # Set the number of untied, unique parameters
        self.nup = self.tie.size
            
    def guess_par(self, full=False):
        """
        Return a set of guess parameters based on the guess parameters for each disk.

        Args:
            full (:obj:`bool`, optional):
                Flag to return the full set of parameters, not just the unique,
                untied ones.

        Returns:
            `numpy.ndarray`_: Guess parameters
        """
        gp = np.concatenate([d.guess_par() for d in self.disk])
        return gp if full else gp[self.tie]

    def _init_par(self, p0, fix):
        """
        Initialize the parameter vectors.

        This includes the parameter vector itself, :attr:`par`, and the boolean
        vector selecting the free parameters, :attr:`free`.  This also sets the
        number of free parameters, :attr:`nfree`.

        Args:
            p0 (`numpy.ndarray`_):
                The parameters to use.  The length can be either the total
                number of parameters (:attr:`np`) or the total number of
                unique/untied parameters (:attr:`nup`).  If None, the parameters
                are set by :func:`guess_par`.
            fix (`numpy.ndarray`_):
                Boolean flags to fix the parameters.  If None or if ``p0`` is
                None, all parameters are assumed to be free.  If not None, the
                length *must* match ``p0``.
        """
        # If the parameters are not provided, use the defaults
        if p0 is None:
            if fix is not None:
                warnings.warn('To fix parameter, must provide the full set of initial guess '
                              'parameters.  Ignoring the fixed parameters provided and fitting '
                              'all parameters.')
            self.par = self.guess_par()
            self.free = np.ones(self.nup, dtype=bool)
            self.nfree = self.nup
            return

        _p0 = np.atleast_1d(p0)
        _free = np.ones(_p0.size, dtype=bool) if fix is None else np.logical_not(fix)
        if _p0.size not in [self.np, self.nup]:
            raise ValueError('Incorrect number of model parameters.  Must be either '
                             f'{self.np} (full) or {self.nup} (unique).')
        if _free.size != _p0.size:
            raise ValueError('Vector selecting fixed parameters has different length from '
                             f'parameter vector: {_free.size} instead of {_p0.size}.')
        self.par = _p0.copy()
        self.par_err = None
        self.free = _free.copy()
        if self.par.size == self.np:
            self.par = self.par[self.tie]
            self.free = self.free[self.tie]
        self.nfree = np.sum(self.free)

    def _set_par(self, par):
        """
        Set the full parameter vector, accounting for any fixed and/or tied
        parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length must be either the total
                number of *untied* parameters (:attr:`nup`) or the total number
                of free parameters (:attr:`nfree`).
        """
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.nup:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError('Must provide {0} or {1} parameters.'.format(self.nup, self.nfree))
        self.par[self.free] = par.copy()

    def _disk_slice(self, index):
        """
        Return the slice selecting parameters for the specified disk from the
        *full* parameter vector.

        Args:
            index (:obj:`int`):
                The index of the disk to select.

        Returns:
            :obj:`slice`: Slice selected the relevant parameters.
        """
        s = 0 if index == 0 else np.sum([d.np for d in self.disk[:index]])
        return slice(s, s + self.disk[index].np)

    def lsq_fit(self, kin, sb_wgt=False, p0=None, fix=None, lb=None, ub=None, scatter=None,
                verbose=0, assume_posdef_covar=False, ignore_covar=True, analytic_jac=True,
                maxiter=5):
        """
        Use non-linear least-squares minimization to simultaneously fit all
        datasets.

        For all input parameter vectors (``p0``, ``fix``, ``lb``, and ``ub``),
        the length of all vectors must be the same, but they can be either the
        total number of parameters (:attr:`np`) or the number of unique (untied)
        parameters (:attr:`nup`).  If the former, note that only the values of
        the tied parameters in the first disk will be used *regardless of
        whether or not the parameters are fixed*.
                
        .. warning::

            Currently, this class *does not construct a model of the
            surface-brightness distribution*.  Instead, any weighting of the
            model during convolution with the beam profile uses the as-observed
            surface-brightness distribution, instead of a model of the intrinsic
            surface brightness distribution.  See ``sb_wgt``.

        Args:
            kin (array-like):
                A list of :class:`~nirvana.data.kinematics.Kinematics` objects
                to fit.  The number of datasets *must* match the number of disks
                (:attr:`ntracer`).
            sb_wgt (:obj:`bool`, optional):
                Flag to use the surface-brightness data provided by each dataset
                to weight the model when applying the beam-smearing.  **See the
                warning above**.
            p0 (`numpy.ndarray`_, optional):
                The initial parameters for the model.  See function description
                for the allowed shapes.  If None, the default guess parameters
                are used; see :func:`guess_par`.
            fix (`numpy.ndarray`_, optional):
                A boolean array selecting the parameters that should be fixed
                during the model fit.  See function description for the allowed
                shapes.  If None, all parameters are free.
            lb (`numpy.ndarray`_, optional):
                The lower bounds for the parameters.  See function description
                for the allowed shapes.  If None, the defaults are used (see
                :func:`par_bounds`).
            ub (`numpy.ndarray`_, optional):
                The upper bounds for the parameters.  See function description
                for the allowed shapes.  If None, the defaults are used (see
                :func:`par_bounds`).
            scatter (:obj:`float`, array-like, optional):
                Introduce a fixed intrinsic-scatter term into the model. The 
                scatter is added in quadrature to all measurement errors in the
                calculation of the merit function. If no errors are available,
                this has the effect of renormalizing the unweighted merit
                function by 1/scatter.  Can be None, which means no intrinsic
                scatter is added.  The number of provided values can be either:

                    - 1 (i.e., a single float) to apply the same scatter to all
                      kinematic moments and datasets, 
                    - the same as the number of disks to apply a separate
                      scatter to both kinematic moments for each dataset
                    - twice the number of disks to apply a separate scatter to
                      all the kinematic moments and datasets.  In this case, the
                      order is velocity scatter for disk 1, velocity dispersion
                      scatter for disk 1, velocity scatter for disk 2, etc.

            verbose (:obj:`int`, optional):
                Verbosity level to pass to `scipy.optimize.least_squares`_.
            assume_posdef_covar (:obj:`bool`, optional):
                If any of the :class:`~nirvana.data.kinematics.Kinematics`
                datasets include covariance matrices, this forces the code to
                proceed assuming the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If any of the :class:`~nirvana.data.kinematics.Kinematics`
                datasets include covariance matrices, ignore them and just use
                the inverse variance.
            analytic_jac (:obj:`bool`, optional):
                Use the analytic calculation of the Jacobian matrix during the
                fit optimization.  If False, the Jacobian is calculated using
                finite-differencing methods provided by
                `scipy.optimize.least_squares`_.
            maxiter (:obj:`int`, optional):
                The call to `scipy.optimize.least_squares`_ is repeated when it
                returns best-fit parameters that are *identical* to the input
                parameters (to within a small tolerance).  This parameter sets
                the maximum number of times the fit will be repeated.  Set this
                to 1 to ignore these occurences; ``maxiter`` cannot be None.
        """

        # Check the input
        self.kin = np.atleast_1d(kin)
        if self.kin.size != self.ntracer:
            raise ValueError('Must provide the same number of kinematic databases as disks '
                             f'({self.ntracer}).')
        _scatter = np.atleast_1d(scatter)
        if _scatter.size not in [1, self.ntracer, 2*self.ntracer]:
            raise ValueError(f'Number of scatter terms must be 1, {self.ntracer}, or '
                             f'{2*self.ntracer}; found {_scatter.size}.')

        # Initialize the parameters.  This checks that the parameters have the
        # correct length.
        self._init_par(p0, fix)

        # Prepare the disks for fitting
        self.disk_fom = [None]*self.ntracer
        self.disk_jac = [None]*self.ntracer

        for i in range(self.ntracer):
            self.disk[i]._init_model(None, self.kin[i].grid_x, self.kin[i].grid_y,
                                     self.kin[i].grid_sb if sb_wgt else None,
                                     self.kin[i].beam_fft, True, None, False)
            self.disk[i]._init_data(self.kin[i], None if scatter is None else scatter[i],
                                    assume_posdef_covar, ignore_covar)
            self.disk_fom[i] = self.disk[i]._get_fom()
            self.disk_jac[i] = self.disk[i]._get_jac()

        self._wrkspc_parslc = [self._disk_slice(i) for i in range(self.ntracer)]
        if analytic_jac:
            # Set the least_squares keywords
            jac_kwargs = {'jac': self.jac}
            # Set up the workspace for dealing with tied parameters
            self._wrkspc_ndata = [np.sum(self.disk[i].vel_gpm) + (0 if self.disk[i].dc is None
                                    else np.sum(self.disk[i].sig_gpm))
                                        for i in range(self.ntracer)]
            self._wrkspc_sdata = np.append([0],np.cumsum(self._wrkspc_ndata)[:-1])
            self._wrkspc_jac = np.zeros((np.sum(self._wrkspc_ndata), self.nup), dtype=float)
        else:
            jac_kwargs = {'diff_step': np.full(self.nup, 0.01, dtype=float)[self.free]}

        # Parameter boundaries
        _lb, _ub = self.par_bounds()
        if lb is None:
            lb = _lb
        if ub is None:
            ub = _ub
        if len(lb) != self.nup or len(ub) != self.nup:
            raise ValueError('Length of one or both of the bound vectors is incorrect.')

        # Setup to iteratively fit, where the iterations are meant to ensure
        # that the least-squares fit actually optimizes the parameters.
        # - Set the random number generator with a fixed seed so that the
        #   result is deterministic.
        rng = np.random.default_rng(seed=909)
        # - Set the free parameters.  These are save to a new vector so that the
        #   initial parameters can be tracked for each iteration without losing
        #   the original input.
        _p0 = self.par[self.free]
        p = _p0.copy()
        # - Reset any parameter errors
        pe = None
        # - Start counting the iterations
        niter = 0
        while niter < maxiter:
            # Run the optimization
            result = optimize.least_squares(self.fom, p, x_scale='jac', method='trf',
                                            xtol=1e-12, bounds=(lb[self.free], ub[self.free]), 
                                            verbose=max(verbose,0), **jac_kwargs)
            # Attempt to calculate the errors
            try:
                pe = np.sqrt(np.diag(cov_err(result.jac)))
            except:
                warnings.warn('Unable to compute parameter errors from precision matrix.')
                pe = None

            # The fit should change the input parameters.
            if np.all(np.absolute(p-result.x) > 1e-3):
                break

            # If it doesn't, something likely went wrong with the fit.  Perturb
            # the input guesses a bit and retry.
            p = _p0 + rng.normal(size=self.nfree)*(pe if pe is not None else 0.1*p0)
            p = np.clip(p, lb[self.free], ub[self.free])
            niter += 1

        # TODO: Add something to the fit status/success flags that tests if
        # niter == maxiter and/or if the input parameters are identical to the
        # final best-fit prameters?  Note that the input parameters, p0, may not
        # be identical to the output parameters because of the iterations mean
        # that p != p0 !

        # Save the fit status
        self.fit_status = result.status
        self.fit_success = result.success

        # Save the best-fitting parameters
        self._set_par(result.x)
        if pe is None:
            self.par_err = None
        else:
            self.par_err = np.zeros(self.nup, dtype=float)
            self.par_err[self.free] = pe

        # Print the report
        if verbose > -1:
            self.report(fit_message=result.message)

    def par_bounds(self):
        """
        Return the lower and upper bounds for the unique, untied parameters.

        Returns:
            :obj:`tuple`: A two-tuple of `numpy.ndarray`_ objects with the lower
            and upper parameter boundaries.
        """
        lb, ub = np.array([list(d.par_bounds()) for d in self.disk]).transpose(1,0,2).reshape(2,-1)
        return lb[self.tie], ub[self.tie]

    def fom(self, par):
        """
        Calculate the figure-of-merit for a model by concatenating the result
        from the figure-of-merit calculations provided for each disk.  This is
        the function used by the least-squares optimization in the simultaneous
        fitting process.

        Args:
            par (`numpy.ndarray`_):
                The model parameters; see :func:`_set_par`.
    
        Returns:
            `numpy.ndarray`_: The vector with the model residuals for each
            fitted data point.
        """
        # Get the tied parameters
        self._set_par(par)
        # Untie them to get the full set
        full_par = self.par[self.untie]
        return np.concatenate([self.disk_fom[i](full_par[self._wrkspc_parslc[i]]) 
                                for i in range(self.ntracer)])

    def jac(self, par):
        """
        Compute the model Jacobian using the analytic derivatives provided by
        each disk independently.

        Args:
            par (`numpy.ndarray`_):
                The model parameters; see :func:`_set_par`.
    
        Returns:
            `numpy.ndarray`_: The array with the model Jacobian.
        """
        # Get the tied parameters
        self._set_par(par)
        # Untie them to get the full set
        full_par = self.par[self.untie]
        _jac = [self.disk_jac[i](full_par[self._wrkspc_parslc[i]]) for i in range(self.ntracer)] 
        self._wrkspc_jac[...] = 0.
        assert np.array_equal(self._wrkspc_ndata, [j.shape[0] for j in _jac]), \
                'Jacobians have incorrect shape!'
        for i in range(self.ntracer):
            sec = np.ix_(np.arange(self._wrkspc_ndata[i])+self._wrkspc_sdata[i],
                         self.untie[self._wrkspc_parslc[i]])
            self._wrkspc_jac[sec] += _jac[i]
        return self._wrkspc_jac[:,self.free]

    def report(self, fit_message=None):
        """
        Report the current parameters of the model to the screen.

        Args:
            fit_message (:obj:`str`, optional):
                The status message returned by the fit optimization.
        """
        if self.par is None:
            print('No parameters to report.')
            return

        print('-'*70)
        print(f'{"Fit Result":^70}')
        print('-'*70)
        if fit_message is not None:
            print(f'Fit status message: {fit_message}')
        if self.fit_status is not None:
            print(f'Fit status: {self.fit_status}')
        print(f'Fit success: {"True" if self.fit_status else "False"}')
        print('-'*50)
        # Indicate which parameters are tied
        if np.any(self.tie_base) or np.any(self.tie_disk):
            disk0_parn = self.disk[0].par_names()
            tied = self.tie_base if not np.any(self.tie_disk) \
                        else np.append(self.tie_base, self.tie_disk)
            print('Tied Parameters:')
            for i in range(tied.size):
                if not tied[i]:
                    continue
                print(f'{disk0_parn[i]:>30}')
        else:
            print('No tied parameters')

        # Print the results for each disk
        full_par = self.par[self.untie]
        full_par_err = None if self.par_err is None else self.par_err[self.untie]
        full_free = self.free[self.untie]
        full_free[np.setdiff1d(np.arange(self.np), self.tie)] = False
        for i in range(self.ntracer):
            print('-'*50)
            print(f'{f"Disk {i+1}":^50}')
            print('-'*50)
            slc = self._disk_slice(i)
            self.disk[i].par = full_par[slc]
            self.disk[i].free = full_free[slc]
            self.disk[i].par_err = None if full_par_err is None else full_par_err[slc]
            self.disk[i].report(component=True)
        print('-'*50)

        resid = self.fom(self.par)
        chisqr = np.sum(resid**2) / (resid.size - self.nfree)
        print(f'Total reduced chi-square: {chisqr}')
        print('-'*70)        


# TODO:
#   - Add keyword for radial sampling for 1D model RCs and dispersion profiles
#   - This is MaNGA-specific and needs to be abstracted
#   - Allow the plot to be constructed from the fits file written by
#     axisym_fit_data
def asymdrift_fit_plot(galmeta, kin, disk, par=None, par_err=None, fix=None, ofile=None):
    """
    Construct the QA plot for the result of fitting an
    :class:`~nirvana.model.axisym.AxisymmetricDisk` model to a galaxy.

    Args:
        galmeta (:class:`~nirvana.data.meta.GlobalPar`):
            Object with metadata for the galaxy to be fit.
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data to be fit
        disk (:class:`~nirvana.models.axisym.AxisymmetricDisk`):
            Object that performed the fit and has the best-fitting parameters.
        par (`numpy.ndarray`_, optional):
            The parameters of the model.  If None are provided, the parameters
            in ``disk`` are used.
        par_err (`numpy.ndarray`_, optional):
            The errors in the model parameters.  If None are provided, the
            parameter errors in ``disk`` are used.
        fix (`numpy.ndarray`_, optional):
            Flags indicating the parameters that were fixed during the fit.  If
            None, all parameters are assumed to have been free.
        ofile (:obj:`str`, optional):
            Output filename for the plot.  If None, the plot is shown to the
            screen.
    """
    logformatter = plot.get_logformatter()

    # Change the style
    rc('font', size=8)

    if disk.par is None and par is None:
        raise ValueError('No model parameters available.  Provide directly or via disk argument.')

    _par = disk.par[disk.untie] if par is None else par
    if disk.par_err is None and par_err is None:
        _par_err = np.full(_par.size, -1., dtype=float)
    else:
        _par_err = disk.par_err[disk.untie] if par_err is None else par_err

    if fix is None:
        _fix = np.logical_not(disk.free)[disk.untie]
        _fix[np.setdiff1d(np.arange(disk.np), disk.tie)] = True
    else:
        _fix = fix

    if _par.size != disk.np:
        raise ValueError('Number of provided parameters has the incorrect size.')
    if _par_err.size != disk.np:
        raise ValueError('Number of provided parameter errors has the incorrect size.')
    if _fix.size != disk.np:
        raise ValueError('Number of provided parameter fixing flags has the incorrect size.')

    mean_vsys = (_par[4] + _par[disk.disk[0].np+4])/2

    # Build the data to plot
    fwhm = galmeta.psf_fwhm[1]
    oversample = 1.5
    maj_wedge = 30.

    sb_map = [None]*2
    v_map = [None]*2
    s_map = [None]*2
    vmod = [None]*2
    vmod_map = [None]*2
    smod = [None]*2
    smod_map = [None]*2

    spax_vrot_r = [None]*2
    spax_vrot = [None]*2
    spax_smaj_r = [None]*2
    spax_smaj = [None]*2

    bin_r = [None]*2
    bin_vrot = [None]*2
    bin_vrote = [None]*2
    bin_vrotn = [None]*2
    bin_smaj = [None]*2
    bin_smaje = [None]*2
    bin_smajn = [None]*2

    for i in range(2):
        sb_map[i] = kin[i].remap('sb')
        v_map[i] = kin[i].remap('vel')
        s_map[i] = np.ma.sqrt(kin[i].remap('sig_phys2', mask=kin[i].sig_mask))

        # Construct the model data, both binned data and maps
        slc = disk._disk_slice(i)
        disk.disk[i].par = _par[slc]
        models = disk.disk[i].model()
        if disk.disk[i].dc is None:
            vmod[i] = kin[i].bin(models)
            vmod_map[i] = kin[i].remap(vmod[i], mask=kin[i].vel_mask)
            smod[i] = None
            smod_map[i] = None
        else:
            vmod[i] = kin[i].bin(models[0])
            vmod_map[i] = kin[i].remap(vmod[i], mask=kin[i].vel_mask)
            smod[i] = kin[i].bin(models[1])
            smod_map[i] = kin[i].remap(smod[i], mask=kin[i].sig_mask)

        # Get the projected rotational velocity
        #   - Disk-plane coordinates
        r, th = projected_polar(kin[i].x - disk.disk[i].par[0], kin[i].y - disk.disk[i].par[1],
                                *np.radians(disk.disk[i].par[2:4]))
        #   - Mask for data along the major axis
        major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)
        #   - Projected rotation velocities
        indx = major_gpm & np.logical_not(kin[i].vel_mask)
        spax_vrot_r[i] = r[indx]
        spax_vrot[i] = (kin[i].vel[indx] - disk.disk[i].par[4])/np.cos(th[indx])
        #   - Major axis velocity dispersions
        indx = major_gpm & np.logical_not(kin[i].sig_mask) & (kin[i].sig_phys2 > 0)
        spax_smaj_r[i] = r[indx]
        spax_smaj[i] = np.sqrt(kin[i].sig_phys2[indx])

        bin_r[i], bin_vrot[i], bin_vrote[i], _, bin_vrotn[i], _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, bin_smaj[i], bin_smaje[i], _, bin_smajn[i], _, _, _, _, _, _, _, _, \
                = kin[i].radial_profiles(fwhm/oversample, xc=disk.disk[i].par[0],
                                         yc=disk.disk[i].par[1], pa=disk.disk[i].par[2],
                                         inc=disk.disk[i].par[3], vsys=disk.disk[i].par[4],
                                         maj_wedge=maj_wedge)

    resid = disk.fom(_par[disk.tie])
    rchisqr = np.sum(resid**2) / (resid.size - disk.nfree)

    r_map, th_map = projected_polar(kin[0].grid_x - disk.disk[0].par[0],
                                    kin[0].grid_y - disk.disk[0].par[1],
                                    np.radians(disk.disk[0].par[2]),
                                    np.radians(disk.disk[0].par[3]))

    ad_map = (v_map[0] - disk.disk[0].par[4])**2 - (v_map[1] - disk.disk[1].par[4])**2
    ad_map = np.ma.divide(ad_map, np.cos(th_map)**2)
    admod_map = (vmod_map[0] - disk.disk[0].par[4])**2 - (vmod_map[1] - disk.disk[1].par[4])**2
    admod_map = np.ma.divide(admod_map, np.cos(th_map)**2)

    ados_map = np.ma.divide(ad_map, kin[1].remap('sig_phys2', mask=kin[1].sig_mask))
    adosmod_map = np.ma.divide(admod_map, smod_map[1]**2)

    # Get the 1D model profiles
    maxr = np.amax(np.concatenate(spax_vrot_r+spax_smaj_r))
    modelr = np.arange(0, maxr, 0.1)
    vrotm = [None]*2
    smajm = [None]*2
    for i in range(2):
        vrotm[i] = disk.disk[i].rc.sample(modelr, par=disk.disk[i].rc_par())
        smajm[i] = disk.disk[i].dc.sample(modelr, par=disk.disk[i].dc_par())

    spax_ad_r, spax_ad, bin_ad_r, bin_ad, bin_ade, bin_adn \
            = asymdrift_radial_profile(disk, kin, fwhm/oversample, maj_wedge=maj_wedge)

    spax_ad = np.ma.sqrt(spax_ad).filled(0.0)
    bin_ad = np.ma.sqrt(bin_ad).filled(0.0)
    bin_ade = np.ma.divide(bin_ade, 2*bin_ad).filled(0.0)

    # Set the radius limits for the radial plots
    r_lim = [0.0, maxr * 1.05]
    rc_lim = growth_lim(np.concatenate(bin_vrot+vrotm), 0.99, 1.3)
    smaj_lim = growth_lim(np.ma.log10(np.concatenate(smajm + [bin_ad])).compressed(), 0.9, 1.5)
    smaj_lim = atleast_one_decade(np.power(10.0, smaj_lim))

    # TODO: Extent may need to be adjusted by 0.25 arcsec!  extent is from the
    # edge of the pixel, not from its center.
    # Set the extent for the 2D maps
    extent = [np.amax(kin[0].grid_x), np.amin(kin[0].grid_x),
              np.amin(kin[0].grid_y), np.amax(kin[0].grid_y)]
    Dx = max(extent[0]-extent[1], extent[3]-extent[2]) # *1.01
    skylim = np.array([ (extent[0]+extent[1] - Dx)/2., 0.0 ])
    skylim[1] = skylim[0] + Dx

    sb_lim = [growth_lim(np.ma.log10(sb_map[0]).compressed(), 0.90, 1.05),
              growth_lim(np.ma.log10(sb_map[1]).compressed(), 0.90, 1.05)]
    sb_lim = [atleast_one_decade(np.power(10.0, sb_lim[0])),
              atleast_one_decade(np.power(10.0, sb_lim[1]))]

    vel_lim = growth_lim(np.ma.concatenate(v_map+vmod_map).compressed(), 0.90, 1.05,
                         midpoint=mean_vsys)

    ad_lim = growth_lim(np.ma.log10(np.ma.append(ad_map, admod_map)).compressed(), 0.70, 1.05)
    ad_lim = atleast_one_decade(np.power(10.0, ad_lim))

    sig_lim = growth_lim(np.ma.log10(np.ma.append(s_map[1], smod_map[1])).compressed(), 0.70, 1.05)
    sig_lim = atleast_one_decade(np.power(10., sig_lim))

#    ados_lim = np.power(10.0, growth_lim(np.ma.log10(np.ma.append(ados_map, adosmod_map).compressed(),
#                                        0.80, 1.05)))
#    ados_lim = atleast_one_decade(sig_lim)

    ados_lim = growth_lim(np.ma.append(ados_map, adosmod_map).compressed(), 0.80, 1.05)

    # Create the plot
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2*w,2*h))

    #-------------------------------------------------------------------
    # Gas velocity field
    ax = plot.init_ax(fig, [0.02, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_map[0], origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$V_g$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Gas velocity field model
    ax = plot.init_ax(fig, [0.02, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(vmod_map[0], origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$V_{g,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Stellar velocity field
    ax = plot.init_ax(fig, [0.215, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_map[1], origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$V_\ast$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Stellar velocity field model
    ax = plot.init_ax(fig, [0.215, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(vmod_map[1], origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$V_{\ast,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Measured AD
    ax = plot.init_ax(fig, [0.410, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ad_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=ad_lim[0], vmax=ad_lim[1]), zorder=4) 
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma^2_a$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # AD Model
    ax = plot.init_ax(fig, [0.410, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(admod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=ad_lim[0], vmax=ad_lim[1]), zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\sigma^2_{a,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion
    ax = plot.init_ax(fig, [0.605, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(s_map[1], origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma_\ast$', ha='left', va='center', transform=ax.transAxes)
 
    #-------------------------------------------------------------------
    # Velocity Dispersion Model
    ax = plot.init_ax(fig, [0.605, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(smod_map[1], origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\sigma_{\ast,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # AD ratio
    ax = plot.init_ax(fig, [0.800, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ados_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, vmin=ados_lim[0], vmax=ados_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma^2_a/\sigma^2_\ast$',
            ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Model AD ratio
    ax = plot.init_ax(fig, [0.800, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(adosmod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, vmin=ados_lim[0], vmax=ados_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$\sigma^2_{a,m}/\sigma^2_{\ast,m}$', ha='left', va='center',
             transform=ax.transAxes)

    #-------------------------------------------------------------------
    # H-alpha surface-brightness
    ax = plot.init_ax(fig, [0.800, 0.305, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.50, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sb_map[0], origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[0][0], vmax=sb_lim[0][1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # TODO: For some reason, the combination of the use of a masked array and
    # setting the formatter to logformatter leads to weird behavior in the map.
    # Use something like the "pallete" object described here?
    #   https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\mu_g$', ha='left', va='center', transform=ax.transAxes)

#    ax.text(0.5, 1.2, 'Intrinsic Model', ha='center', va='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Continuum surface brightness
    ax = plot.init_ax(fig, [0.800, 0.110, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.10, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sb_map[1], origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[1][0], vmax=sb_lim[1][1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\mu_\ast$', ha='left', va='center', transform=ax.transAxes)

#    #-------------------------------------------------------------------
#    # Annotate with the intrinsic scatter included
#    ax.text(0.00, -0.2, r'V scatter, $\epsilon_v$:', ha='left', va='center',
#            transform=ax.transAxes, fontsize=10)
#    ax.text(1.00, -0.2, f'{vsct:.1f}', ha='right', va='center', transform=ax.transAxes,
#            fontsize=10)
#    if disk.dc is not None:
#        ax.text(0.00, -0.3, r'$\sigma^2$ scatter, $\epsilon_{\sigma^2}$:', ha='left', va='center',
#                transform=ax.transAxes, fontsize=10)
#        ax.text(1.00, -0.3, f'{ssct:.1f}', ha='right', va='center', transform=ax.transAxes,
#                fontsize=10)

    #-------------------------------------------------------------------
    # SDSS image
    ax = fig.add_axes([0.01, 0.29, 0.23, 0.23])
    if kin[0].image is not None:
        ax.imshow(kin[0].image)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes,
                fontsize=20)

    ax.text(0.5, 1.05, 'SDSS gri Composite', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if galmeta.primaryplus:
        sample='Primary+'
    elif galmeta.secondary:
        sample='Secondary'
    elif galmeta.ancillary:
        sample='Ancillary'
    else:
        sample='Filler'

    # Assume center, PA, and inclination are tied, and that the systemic velocities are not.

    # MaNGA ID
    ax.text(0.00, -0.05, 'MaNGA ID:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.05, f'{galmeta.mangaid}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    # Observation
    ax.text(0.00, -0.13, 'Observation:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.13, f'{galmeta.plate}-{galmeta.ifu}', ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sample selection
    ax.text(0.00, -0.21, 'Sample:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.21, f'{sample}', ha='right', va='center', transform=ax.transAxes, fontsize=10)
    # Redshift
    ax.text(0.00, -0.29, 'Redshift:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.29, '{0:.4f}'.format(galmeta.z), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Mag
    ax.text(0.00, -0.37, 'Mag (N,r,i):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    if galmeta.mag is None:
        ax.text(1.01, -0.37, 'Unavailable', ha='right', va='center',
                transform=ax.transAxes, fontsize=10)
    else:
        ax.text(1.01, -0.37, '{0:.1f}/{1:.1f}/{2:.1f}'.format(*galmeta.mag), ha='right',
                va='center', transform=ax.transAxes, fontsize=10)
    # PSF FWHM
    ax.text(0.00, -0.45, 'FWHM (g,r):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.45, '{0:.2f}, {1:.2f}'.format(*galmeta.psf_fwhm[:2]), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sersic n
    ax.text(0.00, -0.53, r'Sersic $n$:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.53, '{0:.2f}'.format(galmeta.sersic_n), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Stellar Mass
    ax.text(0.00, -0.61, r'$\log(\mathcal{M}_\ast/\mathcal{M}_\odot$):', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.61, '{0:.2f}'.format(np.log10(galmeta.mass)), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Phot Inclination
    ax.text(0.00, -0.69, r'$i_{\rm phot}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.69, '{0:.1f}'.format(galmeta.guess_inclination(lb=1., ub=89.)),
            ha='right', va='center', transform=ax.transAxes, fontsize=10)
    # Fitted center
    ax.text(0.00, -0.77, r'$x_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[0] else 'k')
    xstr = r'{0:.2f}'.format(_par[0]) if _fix[0] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(_par[0], _par_err[0])
    ax.text(1.01, -0.77, xstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[0] else 'k')
    ax.text(0.00, -0.85, r'$y_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[1] else 'k')
    ystr = r'{0:.2f}'.format(_par[1]) if _fix[1] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(_par[1], _par_err[1])
    ax.text(1.01, -0.85, ystr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[1] else 'k')
    # Position angle
    ax.text(0.00, -0.93, r'$\phi_0$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[2] else 'k')
    pastr = r'{0:.1f}'.format(_par[2]) if _fix[2] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(_par[2], _par_err[2])
    ax.text(1.01, -0.93, pastr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[2] else 'k')
    # Kinematic Inclination
    ax.text(0.00, -1.01, r'$i_{\rm kin}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[3] else 'k')
    incstr = r'{0:.1f}'.format(_par[3]) if _fix[3] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(_par[3], _par_err[3])
    ax.text(1.01, -1.01, incstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[3] else 'k')
    # Systemic velocity
    ax.text(0.00, -1.09, r'$\langle V_{\rm sys}\rangle$ [km/s]',
            ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='k')
    vsysstr = r'{0:.1f}'.format(mean_vsys)
    ax.text(1.01, -1.09, vsysstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10, color='k')
    # Reduced chi-square
    ax.text(0.00, -1.17, r'$\chi^2_\nu$', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.17, f'{rchisqr:.2f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)

    #-------------------------------------------------------------------
    # Radial plot radius limits
    # Select bins with sufficient data
    vrot_indx = [vrn > 5 for vrn in bin_vrotn]
    smaj_indx = [smn > 5 for smn in bin_smajn]
    for i in range(2):
        if not np.any(vrot_indx[i]):
            vrot_indx[i] = bin_vrotn[i] > 0
        if not np.any(smaj_indx[i]):
            smaj_indx[i] = bin_smajn[i] > 0
    ad_indx = bin_adn > 5
    if not np.any(ad_indx):
        ad_indx = bin_adn > 0

    #-------------------------------------------------------------------
    # Rotation curves
    reff_lines = np.arange(galmeta.reff, r_lim[1], galmeta.reff) if galmeta.reff > 1 else None

    ax = plot.init_ax(fig, [0.27, 0.27, 0.51, 0.23], facecolor='0.9', top=False, right=False)
    ax.set_xlim(r_lim)
    ax.set_ylim(rc_lim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())

    # Gas
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C3')])
    ax.scatter(spax_vrot_r[0], spax_vrot[0],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(vrot_indx[0]):
        ax.scatter(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                   marker='o', s=90, alpha=1.0, color='C3', zorder=4)
        ax.errorbar(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                    yerr=bin_vrote[0][vrot_indx[0]], color='C3', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrotm[0], color='C3', zorder=5, lw=1)
    # Stars
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C0')])
    ax.scatter(spax_vrot_r[1], spax_vrot[1],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(vrot_indx[1]):
        ax.scatter(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                   marker='o', s=90, alpha=1.0, color='C0', zorder=4)
        ax.errorbar(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                    yerr=bin_vrote[1][vrot_indx[1]], color='C0', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrotm[1], color='C0', zorder=5, lw=1)

    if reff_lines is not None:
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=1, color='k')

    asec2kpc = galmeta.kpc_per_arcsec()
    if asec2kpc > 0:
        axt = plot.get_twin(ax, 'x')
        axt.set_xlim(np.array(r_lim) * galmeta.kpc_per_arcsec())
        axt.set_ylim(rc_lim)
        ax.text(0.5, 1.14, r'$R$ [$h^{-1}$ kpc]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
    else:
        ax.text(0.5, 1.05, 'kpc conversion unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

#    kin_inc = disk.par[3]
#    axt = plot.get_twin(ax, 'y')
#    axt.set_xlim(r_lim)
#    axt.set_ylim(np.array(rc_lim)/np.sin(np.radians(kin_inc)))
#    plot.rotate_y_ticks(axt, 90, 'center')
#    axt.spines['right'].set_color('0.4')
#    axt.tick_params(which='both', axis='y', colors='0.4')
#    axt.yaxis.label.set_color('0.4')

    ax.add_patch(patches.Rectangle((0.79,0.45), 0.19, 0.09, facecolor='w', lw=0, edgecolor='none',
                                   zorder=5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.97, 0.451, r'$V\ \sin i$ [km/s]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)
#    ax.text(0.97, 0.56, r'$V$ [km/s; right axis]', ha='right', va='bottom', color='0.4',
#            transform=ax.transAxes, fontsize=10, zorder=6)


    #-------------------------------------------------------------------
    # Velocity Dispersion profile
    ax = plot.init_ax(fig, [0.27, 0.04, 0.51, 0.23], facecolor='0.9')
    ax.set_xlim(r_lim)
    ax.set_ylim(sig_lim)#[10,275])
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(logformatter)
    plot.rotate_y_ticks(ax, 90, 'center')

    # Gas
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C3')])
    ax.scatter(spax_smaj_r[0], spax_smaj[0],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(smaj_indx[0]):
        ax.scatter(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                   marker='o', s=90, alpha=1.0, color='C3', zorder=4)
        ax.errorbar(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                    yerr=bin_smaje[0][smaj_indx[0]], color='C3', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, smajm[0], color='C3', zorder=5, lw=1)
    # Stars
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C0')])
    ax.scatter(spax_smaj_r[1], spax_smaj[1],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(smaj_indx[1]):
        ax.scatter(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                   marker='o', s=90, alpha=1.0, color='C0', zorder=4)
        ax.errorbar(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                    yerr=bin_smaje[1][smaj_indx[1]], color='C0', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, smajm[1], color='C0', zorder=5, lw=1)
    # Sigma AD
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('k')])
    ax.scatter(spax_ad_r, spax_ad,
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(ad_indx):
        ax.scatter(bin_ad_r[ad_indx], bin_ad[ad_indx],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_ad_r[ad_indx], bin_ad[ad_indx],
                   marker='o', s=90, alpha=1.0, color='k', zorder=4)
        ax.errorbar(bin_ad_r[ad_indx], bin_ad[ad_indx], yerr=bin_ade[ad_indx],
                    color='k', capsize=0, linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, np.sqrt(vrotm[0]**2 - vrotm[1]**2), color='k', zorder=5, lw=1)
    if reff_lines is not None:
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=1, color='k')

    ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)

    ax.add_patch(patches.Rectangle((0.81,0.86), 0.17, 0.09, facecolor='w', lw=0,
                                    edgecolor='none', zorder=5, alpha=0.7,
                                    transform=ax.transAxes))
    ax.text(0.97, 0.861, r'$\sigma_{\rm maj}$ [km/s]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)

    # TODO:
    #   - Add errors (if available)?
    #   - Surface brightness units?

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

    # Reset to default style
    pyplot.rcdefaults()


def asymdrift_radial_profile(disk, kin, rstep, maj_wedge=30.):
    """
    Construct azimuthally averaged radial profiles of the kinematics.
    """

    if disk.ntracer != 2:
        raise NotImplementedError('Must provide two disks, first gas, second stars.')
    if len(kin) != 2:
        raise NotImplementedError('Must provide two kinematic datasets, first gas, second stars.')

    if not np.allclose(kin[0].grid_x, kin[1].grid_x) \
            or not np.allclose(kin[0].grid_y, kin[1].grid_y):
        raise NotImplementedError('Kinematics datasets must have the same grid sampling!')

    if not np.all(disk.tie_base[:4]):
        raise NotImplementedError('Disk must have tied the xc, yc, pa, and inc for both tracers.')

    par = disk.par[disk.untie]

    # Disk-plane coordinates
    r, th = projected_polar(kin[0].grid_x - par[0], kin[0].grid_y - par[1], np.radians(par[2]),
                            np.radians(par[3]))

    # Mask for data along the major and minor axes
    major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)

    # Set the radial bins
    binr = np.arange(rstep/2, np.amax(r), rstep)
    binw = np.full(binr.size, rstep, dtype=float)

    # Projected gas and stellar rotation velocities (within the major axis wedge)
    gas_vel = kin[0].remap('vel', mask=np.logical_not(disk.disk[0].vel_gpm))
    gas_vel_ivar = kin[0].remap('vel_ivar',
                                mask=np.logical_not(disk.disk[0].vel_gpm) | kin[0].vel_mask)

    str_vel = kin[1].remap('vel', mask=np.logical_not(disk.disk[1].vel_gpm))
    str_vel_ivar = kin[1].remap('vel_ivar',
                                mask=np.logical_not(disk.disk[1].vel_gpm) | kin[1].vel_mask)

    indx = major_gpm & np.logical_not(np.ma.getmaskarray(gas_vel)) \
                & np.logical_not(np.ma.getmaskarray(str_vel))

    ad_r = r[indx]
    gas_vrot = (gas_vel.data[indx] - par[4])/np.cos(th[indx])
    gas_vrot_ivar = gas_vel_ivar.data[indx]*np.cos(th[indx])**2

    str_vrot = (str_vel.data[indx] - par[disk.disk[0].np+4])/np.cos(th[indx])
    str_vrot_ivar = str_vel_ivar.data[indx]*np.cos(th[indx])**2

    ad = gas_vrot**2 - str_vrot**2
    ad_wgt = 1 / 2 / (gas_vrot**2/gas_vrot_ivar + str_vrot**2/str_vrot_ivar)
#    sini = np.sin(np.radians(par[3]))
#    ad /= sini
#    ad_wgt *= sini**2
    _, _, _, _, _, _, _, ad_ewmean, ad_ewsdev, _, _, ad_nbin, ad_bin_gpm \
            = bin_stats(ad_r, ad, binr, binw, wgts=ad_wgt, fill_value=0.0) 

    return ad_r, ad, binr, ad_ewmean, ad_ewsdev, ad_nbin


