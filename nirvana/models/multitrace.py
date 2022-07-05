"""
Module with a class that fits multiple tracers to a single disk.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import warnings

from IPython import embed

import numpy as np
from scipy import optimize
from .util import cov_err


class MultiTracerDisk:
    """
    Define a class that enables multiple kinematic datasets to be simultaneously
    fit with the ThinDisk models.

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
        self.tie_base = None
        self.tie_disk = None
        self.update_tie_base(tie_base)
        self.update_tie_disk(tie_disk)

        # Setup the parameters
        self.par = None
        self.par_err = None
        self.global_mask = 0
        self.fit_status = None
        self.fit_success = None

        # Workspace
        self.disk_fom = None
        self.disk_jac = None
        self._wrkspc_parslc = None
        self._wrkspc_ndata = None
        self._wrkspc_sdata = None
        self._wrkspc_jac = None


    def update_tie_base(self, tie_base):
        """
        Setup parameter tying between the base geometric projection parameters.
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

        """
        types = np.array([d.__class__.__name__ for d in self.disk])
        npar = np.array([d.np for d in self.disk])
        return np.all(types == types[0]) and np.all(npar == npar[0])

    def update_tie_disk(self, tie_disk):
        """
        Setup parameter tying between the disk kinematic parameters
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
        Construct the tying and untying vectors

        full_par = tied_par[self.untie]
        tied_par = full_par[self.tie]

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
                unique/untied parameters (:attr:`nup`).  If the former, the
                vector is tied and untied to make sure that the tied parameters
                are identical.  If None, the parameters are set by
                :func:`guess_par`.
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
                number of *untied* parameters or the total number of free
                parameters.
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

        For all input parameter vectors (``p0``, ``fix``, ``lb``, and ``ub``),
        the length of all vectors must be the same, but they can be either the
        total number of parameters (:attr:`np`) or the number of unique (untied)
        parameters (:attr:`nup`).  If the former, note that only the values of
        the tied parameters in the first disk will be used.
                
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

        # Check the length of scatter

        self.kin = np.atleast_1d(kin)
        if self.kin.size != self.ntracer:
            raise ValueError('Must provide the same number of kinematic databases as disks '
                             f'({self.ntracer}).')

        self.disk_fom = [None]*self.ntracer
        self.disk_jac = [None]*self.ntracer

        self._init_par(p0, fix)

        # Prepare the disks for fitting
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
        lb, ub = self.par_bounds()
        result = optimize.least_squares(self.fom, self.par[self.free], x_scale='jac', method='trf',
                                        xtol=1e-12, bounds=(lb[self.free], ub[self.free]), 
                                        verbose=max(verbose,0), **jac_kwargs)
        embed()
        exit()

    def par_bounds(self):
        """
        """
        lb, ub = np.array([list(d.par_bounds()) for d in self.disk]).transpose(1,0,2).reshape(2,-1)
        return lb[self.tie], ub[self.tie]

    def fom(self, par):
        """
        """
        # Get the tied parameters
        self._set_par(par)
        # Untie them to get the full set
        full_par = self.par[self.untie]
        return np.concatenate([self.disk_fom[i](full_par[self._wrkspc_parslc[i]]) 
                                for i in range(self.ntracer)])

    def jac(self, par):
        """
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

            

        

        





