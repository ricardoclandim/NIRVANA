"""
Two-dimenaional binning routines.
"""
import warnings

import numpy as np
from matplotlib import pyplot

from vorbin.voronoi_2d_binning import voronoi_2d_binning
from .util import get_map_bin_transformations, fill_matrix

# (Mostly) Copied from MaNGA DAP
class VoronoiBinning:
    """
    Class that wraps Voronoi binning code.
    """
    def __init__(self):
        self.covar = None

    def sn_calculation_no_covariance(self, index, signal, noise): 
        """
        S/N calculation for independent data.

        Args:
            index (`numpy.ndarray`_):
                Indices of the measurements in a single bin.
            signal (`numpy.ndarray`_):
                The signal measurements.
            noise (`numpy.ndarray`_):
                The noise measurements.

        Returns:
            :obj:`float`: The nominal signal-to-noise reached by
            summing the measurements selected by ``index``.
        """
        return np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))

    def sn_calculation_covariance_matrix(self, index, signal, noise):
        """
        Calculate the S/N using a full covariance matrix.

        The method uses the internal :attr:`covar`.

        Args:
            index (`numpy.ndarray`_):
                Indices of the measurements in a single bin.
            signal (`numpy.ndarray`_):
                The signal measurements.
            noise (`numpy.ndarray`_):
                The noise measurements.

        Returns:
            :obj:`float`: The nominal signal-to-noise reached by
            summing the measurements selected by ``index``, including
            any covariance.
        """
        return np.sum(signal[index]) / np.sqrt(np.sum(self.covar[np.ix_(index, index)]))

    @classmethod
    def bin_index(cls, x, y, signal, noise, target_snr, show=False):
        """
        Bin the data and return the indices of the bins.

        Args:
            x (`numpy.ndarray`_):
                Fiducial Cartesian X position.  Shape must match ``signal``.
            y (`numpy.ndarray`_):
                Fiducial Cartesian Y position.  Shape must match ``signal``.
            signal (`numpy.ndarray`_):
                The signal measurements.  Shape must be 1D.
            noise (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                The noise measurements.  Shape can be 1D or 2D.  If 2D, assumed
                to be a covariance matrix.  The 1D size or the length along one
                axis of a 2D array must match ``signal``.
            target_snr (:obj:`float`):
                Target minimum S/N for the bins.
            show (:obj:`bool`, optional):
                Show the default plot with the binning results.

        Returns:
            `numpy.ndarray`_: An integer bin index for each position.

        Raises:
            ValueError:
                Raised if the sizes of ``x`` and ``y`` do not match, or if
                various checks of the signal, noise, and/or covariance elements
                are incorrectly matched.
        """
        self = cls()

        # Check the position input
        if signal.ndim != 1:
            raise ValueError('Signal values must be in a 1D array.')
        if signal.shape != x.shape:
            raise ValueError('Shape of signal does not match coordinates.')
        if x.shape != y.shape:
            raise ValueError('Shape of x and y coordinates do not match!')
        if noise.ndim == 1:
            if noise.shape != x.shape:
                raise ValueError('Shape of noise does not match coordinates.')
            _noise = np.atleast_1d(noise)
            self.covar = None
            sn_func = self.sn_calculation_no_covariance
        if noise.ndim == 2:
            if noise.shape[0] != x.size:
                raise ValueError('Single axis length of covariance does not match coordinates.')
            if noise.shape[0] != noise.shape[1]:
                raise ValueError('Covariance arrays must be square.')
            _noise = np.sqrt(np.diag(noise))
            self.covar = noise.copy()
            sn_func = self.sn_calculation_covariance_matrix

        # All spaxels have S/N greater than threshold, so return each
        # spaxel in its own "bin"
        if np.min(signal/_noise) > target_snr:
            warnings.warn('All pixels have enough S/N. Binning is not needed')
            return np.arange(signal.size)

        # Cannot reach the S/N using all spaxels, so return all spaxels
        # in a single bin
        sn_total = sn_func(np.arange(signal.size), signal, _noise)
        if sn_total < target_snr:
            warnings.warn('Cannot reach target S/N using all data; all data included in one bin.')
            return np.zeros(signal.size)

        # Bin the data
        try:
            binid, xNode, yNode, xBar, yBar, sn, area, scale = \
                    voronoi_2d_binning(x, y, signal, _noise, target_snr, sn_func=sn_func,
                                       plot=show)
            if show:
                pyplot.show()
        except:
            warnings.warn('Binning algorithm has raised an exception.  Assume this is because '
                          'all the spaxels should be in the same bin.')
            binid = numpy.zeros(signal.size)

        return binid


class Bin2D:
    """
    A utility class for handling two-dimensional binning.

    The core functionality of the class is to compute a set of transformations
    using :func:`~nirvana.data.util.get_map_bin_transformations` and provide
    convenience methods that apply and revert those transformations.

    Args:
        spatial_shape (:obj:`tuple`, optional):
            The 2D spatial shape of the mapped data. Ignored if ``binid`` is
            provided.
        binid (`numpy.ndarray`_, optional):
            The 2D array providing the 0-indexed bin ID number associated with
            each map element. Bin IDs of -1 are assumed to be ignored; no bin ID
            can be less than -1. Shape is ``spatial_shape`` and its size (i.e.
            the number of grid points in the map) is :math:`N_{\rm spaxel}`.
    """
    def __init__(self, spatial_shape=None, binid=None):
        self.spatial_shape = spatial_shape if binid is None else binid.shape
        self.ubinid, self.nbin, self.ubin_indx, self.grid_indx, self.bin_inverse, \
                self.bin_transform = get_map_bin_transformations(spatial_shape=spatial_shape,
                                                                 binid=binid)
        self.unravel = np.unravel_index(self.grid_indx, self.spatial_shape)

    def bin(self, data):
        """
        Provided a set of mapped data, bin it according to the internal bin ID
        map.

        Args:
            data (`numpy.ndarray`_):
                Data to bin. Shape must match :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: A vector with the binned data.

        Raises:
            ValueError:
                Raised if the shape of the input array is incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        return self.bin_transform.dot(data.ravel())

    def bin_covar(self, covar):
        """
        Calculate the covariance in the binned data provided the unbinned
        covariance.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance in the unbinned data.

        Returns:
            `scipy.sparse.csr_matrix`_: Covariance in the binned data.
        """
        return self.bin_transform.dot(covar.dot(self.bin_transform.T))

    def remap(self, data, mask=None, masked=True, fill_value=0):
        """
        Provided the vector of binned data, reconstruct the 2D map filling each
        pixel in the same bin with the binned value.

        Args:
            data (`numpy.ndarray`_):
                The data to remap.  Shape must match the number of unique bin
                IDs.
            mask (`numpy.ndarray`_, optional):
                Boolean mask with the same shape as ``data``.  If None, all data
                are unmasked.
            masked (:obj:`bool`, optional):
                Return data as a masked array, where any pixel not associated
                with a bin is masked (in addition to the provided ``mask``.)
            fill_value (scalar-like, optional):
                Value used to fill the masked pixels, if a masked array is
                *not* requested. Warning: The value is automatically
                converted to be the same data type as the input array or
                attribute.

        Returns:
            `numpy.ndarray`_, `numpy.ma.MaskedArray`_: 2D array with
            the data remapped to a 2D array.

        Raises:
            ValueError:
                Raised if shape of ``data`` does not match the expected 1d
                shape.

        """
        # Check the shapes
        if data.shape != self.nbin.shape:
            raise ValueError('To remap, data must have the same shape as the internal data '
                             'attributes: {0}'.format(self.nbin.shape))
        if mask is not None and mask.shape != self.nbin.shape:
            raise ValueError('To remap, mask must have the same shape as the internal data '
                             'attributes: {0}'.format(self.nbin.shape))

        # Construct the output map
        # NOTE: np.ma.masked_all sets the initial data array to 2.17506892e-314,
        # which just leads to trouble. I've instead used the line below to make
        # sure that the initial value is just 0.
        _data = np.ma.MaskedArray(np.zeros(self.spatial_shape, dtype=data.dtype), mask=True)
        _data[self.unravel] = data[self.bin_inverse]
        if mask is not None:
            np.ma.getmaskarray(_data)[self.unravel] = mask[self.bin_inverse]
        # Return a masked array if requested; otherwise, fill the masked values
        # with the type-cast fill value.
        return _data if masked else _data.filled(_data.dtype.type(fill_value))

    def remap_covar(self, covar):
        """
        Remap a covariance matrix from the binned data to the unbinned map.

        Pixels in the same bin are perfectly correlated.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix to remap.

        Returns:
            `scipy.sparse.csr_matrix`_: A sparse matrix for the remapped
            covariance.
        """
        gpm = np.ones(self.nbin, dtype=bool) if self.ubinid is None \
                    else self.remap(self.ubinid, masked=False, fill_value=-1) > -1
        _bt = self.bin_transform[:,gpm.ravel()].T
        _bt[_bt > 0] = 1.
        return fill_matrix(_bt.dot(covar.dot(_bt.T)), gpm.ravel())

    # TODO: Include error calculations?
    def bin_moments(self, norm, center, stddev):
        r"""
        Bin a set of Gaussian moments.

        Assuming the provided data are the normalization, mean, and standard
        deviation of a set of Gaussian profiles, this method performs a nominal
        calculation of the moments of the summed Gaussian profile.

        .. note::

            Any of the input arguments can be None, but at least one of them
            cannot be!

        Args:
            norm (`numpy.ndarray`_):
                Gaussian normalization.  Shape must match :attr:`spatial_shape`.
            center (`numpy.ndarray`_):
                Gaussian center.  Shape must match :attr:`spatial_shape`.
            stddev (`numpy.ndarray`_):
                Gaussian standard deviation.  Shape must match
                :attr:`spatial_shape`.

        Returns:
            :obj:`tuple`: A tuple of three `numpy.ndarray`_ objects with the
            binned normalization, mean, and standard deviation of the summed
            profile.  If ``norm`` is None on input, the returned 0th moment is 1
            everywhere.  If ``center`` is None on input, the returned 1st moment
            is 0 everywhere.  If ``stddev`` is None on input, the returned 2nd
            moment is 1 everywhere.
        """
        if all([x is None for x in [norm, center, stddev]]):
            raise ValueError('At least one parameter must not be None.')
        shape = [x.shape for x in [norm, center, stddev] if x is not None][0]
        if not all([x is None or x.shape == shape for x in [norm, center, stddev]]):
            raise ValueError('Shape of all input arrays must match.')
        mom0 = np.ones(shape, dtype=float) if norm is None else self.bin(norm)
        inv_mom0 = 1/(mom0 + (mom0 == 0.))
        _center = np.zeros(shape, dtype=float) if center is None else center
        mom1 = self.bin(norm*_center) * inv_mom0
        if stddev is None:
            return mom0, mom1, np.ones(shape, dtype=float)
        _mom2 = _center**2 + stddev**2
        return mom0, mom1, np.sqrt(self.bin(norm*_mom2) * inv_mom0 - mom1**2)


