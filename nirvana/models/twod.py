"""
Implements two-dimensional functions for modeling.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import special
from astropy.modeling import functional_models

class Sersic2D(functional_models.Sersic2D):
    """
    A 2D Sersic distribution.

    Args:
        sb_eff (scalar-like):
            The surface brightness at 1 effective (half-light) radius.
        r_eff (scalar-like):
            The effective (half-light) radius in *arcseconds*.
        n (scalar-like):
            The Sersic index.
        center (scalar-like, optional):
            The coordinates of the Sersic center in *arcseconds* relative to the
            image center.
        ellipticity (scalar-like, optional):
            The ellipticity (1-b/a) of an elliptical Sersic distribution.
        position_angle (scalar-like, optional):
            The position angle for the elliptical Sersic distribution, defined
            as the angle from N through E in degrees.  The coordinate system is
            defined with positive offsets (in RA) toward the east, meaning lower
            pixel indices.
        unity_integral (:obj:`bool`, optional):
            Renormalize the distribution so that the integral is unity.
    """
    def __init__(self, sb_eff, r_eff, n, center=[0,0], ellipticity=1.0, position_angle=0.,
                 unity_integral=False):

        self.position_angle = position_angle
        super().__init__(amplitude=sb_eff, r_eff=r_eff, n=n, x_0=center[0], y_0=center[1],
                         ellip=ellipticity, theta=np.radians(90-self.position_angle))

        self.bn = None
        self.integral = self.get_integral()
        
        if unity_integral:
            self.amplitude /= self.integral
            self.integral = self.get_integral()

    def get_integral(self):
        """
        The analytic integral of the Sersic profile projected on the
        sky.
        """
        # Note the (1-ellipticity) factor.
        self.bn = special.gammaincinv(2. * self.n, 0.5)
        return 2 * np.pi * self.n * np.exp(self.bn) * self.amplitude \
                            * np.square(self.r_eff) * (1-self.ellip) \
                            * special.gamma(2*self.n) * np.power(self.bn, -2*self.n)



