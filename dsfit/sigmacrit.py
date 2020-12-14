"""
ScinvCalculator, a class to calculate the mean inverse
critical density as a function of lens redshift from
a source p(z)

Erin Sheldon, BNL
"""
from __future__ import print_function
import os
import numpy
import esutil
from esutil import cosmology as cosmo
from esutil.ostools import path_join
from esutil.numpy_util import where1
from math import pi as PI


class ScinvCalculator:
    def __init__(
        self,
        zlmin,
        zlmax,
        nzl,
        zsmin,
        zsmax,
        npts=100,
        omega_m=0.3,
        omega_l=0.7,
        omega_k=0.0,
        H0=100.0,
        flat=True,
    ):
        """
        Specialized calculator for integrating over a p(zs) to get a mean
        inverse critical density on a pre-determined grid of zlens.

        parameters
        ----------
        zlmin: float
            Minimum zlens point
        zlmax: float
            Maximum zlens point
        nzl: int
            Number of points in zlens grid
        zsmin: float
            Min val over which to integrate p(zs)
        zsmax: float
            Max val over which to integrate p(zs)
        npts: int, optional
            Number of points for integration over p(zs)
        omega_m: float, optional
            default 0.3
        omega_l: float, optional
            default 0.7
        omega_k: float, optional
            default 0.0
        flat: bool, optional
            default True, flat universe

        NOTE: npts for the distance calculations is always 5, which is
            good to 1.e-8 to redshift 1

        usage:
            nzl=65
            zlmin=0.09
            zlmax=0.95
            zsmin=0.0
            zsmax=1.5
            npts=100 # integration of p(z)
            scalc=ScinvCalculator(zlmin, zlmax, nzl, zsmin, zsmax, npts=npts)

            mean_scinv = scalc.calc_mean_scinv(zs, pzs)
        """

        self.omega_m = omega_m
        self.omega_l = omega_l
        self.omega_k = omega_k
        self.flat = flat
        self.H0 = H0

        self.setup_cosmology()
        self.setup_gauleg(zsmin, zsmax, npts)
        self.setup_zl(zlmin, zlmax, nzl)
        self.setup_scinv_grid()

    def calc_mean_scinv(self, zs, pz):
        """
        pz must correspond exactly to the zsvals input

        pz will be interpolated onto the gauss-legendra
        abcissa

        """

        if pz.size != zs.size:
            raise ValueError(
                "pz(%d) and zs(%d) must be same " "size" % (pz.size, zs.size)
            )

        mean_scinv = numpy.zeros(self.nzl, dtype="f8")

        # get p(z) interpolated to the gauleg integration points
        pzinterp = self.interpolate_pofz(zs, pz)

        for i in range(self.nzl):
            # we've pre-computed scinv at zl,zs locations of relevance
            # now just multiply by the interpolated p(z) and do the
            # integral
            numerator = self.f1 * self.scinv[i, :] * pzinterp * self.wii
            denom = self.f1 * pzinterp * self.wii

            mean_scinv[i] = numerator.sum() / denom.sum()

        return mean_scinv

    def interpolate_pofz(self, z, pz):
        """
        interpolate p(z) onto the points used for gauleg integration
        """

        if self.zsmin < z[0] or self.zsmax > z[-1]:
            tup = (z[0], z[-1], self.zsmin, self.zsmax)
            raise ValueError(
                "attempt to interpolate outside of range " "[%g,%g] <-> [%g,%g] " % tup
            )

        pzvals = esutil.stat.interplin(pz, z, self.zsvals_int)

        pzvals.clip(min=0.0, out=pzvals)
        return pzvals

    def setup_cosmology(self):
        """
        Create the cosmology object used for sigmacrit calculations
        """
        self.cosmo = cosmo.Cosmo(
            omega_m=self.omega_m,
            omega_l=self.omega_l,
            omega_k=self.omega_k,
            H0=self.H0,
            flat=self.flat,
        )

    def setup_scinv_grid(self):
        """
        Set up the pre-computed grid of scinv(zl) over which we will interpolate
        """
        self.scinv = numpy.zeros((self.nzl, self.npts), dtype="f8")

        c = self.cosmo
        for i in range(self.nzl):
            zl = self.zlvals[i]
            self.scinv[i, :] = c.sigmacritinv(zl, self.zsvals_int)

    def setup_zl(self, zlmin, zlmax, nzl):
        """
        the points where inverse sigmacrit will be evaluated
        """
        self.zlmin = zlmin
        self.zlmax = zlmax
        self.nzl = nzl
        self.zlvals = numpy.linspace(zlmin, zlmax, nzl)

    def setup_gauleg(self, zsmin, zsmax, npts):
        """
        set up the gauss-legendre weights and x vals used for integration over
        zs
        """

        self.zsmin = zsmin
        self.zsmax = zsmax
        self.npts = npts

        self.xii, self.wii = esutil.integrate.gauleg(-1.0, 1.0, npts)

        self.f1 = (zsmax - zsmin) / 2.0
        self.f2 = (zsmax + zsmin) / 2.0

        # the gauss-legendre integration points: must interpolate the
        # input p(zs) to this grid
        self.zsvals_int = self.xii * self.f1 + self.f2
