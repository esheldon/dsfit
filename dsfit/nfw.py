import numpy as np
from .util import m200_to_r200


class NFW(object):
    """

    Evaluate DeltaSigma or rho for an nfw.

    This has uses fixed omega_m from the input cosmology pars, and z, entered
    on construction.
    """

    def __init__(self, *, cosmo, z, m200, c):
        self.cosmo = cosmo

        self.z = z
        self.c = c

        # convert to Mpc^3
        self.rhomean = self.cosmo.rho_m(self.z) * 1000.0**3

        self.m200 = m200
        self.r200 = m200_to_r200(m200=m200, rho=self.rhomean)

        # buffer region around rs, linearly interpolate through
        # region near rs
        self.ep = 0.001

    def get_dsig(self, r):
        """
        Get DeltaSigma in units of Msolar/pc^2

        Parameters
        ----------
        r: array
            radii in Mpc

        Returns
        -------
        DeltaSigma: array
            Same shape as r
        """
        from numpy import log, arctan, arctanh, sqrt

        ep = self.ep

        c = self.c

        rs = self.r200 / c
        xx = r / rs  # = c*r/r200
        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))

        # fac = rs * del_c * self.rhocrit
        fac = rs * del_c * self.rhomean

        w1, = np.where(xx < (1 - ep))
        w2, = np.where((xx >= (1 - ep)) & (xx <= (1 + ep)))
        w3, = np.where(xx > (1 + ep))

        dsig = np.zeros(r.size, dtype="f8")

        if w1.size > 0:
            x = xx[w1]
            x2 = x ** 2
            # print(x)
            A = arctanh(sqrt((1 - x) / (1 + x)))
            s = (
                8 * A / (x2 * sqrt(1 - x2))
                + 4 * log(x / 2) / x2
                - 2 / (x2 - 1)
                + 4 * A / ((x2 - 1) * sqrt(1 - x2))
            )

            dsig[w1] = s

        # interpolate between the two regions
        if w2.size > 0:

            e = 1 - ep
            e2 = e ** 2
            A = arctanh(sqrt((1 - e) / (1 + e)))
            s1 = (
                8 * A / (e2 * sqrt(1 - e2))
                + 4 * log(e / 2) / e2
                - 2 / (e2 - 1)
                + 4 * A / ((e2 - 1) * sqrt(1 - e2))
            )

            e = 1 + ep
            e2 = e ** 2
            A = arctan(sqrt((e - 1) / (1 + e)))
            s2 = (
                8 * A / (e2 * sqrt(e2 - 1))
                + 4 * log(e / 2) / (e2)
                - 2 / (e2 - 1)
                + 4 * A / ((e2 - 1) ** (3 / 2.0))
            )

            e1 = 1 - ep
            e2 = 1 + ep
            x = xx[w2]
            s = (x - e1) * s2 / (e2 - e1) + (x - e2) * s1 / (e1 - e2)

            dsig[w2] = s

        if w3.size > 0:
            x = xx[w3]
            x2 = x ** 2

            A = arctan(sqrt((x - 1) / (1 + x)))
            s = (
                8 * A / (x2 * sqrt(x2 - 1))
                + 4 * log(x / 2) / x2
                - 2 / (x2 - 1)
                + 4 * A / ((x2 - 1) ** (3 / 2.0))
            )
            dsig[w3] = s

        dsig *= fac

        # convert to Msolar/pc^2
        return dsig / 1.e6**2

    def get_rho(self, r):
        from numpy import log

        c = self.c

        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))
        rs = self.r200 / c
        x = r / rs

        # rho = del_c * self.rhocrit / x / (1 + x) ** 2
        rho = del_c * self.rhomean / x / (1 + x) ** 2
        return rho

    def get_mass(self, r):
        """
        Mass less than radius r in solar masses
        r and r200 in Mpc.
        """
        from numpy import log

        c = self.c

        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))
        rs = self.r200 / c
        x = r / rs

        rhomean = self.rhomean

        m = 4 * np.pi * del_c * rhomean * rs ** 3 * (log(1 + x) - x / (1 + x))  # noqa
        return m * 1.0e12

    def plot_rho(self, rmin=0.01, rmax=20.0, npts=1000):
        import hickory

        r = np.logspace(np.log10(rmin), np.log10(rmax), npts)
        rho = self.get_rho(r)

        plt = hickory.Plot()
        plt.curve(r, rho)
        plt.set_xscale('log')
        plt.set_yscale('log')

        plt.show()

    def plot_m(self, rmin=0.01, rmax=20.0, npts=1000):
        import hickory

        r = np.logspace(np.log10(rmin), np.log10(rmax), npts)
        mass = self.get_mass(r)

        plt = hickory.Plot()
        plt.curve(r, mass)
        plt.set_xscale('log')
        plt.set_yscale('log')

        plt.show()

    def plot_dsig(self, rmin=0.01, rmax=20.0, npts=1000):
        import hickory

        r = np.logspace(np.log10(rmin), np.log10(rmax), npts)
        dsig = self.get_dsig(r)

        plt = hickory.Plot()
        plt.curve(r, dsig)
        plt.set_xscale('log')
        plt.set_yscale('log')

        plt.show()
