import numpy as np


class NFW(object):
    """

    Evaluate DeltaSigma or rho for an nfw.

    This has uses fixed omega_m from the input cosmology pars, and z, entered
    on construction.
    """

    def __init__(self, *, cosmo):
        self.cosmo = cosmo

        # buffer region around rs, linearly interpolate through
        # region near rs
        self.ep = 0.001

    def dsig(self, *, z, r, r200, c):
        """
        Get DeltaSigma in units of Msolar/pc^2

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            radii in Mpc
        r200: array
            r200 mean in Mpc
        c: float
            concentration

        Returns
        -------
        DeltaSigma: array
            Same shape as r
        """
        from numpy import log, arctan, arctanh, sqrt

        # convert to Mpc^2
        rhomean = self.cosmo.rho_m(z) * 1000.0**3

        # print('r200:', r200, 'c:', c)
        # if r200 < 0 or c < 0:
        #     return r*0 - 9999

        ep = self.ep

        rs = r200 / c
        xx = r / rs  # = c*r/r200
        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))

        # fac = rs * del_c * self.rhocrit
        fac = rs * del_c * rhomean

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

    def rho(self, *, z, r, r200, c):
        from numpy import log

        rhomean = self.cosmo.rho_m(z)

        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))
        rs = r200 / c
        x = r / rs

        # rho = del_c * self.rhocrit / x / (1 + x) ** 2
        rho = del_c * rhomean / x / (1 + x) ** 2
        return rho

    def plot_rho(self, *, z, r200, c):
        from biggles import FramedPlot, Curve

        n = 1000
        r = np.linspace(0.01, 20.0, n)
        rho = self.rho(z=z, r=r, r200=r200, c=c)

        plt = FramedPlot()
        plt.add(Curve(r, rho))
        plt.xlog = True
        plt.ylog = True

        plt.show()

    def m(self, *, z, r, r200, c):
        """
        Mass less than radius r in solar masses
        r and r200 in Mpc.
        """
        from numpy import log
        rhomean = self.cosmo.rho_m(z)

        del_c = (200 / 3.0) * c ** 3 / (log(1.0 + c) - c / (1.0 + c))
        rs = r200 / c
        x = r / rs

        m = 4 * np.pi * del_c * rhomean * rs ** 3 * (log(1 + x) - x / (1 + x))  # noqa
        return m * 1.0e12

    def m200(self, *, z, r200):
        """

        Gives mass in solar masses for r200 in Mpc

        Same as puttin r=r200 in the .m() method
        Note independent of c

        """

        # Msolar/kpc^3
        rhomean = self.cosmo.rho_m(z)

        # convert to Mpc^3
        rhomean = rhomean * 1000.0**3

        m200 = 200 * rhomean * (4.0 / 3.0) * np.pi * r200 ** 3
        return m200

    def plot_m(self, *, z, r200, c):
        from biggles import FramedPlot, Curve

        n = 1000
        r = np.linspace(0.01, 20.0, n)
        m = self.m(z=z, r=r, r200=r200, c=c)

        plt = FramedPlot()
        plt.add(Curve(r, m))
        plt.xlog = True
        plt.ylog = True

        plt.show()

    def plot_dsig(self, *, z, r200, c):
        from biggles import FramedPlot, Curve

        n = 1000
        r = np.linspace(0.01, 20.0, n)
        ds = self.dsig(z=z, r200=r200, c=c)

        plt = FramedPlot()
        plt.add(Curve(r, ds))
        plt.xlog = True
        plt.ylog = True

        plt.show()
