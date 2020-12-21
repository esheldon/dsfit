import numpy as np
from . import project


class Linear(object):
    """
    Compute delta_sigma(r) for the linear correlation function
    """

    def __init__(self, *, cosmo, z, b):
        self.cosmo = cosmo
        self.z = z
        self.b = b

        # convert to Mpc^3
        self.rhomean = self.cosmo.rho_m(self.z) * 1000.0**3

    def get_dsig(self, r):
        """
        Get DeltaSigma in units of Msolar/pc^2

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            radii in Mpc
        b: float, optional
            bias

        Returns
        -------
        DeltaSigma: array
            Same shape as r
        """

        drho = self.get_drho(r)

        sig = project.project3d(r, drho)
        dsig = self.sigma2dsig(r=r, sig=sig)

        return dsig

    def get_drho(self, r):
        """
        calculate rho_matter(z) * xi

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            Radii in comoving Mpc at which to evaluate
        b: float, optional
            Bias term, default 1.0

        Returns
        -------
        drho: array
            rho_matter(z) * xi
        """

        xi = self.get_xi(r)
        return self.rhomean * xi * self.b

    def get_xi(self, r):
        """
        Calculate the linear matter-matter correlation function

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            Radii in comoving Mpc at which to evaluate

        Returns
        -------
        xi: array
            xi values at the requested radii
        """
        return self.cosmo.correlationFunction(r, self.z)

    def get_j3(self, r):
        """
        This is proportional to the mass enclosed.  Mass in units
        of 10^12 is then rhomean * j3 * 1.0e12

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            Radii in comoving Mpc at which to evaluate

        Returns
        -------
        j3: array
            same size as r
        """
        from numpy import log, roll

        xi = self.xi(r)

        w, = np.where(xi <= 0)
        if w.size > 0:
            raise ValueError(
                "all xi must be > 0 for power law " "interpolation",
            )

        lr = log(r)
        lxi = log(xi)

        al = (lxi - roll(lxi, 1)) / (lr - roll(lr, 1))
        al[0] = al[1]

        A = xi / r ** (al)

        ex = 3.0 + al
        Rb = r
        Ra = roll(r, 1)
        Ra[0] = 0.0
        int0 = A * (1.0 / ex) * (Rb ** ex - Ra ** ex)

        j3 = 4 * np.pi * int0.cumsum()
        return j3

    def m(self, r):
        """
        The mass enclosed.

        Parameters
        ----------
        z: float
            Redshift at which to evaluate
        r: array
            Radius in comoving Mpc at which to evaluate
        b: float, optional
            Bias term, default 1

        Returns
        -------
        mass: array
            same size as r
        """

        j3 = self.get_j3(r)
        return j3 * self.rhomean * self.b * 1.0e3

    def sigma2dsig(self, *, r, sig):
        """

        Compute DeltaSigma from 2D sigma. Assumes power law
        interpolation and powerlaw extrapolation

        """
        from numpy import log, roll

        # slopes
        sigfix = sig.clip(min=1.0e-5)
        al = log(sigfix / roll(sigfix, 1)) / log(r / roll(r, 1))
        al[0] = al[1]
        slope_min = -1.95
        if al[0] < slope_min:
            print("Warning: profile too steep to converge: ", al[0])
            print("truncating to inner slope of: ", slope_min)
            al[0] = slope_min

        A = sig / (r ** al)

        RA = roll(r, 1)
        RA[0] = 0.0
        RB = r

        Ints = A / (al + 2.0) * (RB ** (al + 2.0) - RA ** (al + 2.0))
        Icum = Ints.cumsum()
        avg_sig = 2.0 * Icum / (r ** 2)

        dsig = avg_sig - sig

        # convert to Msolar/pc^2
        return dsig / 1.e6**2

    def xigen(self, rmin=0.01, rmax=50.0, npts=1000):
        """
        Generate some xi values on log spaced grid
        """
        r = np.logspace(np.log10(rmin), np.log10(rmax), npts)
        return r, self.get_xi(r)

    def plot_xi(self, r, xi):
        import hickory

        minval = 1.0e-4

        xi = np.where(xi < minval, minval, xi)

        plt = hickory.Plot()
        plt.set(
            xlabel=r"$r [Mpc/h]$",
            ylabel=r"$\xi_{lin}(r)$",
        )

        plt.set_xscale('log')
        plt.set_yscale('log')
        plt.curve(r, xi)

        plt.show()

    def plot_xigen(self):
        r, xi = self.xigen()
        self.plot_xi(r, xi)


# convergence test for xi with the number of p(k) sample
# points used.
def test_xi_converge_nplk(epsfile=None):
    """
    Test how xi converges with the number of k points per log10(k)
    Note we should test other convergence factors too!
    """
    import biggles
    tab = biggles.Table(2, 1)
    pltbig = biggles.FramedPlot()
    pltzoom = biggles.FramedPlot()

    pltbig.xlabel = "r"
    pltbig.ylabel = "xi(r)"
    pltbig.xlog = True
    pltbig.ylog = True
    pltzoom.xlabel = "r"
    pltzoom.ylabel = "xi(r)"

    lin = Linear()
    r = 10.0 ** np.linspace(0.0, 2.3, 1000)
    nplk_vals = [20, 60, 100, 140, 160]
    color_vals = ["blue", "skyblue", "green", "orange", "magenta", "red"]

    plist = []
    lw = 2.4
    for nplk, color in zip(nplk_vals, color_vals):
        print("nplk:", nplk)
        xi = lin.xi(r, nplk=nplk)

        limxi = np.where(xi < 1.0e-5, 1.0e-5, xi)
        climxi = biggles.Curve(r, limxi, color=color, linewidth=lw)
        climxi.label = "nplk: %i" % nplk
        pltbig.add(climxi)

        plist.append(climxi)

        w, = np.where(r > 50.0)
        cxi = biggles.Curve(r[w], xi[w], color=color, linewidth=lw)
        pltzoom.add(cxi)

    key = biggles.PlotKey(0.7, 0.8, plist)
    pltzoom.add(key)
    tab[0, 0] = pltbig
    tab[1, 0] = pltzoom
    if epsfile is not None:
        tab.write_eps(epsfile)
    else:
        tab.show()
