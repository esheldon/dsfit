import numpy as np
from . import nfw
from . import linear
from colossus.cosmology.cosmology import (
    Cosmology,
    setCosmology,
)


class NFWBiasFitter(object):
    def __init__(
        self,
        *,
        z,
        r,
        cosmo='planck18',
        withlin=True,
    ):
        """
        Fit NFW + linear to delta sigma

        Parameters
        ----------
        z: float
            redshift of lens
        r: array
            Array of radii, Mpc
        cosmo: bool or Cosmology object from colossus
            The cosmology to use, e.g.  'planck18' or Cosmology
            object
        withlin: bool
            Include the linear bias term.  Default True.
        """

        if not isinstance(cosmo, Cosmology):
            cosmo = setCosmology(cosmo)

        self.z = z
        self.r_ = r.copy()
        self.cosmo = cosmo

        self.withlin = withlin

        self.nfw = nfw.NFW(cosmo=self.cosmo)

        if self.withlin:
            self.lin = linear.Linear(cosmo=cosmo)
            # print("pre-computing linear dsig at %s points" % r.size)
            self.lin_dsig_ = self.lin.dsig(z=z, r=self.r_, b=1.0)

    def fit(self, *, dsig, dsigcov, guess,
            c_bounds=[0.1, 10.0],
            r200_bounds=[0.1, 10.0],
            b_bounds=[0.2, 3.0]):
        """
        Class:
            NFWBiasFitter
        Method:
            fit
        Usage:
            fitter.fit(dsig, dsigcov, guess, more=False)

        Inputs:
            dsig, dsigcov: Must correspond to the radii given
                on construction.
            guess: [r200, c, b] or [r200, c] if withlin=False

        Outputs:
            a dict with  'p' and 'cov' along with
                r200,r200_err
                c, c_err
                b, b_err  (-9999 if withlin=False)
                m200, m200_err
            plus their errors

        """
        from scipy.optimize import curve_fit

        if (
            dsig.size != self.r_.size
            or dsigcov.shape[0] != self.r_.size
            or dsigcov.shape[0] != self.r_.size
        ):
            raise ValueError("dsig, dsigcov must be same size as r")

        if self.withlin:
            npar = 3
            func = self._dsig_nfw_lin
        else:
            func = self._dsig_nfw
            npar = 2

        if len(guess) != npar:
            raise ValueError("parameter guess must have %s elements" % npar)

        # print("running curve_fit")
        bounds = (
            # np.array([0.05, 0.1, 0.1]),
            # np.array([5.00, 5.0, 3.0])
            np.array([r200_bounds[0], c_bounds[0], b_bounds[0]]),
            np.array([r200_bounds[1], c_bounds[1], b_bounds[1]])
        )
        print('bounds:', bounds)
        p, cov = curve_fit(
            func,
            self.r_,
            dsig,
            sigma=dsigcov,
            p0=guess,
            bounds=bounds,
        )
        return self.pars2more(p, cov)

    def _dsig_nfw(self, r, r200, c):
        """
        for fitting, accepts non-keywords pars
        """
        return self.nfw_dsig(r=r, r200=r200, c=c)

    def _dsig_nfw_lin(self, r, r200, c, b):
        """
        for fitting, accepts non-keywords pars
        """
        assert self.withlin
        return self.dsig(r=r, r200=r200, c=c, b=b)

    def dsig(self, *, r, r200, c, b=None):
        """
        get delta sigma for the input data

        called from curve_fit so must accept pars as non-keyword
        """
        if self.withlin:
            assert b is not None, "send b for withlin"
            nfw_dsig = self.nfw_dsig(r=r, r200=r200, c=c)
            lin_dsig = self.lin_dsig(r=r, b=b)
            return nfw_dsig + lin_dsig

        else:
            return self.nfw_dsig(r=r, r200=r200, c=c)

    def nfw_dsig(self, *, r, r200, c):
        return self.nfw.dsig(z=self.z, r=r, r200=r200, c=c)

    def lin_dsig(self, *, r, b):
        """
        r values must coinciced with linear dsig values
        """
        if r.size != self.lin_dsig_.size:
            raise ValueError("r and dsig must be same size")

        return b * self.lin_dsig_

    def pars2more(self, p, cov):
        r200 = p[0]
        r200_err = np.sqrt(cov[0, 0])
        c = p[1]
        c_err = np.sqrt(cov[1, 1])

        m200 = self.nfw.m200(z=self.z, r200=r200)
        m200_err = 3 * m200 * r200_err / r200

        res = {
            "p": p,
            "cov": cov,
            "r200": r200,
            "r200_err": r200_err,
            "c": c,
            "c_err": c_err,
            "m200": m200,
            "m200_err": m200_err,
        }

        if self.withlin:
            res["b"] = p[2]
            res["b_err"] = np.sqrt(cov[2, 2])

        return res


def plot(
    *,
    r,
    z,
    r200,
    c,
    b=None,
    dsig=None,
    dsigcov=None,
    xlim=None,
    ylim=None,
    plt=None,
    show=False,
):

    import hickory

    if b is None:
        withlin = False
    else:
        withlin = True

    fitter = NFWBiasFitter(z=z, r=r, withlin=withlin)
    nfw_dsig = fitter.nfw_dsig(r=r, r200=r200, c=c)
    print('nfw m200: %e' % fitter.nfw.m200(z=0.3, r200=0.2))
    print('nfw:', nfw_dsig)
    yfit = fitter.dsig(r=r, r200=r200, c=c, b=b)
    print('yfit:', yfit)

    if dsig is not None or dsigcov is not None:
        assert dsigcov is not None, "send both dsig and dsigcov"
        assert dsig is not None, "send both dsig and dsigcov"
        dsigerr = np.sqrt(np.diag(dsigcov))

        ymin, ymax = [
            0.5 * (dsig - dsigerr).min(),
            1.5 * (dsig + dsigerr).max(),
        ]
        if ymin < 0:
            ymin = 0.1
    else:
        ymin, ymax = [0.5 * yfit.min(), 1.5 * yfit.max()]

    if ylim is None:
        ylim = [0.5*ymin, 1.5*ymax]

    if xlim is None:
        rmin, rmax = r.min(), r.max()
        xlim = [0.5 * rmin, 1.5 * rmax]

    if plt is None:
        dolegend = True
        plt = hickory.Plot(
            xlabel=r"$r$ [$h^{-1}$ Mpc]",
            ylabel=r"$\Delta\Sigma ~ [h \mathrm{M}_{\odot} \mathrm{pc}^{-2}]$",
            xlim=xlim,
            ylim=ylim,
        )
        plt.set_xscale('log')
        plt.set_yscale('log')
    else:
        dolegend = False

    alpha = 0.5
    if dsig is not None:
        plt.errorbar(r, dsig, dsigerr, alpha=alpha)

    plt.curve(r, yfit, label='model')
    if withlin:
        yfit_lin = fitter.lin_dsig(r=r, b=b)
        yfit_nfw = fitter.nfw_dsig(r=r, r200=r200, c=c)
        plt.curve(r, yfit_nfw, label='nfw')
        plt.curve(r, yfit_lin, label='linear')

    if dolegend:
        plt.legend()

    if show:
        plt.show()

    return plt


def fit_nfw_lin_dsig(
    *,
    z, r, dsig, dsigcov, guess,
    cosmo_pars=None, withlin=True,
):
    """
    Name:
        fit_nfw_line_dsig
    Calling Sequence:
        fit_nfw_lin_dsig(omega_m, z, r, ds, dscov, guess,
                         **kwds)
    Inputs:
        omega_m:
        z: the redshift
        r: radius in Mpc
        ds: delta sigma in pc^2/Msun
        dscov: cov matrix
        guess: guesses for [r200,c,b]

    Keywords:
        withlin: True.  Include the linear bias term.
        more: Controls the output.  See Outputs section below.

        Keywords for the Linear() class. See the Linear() class for the
        defaults
            omega_b
            sigma_8
            h
            ns
    Outputs:
        if more=False:
            p,cov:  The parameters array and the covariance arra
        if more=True:
            a dict with  'p' and 'cov' along with
                r200,r200_err
                c, c_err
                b, b_err
                m200, m200_err
        plus their errors

    """

    fitter = NFWBiasFitter(z=z, r=r, cosmo_pars=cosmo_pars, withlin=withlin)
    return fitter.fit(dsig=dsig, dsigcov=dsigcov, guess=guess)


def fit_nfw_dsig(omega_m, z, r, ds, dserr, guess, rhofac=180):
    """
    Name:
        fit_nfw_dsig
    Calling Sequence:
        fit_nfw_dsig(omega_m, z, r, ds, dserr, guess)
    Inputs:
        omega_m:
        z: the redshift
        r: radius in Mpc
        ds: delta sigma in pc^2/Msun
        dserr: error
        guess: guesses for [r200,c]

        rhofac=180 for calculation mass relative to
            rhofac times mean

    Outputs:
        This returns a dict with:

            r200
            c
            m200

        plus errors and a covariance between r200 and c

        A mass relative to rhofac*rhmean is also computed.

    """
    from scipy.optimize import curve_fit

    n = nfw.NFW(omega_m, z)
    p, cov = curve_fit(n.dsig, r, ds, sigma=dserr, p0=guess)
    r200 = p[0]
    r200_err = np.sqrt(cov[0, 0])
    c = p[1]
    c_err = np.sqrt(cov[1, 1])

    m200 = n.m200(r200)
    m200_err = 3 * m200 * r200_err / r200

    # get rhofac times mean
    rm = n.r_fmean(r200, c, rhofac)
    rm_err = r200_err * (rm / r200)
    mm = n.m(rm, r200, c)
    mm_err = m200_err * (mm / m200)

    rtag = "r%sm" % rhofac
    mtag = "m%sm" % rhofac

    res = {
        "r200": r200,
        "r200_err": r200_err,
        "c": c,
        "c_err": c_err,
        "cov": cov,
        "m200": m200,
        "m200_err": m200_err,
        rtag: rm,
        rtag + "_err": rm_err,
        mtag: mm,
        mtag + "_err": mm_err,
    }
    return res


def test_fit_nfw_lin_dsig(rmin=0.01):

    z = 0.25

    r200 = 1.0
    c = 5.0
    b = 10.0

    rmax = 50.0
    log_rmin = np.log10(rmin)
    log_rmax = np.log10(rmax)
    npts = 30
    r = 10.0 ** np.linspace(log_rmin, log_rmax, npts)

    fitter = NFWBiasFitter(z=z, r=r)
    ds = fitter.dsig(r, r200, c, b)
    # 10% errors
    dserr = 0.1 * ds
    ds += dserr * np.random.standard_normal(ds.size)
    dscov = np.diag(dserr)

    guess = np.array([r200, c, b], dtype="f8")
    # add 10% error to the guess
    guess += 0.1 * guess * np.random.standard_normal(guess.size)

    res = fitter.fit(
        dsig=ds,
        dsigcov=dscov,
        guess=guess,
    )

    r200_fit = res["r200"]
    r200_err = res["r200_err"]
    c_fit = res["c"]
    c_err = res["c_err"]
    b_fit = res["b"]
    b_err = res["b_err"]

    print("Truth:")
    print("    r200: %f" % r200)
    print("       c: %f" % c)
    print("       b: %f" % b)
    print("r200_fit: %f +/- %f" % (r200_fit, r200_err))
    print("   c_fit: %f +/- %f" % (c_fit, c_err))
    print("   b_fit: %f +/- %f" % (b_fit, b_err))
    print("Cov:")
    print(res["cov"])

    plot(
        r=r,
        z=z,
        r200=r200_fit,
        c=c_fit,
        b=b_fit,
        dsig=ds,
        dsigcov=dscov,
        show=True,
    )


def test_fit_nfw_dsig(rmin=0.01):

    from biggles import FramedPlot, Points, SymmetricErrorBarsY, Curve

    omega_m = 0.25
    z = 0.25
    n = nfw.NFW(omega_m, z)

    r200 = 1.0
    c = 5.0

    rmax = 5.0
    log_rmin = np.log10(rmin)
    log_rmax = np.log10(rmax)
    npts = 25
    logr = np.linspace(log_rmin, log_rmax, npts)
    r = 10.0 ** logr

    ds = n.dsig(r, r200, c)
    # 10% errors
    dserr = 0.1 * ds
    ds += dserr * np.random.standard_normal(ds.size)

    guess = np.array([r200, c], dtype="f8")
    # add 10% error to the guess
    guess += 0.1 * guess * np.random.standard_normal(guess.size)

    res = fit_nfw_dsig(omega_m, z, r, ds, dserr, guess)

    r200_fit = res["r200"]
    r200_err = res["r200_err"]

    c_fit = res["c"]
    c_err = res["c_err"]

    print("Truth:")
    print("    r200: %f" % r200)
    print("       c: %f" % c)
    print("r200_fit: %f +/- %f" % (r200_fit, r200_err))
    print("   c_fit: %f +/- %f" % (c_fit, c_err))
    print("Cov:")
    print(res["cov"])

    logr = np.linspace(log_rmin, log_rmax, 1000)
    rlots = 10.0 ** logr
    yfit = n.dsig(rlots, r200_fit, c_fit)

    plt = FramedPlot()
    plt.add(Points(r, ds, type="filled circle"))
    plt.add(SymmetricErrorBarsY(r, ds, dserr))
    plt.add(Curve(rlots, yfit, color="blue"))

    plt.xlabel = r"$r$ [$h^{-1}$ Mpc]"
    plt.ylabel = r"$\Delta\Sigma ~ [M_{sun} pc^{-2}]$"

    plt.xrange = [0.5 * rmin, 1.5 * rmax]
    plt.yrange = [0.5 * (ds - dserr).min(), 1.5 * (ds + dserr).max()]

    plt.xlog = True
    plt.ylog = True
    plt.show()
