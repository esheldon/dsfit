import numpy as np
from numpy import sqrt, log10, linspace
from . import nfw
from . import linear
from .cosmopars import get_cosmo_pars


class NFWBiasFitter(object):
    def __init__(
        self,
        *,
        z,
        r,
        cosmo_pars=None,
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
        withlin: bool
            Include the linear bias term.  Default True.
        """

        if cosmo_pars is None:
            cosmo_pars = get_cosmo_pars()

        self.r_ = r.copy()
        self.cosmo_pars = cosmo_pars

        self.withlin = withlin

        self.nfw = nfw.NFW(cosmo_pars=cosmo_pars, z=z)

        if self.withlin:
            self.lin = linear.Linear(cosmo_pars=cosmo_pars)
            print("pre-computing linear dsig at %s points" % r.size)
            self.lin_dsig_ = self.lin.dsig(self.r_)

    def fit(self, *, dsig, dsigcov, guess, more=False):
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
            guess: [r200,c,B] or [r200,c] if withlin=False

        Outputs:
            if more=False:
                p,cov:  The parameters array and the covariance arra
            if more=True:
                a dict with  'p' and 'cov' along with
                    r200,r200_err
                    c, c_err
                    B, B_err  (-9999 if withlin=False)
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
            func = self.dsig
        else:
            func = self.nfw.dsig
            npar = 2

        if len(guess) != npar:
            raise ValueError("parameter guess must have %s elements" % npar)

        print("running curve_fit")
        p, cov = curve_fit(
            func,
            self.r_,
            dsig,
            sigma=dsigcov,
            p0=guess,
        )
        if not more:
            return p, cov
        else:
            return self.pars2more(p, cov)

    def dsig(self, r, r200, c, B):
        """
        get delta sigma for the input data
        """
        if self.withlin:
            return self.nfw.dsig(r, r200, c) + self.lin_dsig(r, B)
        else:
            return self.nfw.dsig(r, r200, c)

    def lin_dsig(self, r, B):
        """
        r values must coinciced with linear dsig values
        """
        if r.size != self.lin_dsig_.size:
            raise ValueError("r and dsig must be same size")
        return B * self.lin_dsig_

    def pars2more(self, p, cov):
        r200 = p[0]
        r200_err = sqrt(cov[0, 0])
        c = p[1]
        c_err = sqrt(cov[1, 1])

        m200 = self.nfw.m200(r200)
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
            res["B"] = p[2]
            res["B_err"] = sqrt(cov[2, 2])

        return res


def fit_nfw_lin_dsig(
    *,
    z, r, dsig, dsigcov, guess,
    more=False, cosmo_pars=None, withlin=True,
):
    """
    Name:
        fit_nfw_line_dsig
    Calling Sequence:
        fit_nfw_lin_dsig(omega_m, z, r, ds, dscov, guess, more=False,
                         **kwds)
    Inputs:
        omega_m:
        z: the redshift
        r: radius in Mpc
        ds: delta sigma in pc^2/Msun
        dscov: cov matrix
        guess: guesses for [r200,c,B]

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
                B, B_err
                m200, m200_err
        plus their errors

    """

    fitter = NFWBiasFitter(z=z, r=r, cosmo_pars=cosmo_pars, withlin=withlin)
    return fitter.fit(dsig=dsig, dsigcov=dsigcov, guess=guess, more=more)


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
    r200_err = sqrt(cov[0, 0])
    c = p[1]
    c_err = sqrt(cov[1, 1])

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

    import hickory

    z = 0.25

    r200 = 1.0
    c = 5.0
    B = 10.0

    rmax = 50.0
    log_rmin = log10(rmin)
    log_rmax = log10(rmax)
    npts = 30
    r = 10.0 ** linspace(log_rmin, log_rmax, npts)

    fitter = NFWBiasFitter(z=z, r=r)
    ds = fitter.dsig(r, r200, c, B)
    # 10% errors
    dserr = 0.1 * ds
    ds += dserr * np.random.standard_normal(ds.size)
    dscov = np.diag(dserr)

    guess = np.array([r200, c, B], dtype="f8")
    # add 10% error to the guess
    guess += 0.1 * guess * np.random.standard_normal(guess.size)

    res = fitter.fit(
        dsig=ds,
        dsigcov=dscov,
        guess=guess
        , more=True,
    )

    r200_fit = res["r200"]
    r200_err = res["r200_err"]
    c_fit = res["c"]
    c_err = res["c_err"]
    B_fit = res["B"]
    B_err = res["B_err"]

    print("Truth:")
    print("    r200: %f" % r200)
    print("       c: %f" % c)
    print("       B: %f" % B)
    print("r200_fit: %f +/- %f" % (r200_fit, r200_err))
    print("   c_fit: %f +/- %f" % (c_fit, c_err))
    print("   B_fit: %f +/- %f" % (B_fit, B_err))
    print("Cov:")
    print(res["cov"])

    rfine = 10.0 ** linspace(log_rmin, log_rmax, 100)
    fitter2 = NFWBiasFitter(z=z, r=rfine)

    yfit = fitter2.dsig(rfine, r200_fit, c_fit, B_fit)
    yfit_nfw = fitter2.nfw.dsig(rfine, r200_fit, c_fit)
    yfit_lin = fitter2.lin_dsig(rfine, B_fit)

    plt = hickory.Plot(
        xlabel=r"$r$ [$h^{-1}$ Mpc]",
        ylabel=r"$\Delta\Sigma ~ [h \mathrm{M}_{\odot} \mathrm{pc}^{-2}]$",
        xlim=[0.5 * rmin, 1.5 * rmax],
        ylim=[0.5 * (ds - dserr).min(), 1.5 * (ds + dserr).max()]
    )
    plt.set_xscale('log')
    plt.set_yscale('log')
    plt.errorbar(r, ds, dserr)

    plt.curve(rfine, yfit, label='model')
    plt.curve(rfine, yfit_nfw, label='nfw')
    plt.curve(rfine, yfit_lin, label='linear')
    plt.legend()
    plt.show()


def test_fit_nfw_dsig(rmin=0.01):

    from biggles import FramedPlot, Points, SymmetricErrorBarsY, Curve

    omega_m = 0.25
    z = 0.25
    n = nfw.NFW(omega_m, z)

    r200 = 1.0
    c = 5.0

    rmax = 5.0
    log_rmin = log10(rmin)
    log_rmax = log10(rmax)
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
