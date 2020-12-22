import numpy as np
from . import nfw
from . import linear
from .cosmo import get_cosmo
from colossus.cosmology.cosmology import (
    Cosmology,
)
from .concentration import get_conc


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
            cosmo = get_cosmo(cosmo)

        self.z = z
        self.r_ = r.copy()
        self.cosmo = cosmo

        self.withlin = withlin

        if self.withlin:
            self.lin = linear.Linear(cosmo=cosmo, z=z, b=1)
            # print("pre-computing linear dsig at %s points" % r.size)
            self.lin_dsig_ = self.lin.get_dsig(self.r_)

    def fit(self, *, dsig, dsigcov, guess,
            lm200_bounds=[11, 15],
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
            guess: [log10(m200), b] or [log10(m200)] if withlin=False

        Outputs:
            a dict with  'p' and 'cov' along with
                lm200, lm200_err (which are  actually for log10(m200))
                c, c_err
                b, b_err  (-9999 if withlin=False)
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
            # lm200, b
            npar = 2
            func = self._get_dsig_nfw_lin
        else:
            # lm200
            func = self._get_dsig_nfw
            npar = 1

        if len(guess) != npar:
            raise ValueError("parameter guess must have %s elements" % npar)

        # print("running curve_fit")
        bounds = (
            # np.array([0.05, 0.1, 0.1]),
            # np.array([5.00, 5.0, 3.0])
            np.array([lm200_bounds[0], b_bounds[0]]),
            np.array([lm200_bounds[1], b_bounds[1]])
        )

        p, cov = curve_fit(
            func,
            self.r_,
            dsig,
            sigma=dsigcov,
            p0=guess,
            bounds=bounds,
        )
        return self.pars2more(p, cov)

    def _get_dsig_nfw(self, r, lm200):
        """
        for fitting, accepts non-keywords pars
        """
        m200 = 10.0**lm200
        return self.get_nfw_dsig(r=r, m200=m200)

    def _get_dsig_nfw_lin(self, r, lm200, b):
        """
        for fitting, accepts non-keywords pars
        """
        assert self.withlin
        m200 = 10.0**lm200
        return self.get_dsig(r=r, m200=m200, b=b)

    def get_dsig(self, *, r, m200=None, lm200=None, b=None):
        """
        get delta sigma for the input data

        called from curve_fit so must accept pars as non-keyword
        """
        if self.withlin:
            assert b is not None, "send b for withlin"
            nfw_dsig = self.get_nfw_dsig(r=r, m200=m200, lm200=lm200)
            lin_dsig = self.get_lin_dsig(r=r, b=b)
            return nfw_dsig + lin_dsig

        else:
            return self.get_nfw_dsig(r=r, m200=m200)

    def get_nfw_dsig(self, *, r, m200=None, lm200=None):
        if lm200 is not None:
            m200 = 10.0**lm200
        elif m200 is None:
            raise ValueError('send either m200= or lm200=')

        c, _ = get_conc(cosmo=self.cosmo, z=self.z, M=m200)
        tnfw = nfw.NFW(
            cosmo=self.cosmo,
            z=self.z,
            m200=m200,
            c=c,
        )
        return tnfw.get_dsig(r)

    def get_lin_dsig(self, *, r, b):
        """
        r values must coinciced with linear dsig values
        """
        if r.size != self.lin_dsig_.size:
            raise ValueError("r and dsig must be same size")

        return b * self.lin_dsig_

    def pars2more(self, p, cov):
        lm200 = p[0]
        lm200_err = np.sqrt(cov[0, 0])
        m200 = 10.0**lm200
        c, _ = get_conc(cosmo=self.cosmo, z=self.z, M=m200)

        res = {
            "p": p,
            "cov": cov,
            "c": c,
            "lm200": lm200,
            "lm200_err": lm200_err,
            "m200": m200,
        }

        if self.withlin:
            res["b"] = p[1]
            res["b_err"] = np.sqrt(cov[1, 1])

        return res


def plot(
    *,
    r,
    z,
    m200=None,
    lm200=None,
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
    yfit = fitter.get_dsig(r=r, m200=m200, lm200=lm200, b=b)

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
        yfit_lin = fitter.get_lin_dsig(r=r, b=b)
        yfit_nfw = fitter.get_nfw_dsig(r=r, m200=m200, lm200=lm200)
        plt.curve(r, yfit_nfw, label='nfw')
        plt.curve(r, yfit_lin, label='linear')

    if dolegend:
        plt.legend()

    if show:
        plt.show()

    return plt


def plot_residuals(
    *,
    r,
    dsig,
    dsigcov,
    z,
    m200=None,
    lm200=None,
    b=None,
    xlim=None,
    ylim=None,
    resid_axis_kw={},
    no_resid_xticklabels=False,
    no_resid_yticklabels=False,
    plt=None,
    show=False,
):

    import hickory

    if b is None:
        withlin = False
    else:
        withlin = True

    fitter = NFWBiasFitter(z=z, r=r, withlin=withlin)
    yfit = fitter.get_dsig(r=r, m200=m200, lm200=lm200, b=b)

    dsigerr = np.sqrt(np.diag(dsigcov))

    ymin, ymax = [
        0.5 * (dsig - dsigerr).min(),
        1.5 * (dsig + dsigerr).max(),
    ]
    if ymin < 0:
        ymin = 0.1

    if ylim is None:
        ylim = [0.5*ymin, 1.5*ymax]

    if xlim is None:
        rmin, rmax = r.min(), r.max()
        xlim = [0.5 * rmin, 1.5 * rmax]

    if plt is None:
        dolegend = True
        plt = hickory.Plot(
            xlabel=r"$r$ [Mpc]",
            ylabel=r"$\Delta\Sigma ~ [\mathrm{M}_{\odot} \mathrm{pc}^{-2}]$",
            xlim=xlim,
            ylim=ylim,
        )
        if 'ylabel' not in resid_axis_kw:
            resid_axis_kw['ylabel'] = r'$\Delta$'

        plt.set_xscale('log')
        plt.set_yscale('log')
    else:
        plt.set(xlim=xlim, ylim=ylim)
        dolegend = False


    alpha = 0.5
    _residuals_plots(
        plt=plt,
        x=r, y=dsig, yerr=dsigerr, model=yfit,
        resid_axis_kw=resid_axis_kw,
        data_kw={'alpha': alpha},
        model_kw={'label': 'model'},
        no_resid_xticklabels=no_resid_xticklabels,
        no_resid_yticklabels=no_resid_yticklabels,
    )

    if withlin:
        yfit_lin = fitter.get_lin_dsig(r=r, b=b)
        yfit_nfw = fitter.get_nfw_dsig(r=r, m200=m200, lm200=lm200)
        plt.curve(r, yfit_nfw, label='nfw')
        plt.curve(r, yfit_lin, label='linear')

    if dolegend:
        plt.legend()

    if show:
        plt.show()

    return plt


def _residuals_plots(
    *, plt, x, y, model, yerr=None, frac=0.2, pad=0,
    data_kw={}, model_kw={},
    resid_axis_kw={},
    no_resid_xticklabels=False,
    no_resid_yticklabels=False,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt)
    size_string = '%d%%' % (frac*100)
    ax2 = divider.append_axes("bottom", size=size_string, pad=pad)
    plt.figure.add_axes(ax2, label='%d' % np.random.randint(0, 2**15))

    # residuals = model - y
    residuals_err = yerr
    residuals = (model - y) * x
    residuals_err = yerr * x

    ax2.set(**resid_axis_kw)
    ax2.set_xscale('log')
    ax2.axhline(0, color='black')

    if yerr is not None:
        plt.errorbar(x, y, yerr, **data_kw)
        ax2.errorbar(x, residuals, residuals_err, **data_kw)
    else:
        plt.plot(x, y, **data_kw)
        ax2.plot(x, residuals, **data_kw)

    plt.curve(x, model, **model_kw)

    # plt.set_xticks([])
    # if no_xticks:
    #     ax2.set_xticks([])
    # if no_yticks:
    #     ax2.set_yticks([])
    if no_resid_xticklabels:
        ax2.set_xticklabels([])
    if no_resid_yticklabels:
        ax2.set_yticklabels([])


def fit_nfw_lin_dsig(
    *,
    z, r, dsig, dsigcov, guess,
    cosmo='planck18', withlin=True,
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
        guess: guesses for [log10(m200),c,b]

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
                lm200, lm200_err
                b, b_err
                c,
    """

    fitter = NFWBiasFitter(z=z, r=r, cosmo=cosmo, withlin=withlin)
    return fitter.fit(dsig=dsig, dsigcov=dsigcov, guess=guess)


def test_fit_nfw_lin_dsig(rmin=0.01):

    cosmo = get_cosmo('planck18')
    z = 0.25

    lm200 = 12.5
    b = 2.0

    log_rmin = np.log10(rmin)
    log_rmax = np.log10(50.0)
    npts = 30
    r = np.logspace(log_rmin, log_rmax, npts)

    fitter = NFWBiasFitter(cosmo=cosmo, z=z, r=r)
    ds = fitter.get_dsig(r=r, lm200=lm200, b=b)

    # 10% errors
    dserr = 0.1 * ds
    ds += dserr * np.random.standard_normal(ds.size)
    dscov = np.diag(dserr)

    guess = np.array([lm200, b], dtype="f8")
    guess += 0.1 * guess * np.random.standard_normal(guess.size)

    res = fitter.fit(
        dsig=ds,
        dsigcov=dscov,
        guess=guess,
    )

    lm200_fit = res["lm200"]
    lm200_err = res["lm200_err"]
    b_fit = res["b"]
    b_err = res["b_err"]

    print("Truth:")
    print("    lm200: %f" % lm200)
    print("        b: %f" % b)
    print("lm200_fit: %f +/- %f" % (lm200_fit, lm200_err))
    print("    b_fit: %f +/- %f" % (b_fit, b_err))
    print("Cov:")
    print(res["cov"])

    plot(
        r=r,
        z=z,
        lm200=lm200_fit,
        b=b_fit,
        dsig=ds,
        dsigcov=dscov,
        show=True,
    )
