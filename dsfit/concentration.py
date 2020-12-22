"""
code copied from the colossus package, converted to not use
global variables for cosmology

https://bitbucket.org/bdiemer/colossus/src/master/
"""
import numpy as np
from colossus import defaults
from colossus.utils import constants


def get_conc(*, cosmo, z, M, mdef='200m'):
    """
    concentration for the Bhattacharya13 model.  Copied from the colossus
    package, converted to not use a global variable

    Parameters
    ----------
    cosmo: a colossus Cosmology object
        For evaluating the growth factor
    """

    D = cosmo.growthFactor(z)

    # Note that peak height in the B13 paper is defined wrt. the mass
    # definition in question, so we can just use M to evaluate nu.
    nu = get_peak_height(cosmo=cosmo, M=M, z=z)

    if mdef == '200c':
        c_fit = 5.9 * D**0.54 * nu**-0.35
    elif mdef == 'vir':
        c_fit = 7.7 * D**0.90 * nu**-0.29
    elif mdef == '200m':
        c_fit = 9.0 * D**1.15 * nu**-0.29
    else:
        msg = (
            'Invalid mass definition for Bhattacharya '
            'et al. 2013 model, %s.' % mdef
        )
        raise ValueError(msg)

    M_min = 2E12
    M_max = 2E15
    if z > 0.5:
        M_max = 2E14
    if z > 1.5:
        M_max = 1E14

    mask = (M >= M_min) & (M <= M_max) & (z <= 2.0)

    return c_fit, mask


def get_peak_height(
    *, cosmo, M, z, ps_args=defaults.PS_ARGS,
    sigma_args=defaults.SIGMA_ARGS, deltac_args={},
):
    """
    The peak height corresponding to a given a halo mass.

    Peak height is defined as :math:`\\nu \\equiv \\delta_c / \\sigma(M)`.
    This function takes optional parameter lists for the
    :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum`,
    :func:`~cosmology.cosmology.Cosmology.sigma`, and
    :func:`collapseOverdensity` functions. Please see the documentation of
    those funtions for details.

    Note that the peak height does not explicitly depend on the mass
    definition in which ``M`` is given, but that it corresponds to that
    mass definition. For example, if M200m is given, that mass and the
    corresponding peak height will be larger than R500c for the same halo.

    Parameters
    -------------------------------------------------------------------------------------------
    M: array_like
        Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy
        array.
    z: float
        Redshift.
    ps_args: dict
        Arguments passed to the
        :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum`
        function.
    sigma_args: dict
        Arguments passed to the
        :func:`~cosmology.cosmology.Cosmology.sigma` function.
    deltac_args: dict
        Arguments passed to the :func:`collapseOverdensity` function.

    Returns
    -------------------------------------------------------------------------------------------
    nu: array_like
        Peak height; has the same dimensions as ``M``.

    See also
    -------------------------------------------------------------------------------------------
    massFromPeakHeight: Halo mass from peak height.
    """

    lagrangian_R = (
        (3.0 * M / 4.0 / np.pi / cosmo.rho_m(0.0) / 1E9)**(1.0 / 3.0)
    )
    sigma = cosmo.sigma(
        lagrangian_R, z, ps_args=ps_args, **sigma_args
    )
    nu = get_collapse_overdensity(cosmo=cosmo, z=z, **deltac_args) / sigma

    return nu


def get_collapse_overdensity(*, cosmo, corrections=False, z=None):
    """
    The linear overdensity threshold for halo collapse.

    The linear overdensity threshold for halo collapse according to the
    spherical top-hat collapse model (`Gunn & Gott 1972
    <http://adsabs.harvard.edu/abs/1972ApJ...176....1G>`_). In an EdS
    universe, this number is :math:`3/5 (3\\pi/2)^{2/3}=1.686`.

    This value is modified very slightly in a non-EdS universe (by less
    than 3% for any realistic cosmology). Such corrections are applied if
    desired, by default this function returns the constant value (see,
    e.g., `Mo, van den Bosch & White
    <http://adsabs.harvard.edu/abs/2010gfe..book.....M>`_ for a derivation
    of the corrections). Note that correction formulae are implemented for
    flat cosmologies and cosmologies without dark energy, but not for the
    general case (both curvature and dark energy). The correction is
    essentially identical in effect to the Equation A6 of `Kitayama & Suto
    1996
    <https://ui.adsabs.harvard.edu/abs/1996ApJ...469..480K/abstract>`_.

    Parameters
    -------------------------------------------------------------------------------------------
    corrections: bool
        If True, corrections to the collapse overdensity are applied in a
        non-EdS cosmology. In this case, a redshift must be passed.
    z: float
        Redshift where the collapse density is evaluated. Only necessary if
        ``corrections == True``.

    Returns
    -------------------------------------------------------------------------------------------
    delta_c: float
            The threshold overdensity for collapse.
    """

    delta_c = constants.DELTA_COLLAPSE

    if corrections:
        if z is None:
            raise ValueError(
                'If corrections == True, a redshift must be passed.'
            )

        Om = cosmo.Om(z)
        if cosmo.flat:
            delta_c *= Om**0.0055
        elif cosmo.Ode0 == 0.0:
            delta_c *= Om**0.0185

    return delta_c
