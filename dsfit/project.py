"""
project3d:

    Compute the 2D projection of points prepresenting a function of the
    radius r=sqrt(x**2 + y**2 + z**2). E.g. for a function rho(r) where
    r is a three-d radius, compute the integral along a line of sight.

    Uses a three point neighborhood to interpolate the
    function as a quadratic. Then uses analytic formulas to compute
    the integral endpoint correction by extrapolating.

    The last point is computed using a powerlaw extrapolation. You might
    want to just remove the last point if you don't trust this.

    Based on original IDL code by Dave Johnston

lag_quad_poly(x,y):
    Computes the uniq quadratic poly through three x,y points

"""
import numpy as np
from numpy import int32, log, log10, arange, sqrt

DEFAULT_POWER = -3.12


def project3d(r, rho, extrapolate=True, power=DEFAULT_POWER):
    """

    Compute the 2D projection of points prepresenting a function of the
    radius r=sqrt(x**2 + y**2 + z**2). E.g. for a function rho(r) where
    r is a three-d radius, compute the integral along a line of sight.

    Uses a three point neighborhood to interpolate the
    function as a quadratic. Then uses analytic formulas to compute
    the integral endpoint correction by extrapolating.

    The last point is computed using a powerlaw extrapolation. You might
    want to just remove the last point if you don't trust this.

    Parameters
    ----------
    r: array
        Radius in Mpc
    rho: array
        profile to be projected
    extrapolate: bool
        Set True to extrapolate
    power: float
        power law index for projection.  Set to None to estimate from
        the data (watch out this can fail).
    """

    nr = r.size
    if extrapolate:
        # extrapolate then recurse
        r_ext, rho_ext = extrapolate_powerlaw(r, rho, power=power)
        sig = project3d(r_ext, rho_ext, extrapolate=False)
        sig = sig[0:nr]
        return sig

    sig = np.zeros(nr, dtype="f8")
    for j in range(nr - 2):
        RR = r[j]
        RR2 = RR ** 2
        num = nr - j
        Int = np.zeros(num, dtype="f8")

        for i in range(num - 2):
            x = r[i + j: i + j + 3]
            y = rho[i + j: i + j + 3]
            p = lag_quad_poly(x, y)
            A = x[0]
            B = x[1]

            I0, I1, I2 = getI(RR, RR2, A, B, p)

            Int[i] = 2 * (I0 + I1 + I2)
        sig[j] = Int.sum()
    return sig


def getI(RR, RR2, A, B, p):
    Rad = B
    Rad2 = Rad ** 2
    S = sqrt(Rad2 - RR2)
    I0_B = S
    I1_B = S * Rad / 2.0 + RR2 * log(Rad + S) / 2.0
    I2_B = S * (2 * RR2 + Rad2) / 3.0

    Rad = A
    Rad2 = Rad ** 2
    S = sqrt(Rad2 - RR2)
    I0_A = S
    I1_A = S * Rad / 2.0 + RR2 * log(Rad + S) / 2.0
    I2_A = S * (2 * RR2 + Rad2) / 3.0

    I0 = (I0_B - I0_A) * p[0]
    I1 = (I1_B - I1_A) * p[1]
    I2 = (I2_B - I2_A) * p[2]

    return I0, I1, I2


def extrapolate_powerlaw(r, rho, power=DEFAULT_POWER):
    nr = r.size
    # extrapolate for the integral, but then trim back

    # the fraction of the log interval to add on, 0.5 is probably good
    fac = 1.0
    n_extf = nr * fac
    if n_extf < 10.0:
        n_extf = 10.0
    n_ext = int32(n_extf)

    rmin = r.min()
    rmax = r.max()
    Lext = (log10(rmax) - log10(rmin)) * fac

    grid = 1 + arange(n_ext, dtype="f8")
    r_ext = rmax * 10.0 ** (Lext * grid / (n_ext - 1))

    if power is None:
        power = log(rho[nr - 1] / rho[nr - 2]) / log(r[nr - 1] / r[nr - 2])

    A = rho[nr - 1] / (r[nr - 1] ** power)
    rho_ext = A * r_ext ** power

    r_ext = np.concatenate((r, r_ext))
    rho_ext = np.concatenate((rho, rho_ext))

    return r_ext, rho_ext


def lag_quad_poly(x, y):
    """
    NAME:
        lag_quad_poly

    PURPOSE:
        Computes the uniq quadratic poly through three x,y points

    CALLING SEQUENCE:
        p=lag_quad_poly(x,y)

    INPUTS:
        x,y -  set of three points

    OUTPUTS:
        polynomial -  three numbers , usual IDL notation

    METHOD:
        Uses Lagrange formula
    """

    if x.size != 3 or y.size != 3:
        raise ValueError("x,y must be length 3")

    q = np.zeros(3, dtype="f8")
    p = np.zeros(3, dtype="f8")

    q[0] = y[0] / ((x[0] - x[1]) * (x[0] - x[2]))
    q[1] = y[1] / ((x[1] - x[0]) * (x[1] - x[2]))
    q[2] = y[2] / ((x[2] - x[0]) * (x[2] - x[1]))

    p[0] = q[0] * x[1] * x[2] + q[1] * x[0] * x[2] + q[2] * x[0] * x[1]
    p[1] = -q[0] * (x[1] + x[2]) - q[1] * (x[0] + x[2]) - q[2] * (x[0] + x[1])
    p[2] = q.sum()

    return p
