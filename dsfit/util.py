import numpy as np


def r200_to_m200(*, r200, rho):
    return 200 * rho * (4.0 / 3.0) * np.pi * r200 ** 3


def m200_to_r200(*, m200, rho):
    power = 1.0/3.0
    return (m200/200/rho/np.pi/(4.0 / 3.0))**power
