OMEGA_M_DEFAULT = 0.30
OMEGA_B_DEFAULT = 0.0486
SIGMA_8_DEFAULT = 0.8
H_DEFAULT = 0.70
NS_DEFAULT = 0.98


def get_cosmo_pars(
    *,
    omega_m=OMEGA_M_DEFAULT,
    omega_b=OMEGA_B_DEFAULT,
    sigma_8=SIGMA_8_DEFAULT,
    h=H_DEFAULT,
    ns=NS_DEFAULT,
):
    return {
        'omega_m': omega_m,
        'omega_b': omega_b,
        'sigma_8': sigma_8,
        'h': h,
        'ns': ns,
    }
