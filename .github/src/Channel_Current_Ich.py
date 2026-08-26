import torch
from Residual import Vfbs


def channel_integral_I_phi(
    Vgs,
    Vds,
    phis_s0,
    phis_sL,
    Vfbs0,
    Cox,
    Qox,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,
    eps_sic
):

    Vgs = Vgs.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    phis_s0 = phis_s0.reshape(-1, 1)
    phis_sL = phis_sL.reshape(-1, 1)

    q = Vgs.new_tensor(1.602e-19)
    k_B = Vgs.new_tensor(1.381e-23)

    T_t = Vgs.new_tensor(T)
    NA_t = Vgs.new_tensor(NA)
    eps_t = Vgs.new_tensor(eps_sic)
    Cox_t = Vgs.new_tensor(Cox)

    phi_t = k_B * T_t / q

    gamma = torch.sqrt(
        torch.clamp(
            2.0 * eps_t * q * NA_t,
            min=0.0
        )
    ) / Cox_t

    Vfbs_drain = Vfbs(
        Vfbs0,
        Cox,
        Qox,
        phis_sL,
        Vds,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T,
        NA,
        Eg,
    )

    s0 = torch.clamp(
        phis_s0,
        min=0.0
    )

    sL = torch.clamp(
        phis_sL,
        min=0.0
    )

    term1 = (
        Cox_t
        * (Vgs - Vfbs_drain + phi_t)
        * (sL - s0)
    )

    term2 = (
        -0.5
        * Cox_t
        * (sL**2 - s0**2)
    )

    term3 = (
        -(2.0 / 3.0)
        * gamma
        * Cox_t
        * (sL**1.5 - s0**1.5)
    )

    term4 = (
        phi_t
        * gamma
        * Cox_t
        * (
            torch.sqrt(sL)
            - torch.sqrt(s0)
        )
    )

    return (
        term1
        + term2
        + term3
        + term4
    )

def channel_current_Ich(
    Vgs,
    Vds,
    phis_s0,
    phis_sL,
    Vfbs0,
    Cox,
    Qox,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,
    eps_sic,
    W_eff_cm,
    Lch_cm,
    mu_eff_cm2_Vs,
    lambda_clm,
):

    I_phi = channel_integral_I_phi(
        Vgs,
        Vds,
        phis_s0,
        phis_sL,
        Vfbs0,
        Cox,
        Qox,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T,
        NA,
        Eg,
        eps_sic,
    )

    W = I_phi.new_tensor(W_eff_cm)
    L = I_phi.new_tensor(Lch_cm)
    mu = I_phi.new_tensor(mu_eff_cm2_Vs)
    lam = I_phi.new_tensor(lambda_clm)

    Ich = (
        (W * mu / L)
        * (
            1.0
            + lam * Vds.reshape(-1, 1)
        )
        * I_phi
    )

    return torch.clamp(
        Ich,
        min=0.0
    )