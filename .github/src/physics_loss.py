import torch

from Residual import *

"""
    Physics-informed surface-potential loss:
    L_SPE = mean[(Vgs - Vfbs - phis_PINN - gamma * H(phis_PINN))^2]
"""

def physics_loss(
    phis_PINN,
    Vgs,
    Vds,
    T,
    NA,
    eps_sic,
    Cox,
    Vfb0,
    Qox,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    Eg,
):

    q = torch.as_tensor(
        1.602e-19,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    k_B = torch.as_tensor(
        1.381e-23,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    T = torch.as_tensor(
        T,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    NA = torch.as_tensor(
        NA,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    eps_sic = torch.as_tensor(
        9.26 * 8.854e-14,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    Cox = torch.as_tensor(
        Cox,
        dtype=phis_PINN.dtype,
        device=phis_PINN.device
    )

    Vgs = Vgs.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    phis_PINN = phis_PINN.reshape(-1, 1)

    phi_t = (
        k_B * T / q
    )


    phi_Fermi = phi_fermi(
        T,
        NA,
        Eg,
        ref=phis_PINN
    )

    phi_f = Vds

    gamma = (
        torch.sqrt(
            torch.clamp(
                2.0 * eps_sic * q * NA,
                min=0.0
            )
        )
        / Cox
    )

    Vfbs_p = Vfbs(
        Vfb0,
        Cox,
        Qox,
        phis_PINN,
        phi_f,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T,
        NA,
        Eg,
    )

    H = H_phis(
        phis_PINN,
        phi_t,
        phi_Fermi,
        phi_f,
    )

    residual = (
        Vgs
        - Vfbs_p
        - phis_PINN
        - gamma * H
    )

    loss_SPE = torch.mean(
        residual ** 2
    )

    return loss_SPE