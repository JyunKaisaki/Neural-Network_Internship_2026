import torch

from Residual import H_phis, Vfbs_from_independent, phi_fermi


def surface_potential_residual(
    phis_PINN,
    Vgs,
    Vds,
    T,
    NA,
    eps_sic,
    Cox,
    Vfbs0,
    Qox,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    Eg,
):
    # Evaluate the surface-potential equation residual used by the PINN.
    q = torch.as_tensor(1.602176634e-19, dtype=phis_PINN.dtype, device=phis_PINN.device)
    k_B = torch.as_tensor(1.380649e-23, dtype=phis_PINN.dtype, device=phis_PINN.device)
    T_t = torch.as_tensor(T, dtype=phis_PINN.dtype, device=phis_PINN.device)
    NA_t = torch.as_tensor(NA, dtype=phis_PINN.dtype, device=phis_PINN.device)
    eps_t = torch.as_tensor(eps_sic, dtype=phis_PINN.dtype, device=phis_PINN.device)
    Cox_t = torch.as_tensor(Cox, dtype=phis_PINN.dtype, device=phis_PINN.device)
    Vgs = Vgs.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    phis_PINN = phis_PINN.reshape(-1, 1)
    phi_t = k_B * T_t / q
    phi_Fermi = phi_fermi(T_t, NA_t, Eg, ref=phis_PINN)
    gamma = torch.sqrt(torch.clamp(2.0 * eps_t * q * NA_t, min=0.0)) / Cox_t
    Vfbs_p = Vfbs_from_independent(
        Vfbs0,
        Cox,
        phis_PINN,
        Vds,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T_t,
        NA_t,
        Eg,
    )
    H = H_phis(phis_PINN, phi_t, phi_Fermi, Vds)
    sign_phis = torch.where(phis_PINN >= 0.0, torch.ones_like(phis_PINN), -torch.ones_like(phis_PINN))
    _ = Qox
    return Vgs - Vfbs_p - phis_PINN - sign_phis * gamma * H


def physics_loss(
    phis_PINN,
    Vgs,
    Vds,
    T,
    NA,
    eps_sic,
    Cox,
    Vfbs0,
    Qox,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    Eg,
):
    # Return the mean squared physics residual.
    residual = surface_potential_residual(
        phis_PINN,
        Vgs,
        Vds,
        T,
        NA,
        eps_sic,
        Cox,
        Vfbs0,
        Qox,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        Eg,
    )
    return torch.mean(residual**2)
