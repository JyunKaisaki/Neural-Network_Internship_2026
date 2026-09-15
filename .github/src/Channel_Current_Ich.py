import torch

from Residual import Vfbs_from_independent


def _as_like(value, ref):
    if torch.is_tensor(value):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


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
    eps_sic,
):
    Vgs = Vgs.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    phis_s0 = phis_s0.reshape(-1, 1)
    phis_sL = phis_sL.reshape(-1, 1)
    q = _as_like(1.602176634e-19, Vgs)
    k_B = _as_like(1.380649e-23, Vgs)
    T_t = _as_like(T, Vgs)
    NA_t = _as_like(NA, Vgs)
    eps_t = _as_like(eps_sic, Vgs)
    Cox_t = _as_like(Cox, Vgs)
    phi_t = k_B * T_t / q
    gamma = torch.sqrt(torch.clamp(2.0 * eps_t * q * NA_t, min=0.0)) / Cox_t
    phi_f_source = torch.zeros_like(Vds)
    Vfbs_source = Vfbs_from_independent(
        Vfbs0,
        Cox,
        phis_s0,
        phi_f_source,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T,
        NA,
        Eg,
    )
    s0 = torch.clamp(phis_s0, min=0.0)
    sL = torch.clamp(phis_sL, min=0.0)
    term1 = Cox_t * (Vgs - Vfbs_source + phi_t) * (sL - s0)
    term2 = -0.5 * Cox_t * (sL**2 - s0**2)
    term3 = -(2.0 / 3.0) * gamma * Cox_t * (sL**1.5 - s0**1.5)
    term4 = phi_t * gamma * Cox_t * (torch.sqrt(sL) - torch.sqrt(s0))
    return term1 + term2 + term3 + term4


def effective_vds(Vds, Vdssat, delta):
    Vds = Vds.reshape(-1, 1)
    Vdssat = torch.clamp(_as_like(Vdssat, Vds), min=1.0e-6)
    delta = torch.clamp(_as_like(delta, Vds), min=0.05)
    ratio = torch.clamp(Vds / Vdssat, min=0.0)
    return Vds * torch.pow(1.0 + torch.pow(ratio, delta), -1.0 / delta)


def effective_mobility(mu_lf, Ey, vsat):
    mu_lf = torch.clamp(_as_like(mu_lf, Ey), min=1.0e-12)
    vsat = torch.clamp(_as_like(vsat, Ey), min=1.0e-12)
    a = torch.square(Ey / vsat)
    a_safe = torch.clamp(a, min=1.0e-30)
    mu2_formula = (torch.sqrt(1.0 + 4.0 * a * mu_lf**2) - 1.0) / (2.0 * a_safe)
    mu2 = torch.where(a > 1.0e-20, mu2_formula, mu_lf**2)
    return torch.sqrt(torch.clamp(mu2, min=1.0e-24))


def lateral_field(phis_s0, phis_sL, Lch_cm):
    Lch = torch.clamp(_as_like(Lch_cm, phis_s0), min=1.0e-12)
    return torch.abs(phis_sL - phis_s0) / Lch


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
    K_current_gain,
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
    K = _as_like(K_current_gain, I_phi)
    mu = _as_like(mu_eff_cm2_Vs, I_phi)
    lam = _as_like(lambda_clm, I_phi)
    Ich = K * mu * (1.0 + lam * Vds.reshape(-1, 1)) * I_phi
    return torch.clamp(Ich, min=0.0)


def channel_current_physical(
    Vgs,
    Vds,
    Vdseff,
    phis_s0,
    phis_sL,
    *,
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
    K_current_gain,
    mu_lf_cm2_Vs,
    lambda_clm,
    Lch_cm,
    vsat_cm_s,
):
    I_phi = channel_integral_I_phi(
        Vgs,
        Vdseff,
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
    Ey = lateral_field(phis_s0, phis_sL, Lch_cm)
    mu_eff = effective_mobility(mu_lf_cm2_Vs, Ey, vsat_cm_s)
    K = _as_like(K_current_gain, I_phi)
    lam = _as_like(lambda_clm, I_phi)
    Ich = K * mu_eff * (1.0 + lam * Vds.reshape(-1, 1)) * I_phi
    return torch.clamp(Ich, min=0.0), mu_eff, I_phi
