import torch

from Residual import H_phis, phi_fermi


def _as_like(value, ref):
    if torch.is_tensor(value):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def G_phigd(phigd, Vds, T, NA, Eg):
    phigd = phigd.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    q = _as_like(1.602176634e-19, phigd)
    k_B = _as_like(1.380649e-23, phigd)
    T_t = _as_like(T, phigd)
    phi_t = k_B * T_t / q
    phi_Fermi = phi_fermi(T, NA, Eg, ref=phigd)
    arg_1 = -phigd / phi_t
    arg_2 = -(2.0 * phi_Fermi + Vds) / phi_t
    arg_3 = phigd / phi_t
    exp_1 = torch.exp(torch.clamp(arg_1, min=-80.0, max=80.0))
    exp_2 = torch.exp(torch.clamp(arg_2, min=-80.0, max=80.0))
    exp_3 = torch.exp(torch.clamp(arg_3, min=-80.0, max=80.0))
    return 1.0 - exp_1 + exp_2 * (exp_3 - 1.0)


def Cjfet(phigd, Vds, T, NA, ND, Eg, eps_sic, Agd):
    phigd = phigd.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    q = _as_like(1.602176634e-19, phigd)
    k_B = _as_like(1.380649e-23, phigd)
    T_t = _as_like(T, phigd)
    ND_t = _as_like(ND, phigd)
    eps_t = _as_like(eps_sic, phigd)
    Agd_t = _as_like(Agd, phigd)
    phi_t = k_B * T_t / q
    phi_Fermi = phi_fermi(T, NA, Eg, ref=phigd)
    H = H_phis(phigd, phi_t, phi_Fermi, Vds)
    G = G_phigd(phigd, Vds, T, NA, Eg)
    H_safe = torch.clamp(H, min=1.0e-15)
    return Agd_t * torch.sqrt(2.0 * q * eps_t * ND_t) * 0.5 * G / H_safe


def Cgd(phigd, Vds, T, NA, ND, Eg, eps_sic, Agd, Coxgd):
    C_JFET = Cjfet(phigd, Vds, T, NA, ND, Eg, eps_sic, Agd)
    Coxgd_t = _as_like(Coxgd, phigd)
    denominator = torch.clamp(Coxgd_t + C_JFET, min=1.0e-30)
    return Coxgd_t * C_JFET / denominator


def Cds_no_PT(Vds, ND, eps_sic, Ads, Vbi):
    Vds = Vds.reshape(-1, 1)
    q = _as_like(1.602176634e-19, Vds)
    ND_t = _as_like(ND, Vds)
    eps_t = _as_like(eps_sic, Vds)
    Ads_t = _as_like(Ads, Vds)
    Vbi_t = _as_like(Vbi, Vds)
    Vds_safe = torch.clamp(Vds, min=0.0)
    denominator = torch.clamp(2.0 * (Vbi_t + Vds_safe), min=1.0e-20)
    return Ads_t * torch.sqrt(q * eps_t * ND_t / denominator)


def Cds(Vds, ND, eps_sic, Ads, Vbi, Vpt):
    Vds = Vds.reshape(-1, 1)
    Vpt_t = torch.clamp(_as_like(Vpt, Vds), min=1.0e-6)
    Vds_effective = torch.minimum(torch.clamp(Vds, min=0.0), Vpt_t)
    return Cds_no_PT(Vds_effective, ND, eps_sic, Ads, Vbi)


def Cgs(Vds, Cgs_const):
    Vds = Vds.reshape(-1, 1)
    return torch.ones_like(Vds) * _as_like(Cgs_const, Vds)


def built_in_potential_4h_sic(T, NA, ND, Eg, ref):
    from Residual import ni_4H_SiC
    q = _as_like(1.602176634e-19, ref)
    k_B = _as_like(1.380649e-23, ref)
    T_t = _as_like(T, ref)
    NA_t = _as_like(NA, ref)
    ND_t = _as_like(ND, ref)
    ni = ni_4H_SiC(T_t, Eg, ref)
    phi_t = k_B * T_t / q
    return phi_t * (torch.log(NA_t) + torch.log(ND_t) - 2.0 * torch.log(ni))
