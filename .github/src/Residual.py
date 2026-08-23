import torch


def Qit_phis(
    phis,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,
):

    q = torch.as_tensor(
        1.602e-19,
        dtype=phis.dtype,
        device=phis.device
    )

    Dit_mid = torch.as_tensor(
        Dit_mid,
        dtype=phis.dtype,
        device=phis.device
    )

    Dit_edge = torch.as_tensor(
        Dit_edge,
        dtype=phis.dtype,
        device=phis.device
    )

    sigma_it = torch.as_tensor(
        sigma_it,
        dtype=phis.dtype,
        device=phis.device
    )

    Ec_minus_Ei = torch.as_tensor(
        Ec_minus_Ei,
        dtype=phis.dtype,
        device=phis.device
    )

    phi_Fermi = phi_fermi(T,NA,Eg, ref=phis)

    # EF - Ei(surface), numerically in eV
    EF_minus_Ei = phis - phi_Fermi

    # Since:
    # Ec - Ei = Ec_minus_Ei
    #
    # EF - Ec
    # = (EF - Ei) - (Ec - Ei)

    EF_minus_Ec = (
        EF_minus_Ei
        - Ec_minus_Ei
    )

    # Ei - Ec = -(Ec - Ei)
    Ei_minus_Ec = -Ec_minus_Ei

    Qit = -q * (
        Dit_mid * EF_minus_Ei
        +
        Dit_edge
        * sigma_it
        * (
            torch.exp(
                EF_minus_Ec / sigma_it
            )
            -
            torch.exp(
                Ei_minus_Ec / sigma_it
            )
        )
    )

    return Qit

def Vfbs(
    Vfb0,
    Cox,
    Qox,
    phis,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,
):
    Qit_p = Qit_phis(    phis,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,)

    Vfbs = Vfb0 - (Qit_p + Qox)/ Cox

    return Vfbs

#######################################


def Nc_4H_SiC(T, ref, Nc_300=1.6e19):
    T = torch.as_tensor(
        T,
        dtype=ref.dtype,
        device=ref.device
    )

    Nc_300 = torch.as_tensor(
        Nc_300,
        dtype=ref.dtype,
        device=ref.device
    )

    return Nc_300 * (T / 300.0) ** 1.5


def Nv_4H_SiC(T, ref, Nv_300=2.5e19):
    T = torch.as_tensor(
        T,
        dtype=ref.dtype,
        device=ref.device
    )

    Nv_300 = torch.as_tensor(
        Nv_300,
        dtype=ref.dtype,
        device=ref.device
    )

    return Nv_300 * (T / 300.0) ** 1.5


def ni_4H_SiC(T, Eg, ref):
    k_B_eV = torch.as_tensor(
        8.617333262e-5,
        dtype=ref.dtype,
        device=ref.device
    )

    T = torch.as_tensor(
        T,
        dtype=ref.dtype,
        device=ref.device
    )

    Eg = torch.as_tensor(
        Eg,
        dtype=ref.dtype,
        device=ref.device
    )

    Nc = Nc_4H_SiC(T, ref)
    Nv = Nv_4H_SiC(T, ref)

    return torch.sqrt(Nc * Nv) * torch.exp(
        -Eg / (2.0 * k_B_eV * T)
    )


def phi_fermi(T, NA, Eg, ref):
    q = torch.as_tensor(
        1.602e-19,
        dtype=ref.dtype,
        device=ref.device
    )

    k_B = torch.as_tensor(
        1.381e-23,
        dtype=ref.dtype,
        device=ref.device
    )

    T = torch.as_tensor(
        T,
        dtype=ref.dtype,
        device=ref.device
    )

    NA = torch.as_tensor(
        NA,
        dtype=ref.dtype,
        device=ref.device
    )

    ni = ni_4H_SiC(T, Eg, ref)

    phi_t = k_B * T / q

    return phi_t * torch.log(
        NA / ni
    )


def H_phis(
    phis,
    phi_t,
    phi_Fermi,
    phi_f,
):
    """
    Returns:
    H : torch.Tensor
        H(phi_s)
    """

    # Keep all parameters on the same dtype/device as phis
    phi_t = torch.as_tensor(
        phi_t,
        dtype=phis.dtype,
        device=phis.device
    )

    phi_Fermi = torch.as_tensor(
        phi_Fermi,
        dtype=phis.dtype,
        device=phis.device
    )

    phi_f = torch.as_tensor(
        phi_f,
        dtype=phis.dtype,
        device=phis.device
    )

    def safe_exp(x):
        return torch.exp(
            torch.clamp(
                x,
                min=-80.0,
                max=80.0
            )
        )
    
    # First term:
    # phi_t * exp(-phis / phi_t)
    term_1 = (
        phi_t
        * safe_exp(
            -phis / phi_t
        )
    )

    # Second term:
    # phis - phi_t
    term_2 = (
        phis
        - phi_t
    )

    # Quasi-Fermi exponential term:
    # exp(-(2*phi_Fermi + phi_f) / phi_t)
    quasi_fermi_term = torch.exp(
        -(2.0 * phi_Fermi + phi_f)
        / phi_t
    )

    # Inversion term:
    # phi_t * exp(phis / phi_t) - phis - phi_t
    inversion_term = (
        phi_t
        * torch.exp(phis / phi_t)
        - phis
        - phi_t
    )

    # Expression inside sqrt
    H_squared = (
        term_1
        + term_2
        + quasi_fermi_term * inversion_term
    )

    # Numerical protection
    H_squared_safe = torch.clamp(
        H_squared,
        min=0.0
    )

    H = torch.sqrt(
        H_squared_safe
    )

    return H