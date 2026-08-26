import torch


def Qit_phis(
    phis,
    phi_f,
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
    EF_minus_Ei = phis - phi_Fermi - phi_f

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

    arg_edge = (
    EF_minus_Ec
    / sigma_it
)

    arg_mid = (
    Ei_minus_Ec
    / sigma_it
)


    exp_edge = torch.exp(
    torch.clamp(
        arg_edge,
        min=-80.0,
        max=80.0
    )
)

    exp_mid = torch.exp(
    torch.clamp(
        arg_mid,
        min=-80.0,
        max=80.0
    )
)


    Qit = -q * (
    Dit_mid
    * EF_minus_Ei
    +
    Dit_edge
    * sigma_it
    * (
        exp_edge
        - exp_mid
    )
)

    return Qit

def Vfbs(
    Vfbs0,
    Cox,
    Qox,
    phis,
    phi_f,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Ec_minus_Ei,
    T,
    NA,
    Eg,
):

    # Convert parameters to same dtype/device as phis

    Vfbs0 = torch.as_tensor(
        Vfbs0,
        dtype=phis.dtype,
        device=phis.device
    )

    Cox = torch.as_tensor(
        Cox,
        dtype=phis.dtype,
        device=phis.device
    )

    Qox = torch.as_tensor(
        Qox,
        dtype=phis.dtype,
        device=phis.device
    )

    phi_f = torch.as_tensor(
        phi_f,
        dtype=phis.dtype,
        device=phis.device
    )


    # Interface-trap charge

    Qit = Qit_phis(
        phis,
        phi_f,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Ec_minus_Ei,
        T,
        NA,
        Eg,
    )


    # Effective flat-band voltage
    Vfbs_p = (
        Vfbs0
        - (Qit + Qox) / Cox
    )

    return Vfbs_p

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

    return torch.sqrt(Nc) * torch.sqrt(Nv) * torch.exp(
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


    # --------------------------------------------------------
    # Numerically stable exponent arguments
    # --------------------------------------------------------

    acc_arg = (
        -phis / phi_t
    )

    inv_arg = (
        phis
        - 2.0 * phi_Fermi
        - phi_f
    ) / phi_t

    qf_arg = (
        -(2.0 * phi_Fermi + phi_f)
        / phi_t
    )


    # --------------------------------------------------------
    # Safe exponentials
    # --------------------------------------------------------

    exp_acc = torch.exp(
        torch.clamp(
            acc_arg,
            min=-80.0,
            max=80.0
        )
    )

    exp_inv = torch.exp(
        torch.clamp(
            inv_arg,
            min=-80.0,
            max=80.0
        )
    )

    exp_qf = torch.exp(
        torch.clamp(
            qf_arg,
            min=-80.0,
            max=80.0
        )
    )


    # --------------------------------------------------------
    # Stable form of H^2
    # --------------------------------------------------------

    H_squared = (
        phi_t * exp_acc
        + phis
        - phi_t
        + phi_t * exp_inv
        - exp_qf * (phis + phi_t)
    )


    eps_H = torch.as_tensor(
    1e-24,
    dtype=phis.dtype,
    device=phis.device
)

    H_squared_safe = torch.clamp(
    H_squared,
    min=eps_H
)

    H = torch.sqrt(
    H_squared_safe
)

    return H