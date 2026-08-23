import torch
from Residual import phi_fermi

def phi_init(
    Vgs,
    phi_f, # Surface potential
    T,     # Temperature
    NA,    # Acceptor concentration
    eps_sic, # Permittivity of SiC
    Cox,   # Oxide capacitance
    Vfbs0, # Flat-band voltage at T=0K
    Dit_mid, # Interface trap density at mid-gap
    Dit_edge, # Interface trap density at band edges
    sigma_it, # Standard deviation of interface trap energy distribution
    Eg,   # Bandgap energy
):
    # 1. Thermal Voltage(V) phi_t = k_B * T / q 
    q = torch.as_tensor(1.602e-19, dtype=Vgs.dtype, device=Vgs.device)
    k_B = torch.as_tensor(1.381e-23, dtype=Vgs.dtype, device=Vgs.device)
    T = torch.as_tensor(T, dtype=Vgs.dtype, device=Vgs.device)
    phi_t = k_B * T / q


    # 2. Constant Gamma. 
    #    NA: accpetor doping concentration, from Product manufacturer
    #    eps_sic: permittivity of SiC:https://www.ioffe.ru/SVA/NSM/Semicond/SiC/basic.html), or from Product manufacturer
    #    Cox: oxide capacitance, from Product manufacturer(Cox = eps_ox / tox, tox: oxide thickness)

    #gamma = torch.sqrt((2 * eps_sic * 1.602e-19 * NA)/phi_t) / Cox
    NA = torch.as_tensor(NA, dtype=Vgs.dtype, device=Vgs.device)
    eps_sic = torch.as_tensor(eps_sic, dtype=Vgs.dtype, device=Vgs.device)
    Cox = torch.as_tensor(Cox, dtype=Vgs.dtype, device=Vgs.device)
    Vfbs0 = torch.as_tensor(Vfbs0, dtype=Vgs.dtype, device=Vgs.device)
    Dit_mid = torch.as_tensor(Dit_mid, dtype=Vgs.dtype, device=Vgs.device)
    Dit_edge = torch.as_tensor(Dit_edge, dtype=Vgs.dtype, device=Vgs.device)
    sigma_it = torch.as_tensor(sigma_it, dtype=Vgs.dtype, device=Vgs.device)
    Eg = torch.as_tensor(Eg, dtype=Vgs.dtype, device=Vgs.device)
    phi_f = torch.as_tensor(phi_f, dtype=Vgs.dtype, device=Vgs.device)
    phi_Fermi = phi_fermi(T, NA, Eg, ref=Vgs)
    gamma = torch.sqrt(
    torch.clamp(2.0 * eps_sic * q * NA, min=0.0)
    ) / Cox



    # 3. Effective gate potential: uG
    #    Quasi-Fermi potential(phi_f)'s difference at the semiconductor surface: uf
    uG = (Vgs - Vfbs0) / phi_t
    uf = phi_f / phi_t

    # 4. α: Midgap interface-trap correction(?, inferredfrom GPT)
    alpha = 1.0 + q * Dit_mid / Cox

    # 5. Regions of SiC MOSFET operation
    #    Accumulation: If uG <= 0
    #    phis_init = (-2.0 * phi_t * torch.log(1.0 - uG / gamma))
    acc_arg = (
        1.0
        - torch.sqrt(phi_t) * uG / gamma
    )

    acc_arg_safe = torch.clamp(
        acc_arg,
        min=1e-30
    )

    phis_acc = (
        -2.0
        * phi_t
        * torch.log(acc_arg_safe)
    )

    
    #    Depletion: If uG > 0
    dep_arg = (
        gamma**2
        + 4.0 * alpha * uG
    )

    dep_arg_safe = torch.clamp(
        dep_arg,
        min=0.0
    )

    u_dep = (
        (
            -gamma
            + torch.sqrt(dep_arg_safe)
        )
        /
        (2.0 * alpha)
    )**2

    u_itc = (q* Dit_edge* sigma_it* torch.exp((-Eg / 2.0 - phi_Fermi)/ sigma_it)/ (phi_t * Cox))
    u_it0 = (uf + sigma_it / phi_t * torch.log(gamma / u_itc))
    u_si0 = (uf + 2.0 * phi_Fermi / phi_t)

    #    Weak Inversion: if uit0 <= u_dep
    #u_it = (uf + sigma_it / phi_t * torch.log((uG - alpha * u_it0 - gamma * torch.sqrt(u_it0) + gamma) / u_itc))
    u_it0_safe = torch.clamp(
        u_it0,
        min=0.0
    )

    weak_arg = (
        uG
        - alpha * u_it0
        - gamma * torch.sqrt(u_it0_safe)
        + gamma
    ) / u_itc

    weak_valid = (
        (u_it0 >= 0.0)
        & (weak_arg > 0.0)
    )

    weak_arg_safe = torch.where(
        weak_valid,
        weak_arg,
        torch.ones_like(weak_arg)
    )

    u_it_calc = (
        uf
        + sigma_it / phi_t
        * torch.log(weak_arg_safe)
    )

    u_it = torch.where(
        weak_valid,
        u_it_calc,
        torch.full_like(
            u_it_calc,
            float("inf")
        )
    )

    #    Strong Inversion: if usi0 <= u_dep
    #u_si = u_si0 + torch.log(((uG - u_si0) / gamma)**2 - u_si0 + 1)
    strong_arg = (
    ((uG - u_si0) / gamma)**2
    - u_si0
    + 1.0
)

    strong_valid = (
        strong_arg > 0.0
    )

    strong_arg_safe = torch.where(
        strong_valid,
        strong_arg,
        torch.ones_like(strong_arg)
    )

    u_si_calc = (
        u_si0
        + torch.log(strong_arg_safe)
    )

    u_si = torch.where(
        strong_valid,
        u_si_calc,
        torch.full_like(
            u_si_calc,
            float("inf")
        )
    )

    # FINAL: phis_init extraction
    phis_positive = phi_t * torch.minimum(u_dep, torch.minimum(u_it, u_si))

    phis_init = torch.where(
    uG > 0,
    phis_positive,
    phis_acc
)


    return phis_init