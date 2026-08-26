import torch

from Residual import (
    H_phis,
    phi_fermi,
)


# ============================================================
# G(phi_gd)
#
# Paper Eq. (24)
#
# G(phi_gd)
# =
# 1
# - exp(-phi_gd / phi_t)
# + exp(-(2*phi_F + Vds)/phi_t)
#   * (exp(phi_gd/phi_t) - 1)
# ============================================================

def G_phigd(
    phigd,
    Vds,
    T,
    NA,
    Eg,
):

    phigd = phigd.reshape(
        -1,
        1
    )

    Vds = Vds.reshape(
        -1,
        1
    )


    q = phigd.new_tensor(
        1.602176634e-19
    )

    k_B = phigd.new_tensor(
        1.380649e-23
    )

    T_t = phigd.new_tensor(
        T
    )


    # --------------------------------------------------------
    # Thermal voltage
    # --------------------------------------------------------

    phi_t = (
        k_B
        * T_t
        / q
    )


    # --------------------------------------------------------
    # Fermi potential
    # --------------------------------------------------------

    phi_Fermi = phi_fermi(
        T,
        NA,
        Eg,
        ref=phigd,
    )


    # --------------------------------------------------------
    # Exponential arguments
    # --------------------------------------------------------

    arg_1 = (
        -phigd
        / phi_t
    )


    arg_2 = (
        -(
            2.0
            * phi_Fermi
            + Vds
        )
        / phi_t
    )


    arg_3 = (
        phigd
        / phi_t
    )


    # --------------------------------------------------------
    # Numerical protection
    #
    # This does not change Eq. (24);
    # it only avoids floating-point overflow.
    # --------------------------------------------------------

    exp_1 = torch.exp(
        torch.clamp(
            arg_1,
            min=-80.0,
            max=80.0,
        )
    )


    exp_2 = torch.exp(
        torch.clamp(
            arg_2,
            min=-80.0,
            max=80.0,
        )
    )


    exp_3 = torch.exp(
        torch.clamp(
            arg_3,
            min=-80.0,
            max=80.0,
        )
    )


    # ========================================================
    # Eq. (24)
    # ========================================================

    G = (
        1.0
        - exp_1
        + exp_2
        * (
            exp_3
            - 1.0
        )
    )


    return G


# ============================================================
# CJFET
#
# Paper Eq. (23)
#
# CJFET =
#
# Agd * sqrt(2*q*eps_SiC*ND)
# -------------------------------- * G(phi_gd) / H(phi_gd)
#                 2
#
# IMPORTANT:
# No abs() is used because Eq. (23) does not contain abs().
# ============================================================

def Cjfet(
    phigd,
    Vds,
    T,
    NA,
    ND,
    Eg,
    eps_sic,
    Agd,
):

    phigd = phigd.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)

    q = phigd.new_tensor(
        1.602176634e-19
    )

    k_B = phigd.new_tensor(
        1.380649e-23
    )

    T_t = phigd.new_tensor(T)

    ND_t = phigd.new_tensor(ND)

    eps_t = phigd.new_tensor(
        eps_sic
    )

    Agd_t = phigd.new_tensor(
        Agd
    )


    phi_t = (
        k_B
        * T_t
        / q
    )


    phi_Fermi = phi_fermi(
        T,
        NA,
        Eg,
        ref=phigd,
    )


    H = H_phis(
        phigd,
        phi_t,
        phi_Fermi,
        Vds,
    )


    G = G_phigd(
        phigd,
        Vds,
        T,
        NA,
        Eg,
    )


    H_safe = torch.clamp(
        H,
        min=1e-15,
    )


    # Eq. (23)
    C_JFET = (
        Agd_t
        * torch.sqrt(
            2.0
            * q
            * eps_t
            * ND_t
        )
        / 2.0
        * G
        / H_safe
    )


    return C_JFET


# ============================================================
# Cgd
#
# Paper Eq. (22)
#
# Cgd =
#
# Coxgd * CJFET
# -----------------
# Coxgd + CJFET
# ============================================================

def Cgd(
    phigd,
    Vds,
    T,
    NA,
    ND,
    Eg,
    eps_sic,
    Agd,
    Coxgd,
):

    C_JFET = Cjfet(
        phigd,
        Vds,
        T,
        NA,
        ND,
        Eg,
        eps_sic,
        Agd,
    )


    Coxgd_t = phigd.new_tensor(
        Coxgd
    )


    Cgd_value = (
        Coxgd_t
        * C_JFET
        /
        (
            Coxgd_t
            + C_JFET
        )
    )


    return Cgd_value


# ============================================================
# Cds without punch-through
#
# Paper Eq. (25)
#
# Cds =
#
# Ads * sqrt(
#     q * eps_SiC * ND
#     ----------------
#     2 * (Vbi + Vds)
# )
# ============================================================

def Cds_no_PT(
    Vds,
    ND,
    eps_sic,
    Ads,
    Vbi,
):

    Vds = Vds.reshape(
        -1,
        1
    )


    q = Vds.new_tensor(
        1.602176634e-19
    )

    ND_t = Vds.new_tensor(
        ND
    )

    eps_t = Vds.new_tensor(
        eps_sic
    )

    Ads_t = Vds.new_tensor(
        Ads
    )

    Vbi_t = Vds.new_tensor(
        Vbi
    )


    Vds_safe = torch.clamp(
        Vds,
        min=0.0,
    )


    # ========================================================
    # Eq. (25)
    # ========================================================

    Cds_value = (
        Ads_t
        * torch.sqrt(
            q
            * eps_t
            * ND_t
            /
            (
                2.0
                * (
                    Vbi_t
                    + Vds_safe
                )
            )
        )
    )


    return Cds_value


# ============================================================
# Cds with punch-through
#
# Screenshot:
# PT effect is also considered for Cds.
#
# Therefore after Vds reaches Vpt, the depletion width
# is no longer allowed to increase.
# ============================================================

def Cds(
    Vds,
    ND,
    eps_sic,
    Ads,
    Vbi,
    Vpt,
):

    Vds = Vds.reshape(
        -1,
        1
    )


    Vpt_t = Vds.new_tensor(
        Vpt
    )


    Vds_effective = torch.clamp(
        Vds,
        min=0.0,
        max=Vpt_t.item(),
    )


    return Cds_no_PT(
        Vds_effective,
        ND,
        eps_sic,
        Ads,
        Vbi,
    )


# ============================================================
# Cgs
#
# Screenshot:
# Cgs is modeled as a constant.
# ============================================================

def Cgs(
    Vds,
    Cgs_const,
):

    Vds = Vds.reshape(
        -1,
        1
    )


    Cgs_t = Vds.new_tensor(
        Cgs_const
    )


    return (
        torch.ones_like(
            Vds
        )
        * Cgs_t
    )