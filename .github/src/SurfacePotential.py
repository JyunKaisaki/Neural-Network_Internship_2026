import torch
import torch.nn as nn

from phis_init import phi_init


# Surface-potential PINN
class DeltaPhisPINN(nn.Module):

    def __init__(
        self,
        vgs_min=-10.0,
        vgs_max=30.0,
        phif_min=0.0,
        phif_max=15.0,
    ):

        super().__init__()

        self.vgs_min = vgs_min
        self.vgs_max = vgs_max

        self.phif_min = phif_min
        self.phif_max = phif_max

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

        nn.init.zeros_(
            self.net[2].weight
        )

        nn.init.zeros_(
            self.net[2].bias
        )


    def forward(
        self,
        Vgs,
        phi_f,
    ):

        Vgs = Vgs.reshape(-1, 1)

        phi_f = phi_f.reshape(-1, 1)


        Vgs_norm = (
            2.0
            * (
                Vgs
                - self.vgs_min
            )
            / (
                self.vgs_max
                - self.vgs_min
            )
            - 1.0
        )


        phif_norm = (
            2.0
            * (
                phi_f
                - self.phif_min
            )
            / (
                self.phif_max
                - self.phif_min
            )
            - 1.0
        )


        x = torch.cat(
            [
                Vgs_norm,
                phif_norm,
            ],
            dim=1,
        )


        return self.net(x)


# Load trained surface-potential PINN
def load_surface_potential_model(
    checkpoint,
    device,
    dtype,
    vgs_min=-10.0,
    vgs_max=30.0,
    phif_min=0.0,
    phif_max=15.0,
    freeze=True,
):

    model = DeltaPhisPINN(
        vgs_min=vgs_min,
        vgs_max=vgs_max,
        phif_min=phif_min,
        phif_max=phif_max,
    ).to(
        device=device,
        dtype=dtype,
    )


    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )


    model.load_state_dict(
        state_dict,
        strict=True,
    )


    model.eval()


    if freeze:

        for parameter in model.parameters():

            parameter.requires_grad_(False)


    return model


# Predict physical surface potential
@torch.no_grad()
def predict_phis(
    model,
    Vgs,
    phi_f,
    *,
    T,
    NA,
    eps_sic,
    Cox,
    Vfbs0,
    Dit_mid,
    Dit_edge,
    sigma_it,
    Eg,
    dtype,
    device,
):

    Vgs = torch.as_tensor(
        Vgs,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)


    phi_f = torch.as_tensor(
        phi_f,
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)


    phis_init_value = phi_init(
        Vgs,
        phi_f,
        T,
        NA,
        eps_sic,
        Cox,
        Vfbs0,
        Dit_mid,
        Dit_edge,
        sigma_it,
        Eg,
    ).reshape(-1, 1)


    delta_phis = model(
        Vgs,
        phi_f,
    )


    phis = (
        phis_init_value
        + delta_phis
    )


    return phis