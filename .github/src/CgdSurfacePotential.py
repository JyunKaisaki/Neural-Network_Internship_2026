from pathlib import Path

import torch
import torch.nn as nn

from phis_init import phi_init


class DeltaPhiGdPINN(nn.Module):
    def __init__(self, vdg_min=0.0, vdg_max=1000.0, vds_min=0.0, vds_max=1000.0):
        super().__init__()
        self.vdg_min = float(vdg_min)
        self.vdg_max = float(vdg_max)
        self.vds_min = float(vds_min)
        self.vds_max = float(vds_max)
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def _normalize(self, value, lower, upper):
        return 2.0 * (value - lower) / (upper - lower) - 1.0

    def forward(self, Vdg, Vds):
        Vdg = Vdg.reshape(-1, 1)
        Vds = Vds.reshape(-1, 1)
        x = torch.cat(
            [
                self._normalize(Vdg, self.vdg_min, self.vdg_max),
                self._normalize(Vds, self.vds_min, self.vds_max),
            ],
            dim=1,
        )
        return self.net(x)


def phi_gd_surface(
    model,
    Vdg,
    Vds,
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
):
    Vdg = Vdg.reshape(-1, 1)
    Vds = Vds.reshape(-1, 1)
    phigd_ini = phi_init(
        Vdg,
        Vds,
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
    return phigd_ini + model(Vdg, Vds)


def save_cgd_pinn_pkl(path, model, *, physical_parameters=None, training_metadata=None):
    checkpoint = {
        "format_version": 1,
        "model_name": "SiC_MOSFET_Cgd_Surface_Potential_PINN",
        "paper_role": "phi_gd surface-potential PINN using the project Vdg convention",
        "architecture": {
            "input_dim": 2,
            "hidden_dims": [8],
            "output_dim": 1,
            "activation": "tanh",
        },
        "domain": {
            "Vdg_min": model.vdg_min,
            "Vdg_max": model.vdg_max,
            "Vds_min": model.vds_min,
            "Vds_max": model.vds_max,
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "physical_parameters": physical_parameters or {},
        "training_metadata": training_metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_cgd_pinn_pkl(path, *, device="cpu", dtype=torch.float32, freeze=False):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    domain = checkpoint["domain"]
    model = DeltaPhiGdPINN(
        domain["Vdg_min"],
        domain["Vdg_max"],
        domain["Vds_min"],
        domain["Vds_max"],
    ).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, checkpoint
