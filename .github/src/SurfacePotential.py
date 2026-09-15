from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from phis_init import phi_init


class DeltaPhisPINN(nn.Module):
    def __init__(
        self,
        vgs_min=-10.0,
        vgs_max=30.0,
        phif_min=0.0,
        phif_max=20.0,
        normalize_inputs=True,
    ):
        super().__init__()
        self.vgs_min = float(vgs_min)
        self.vgs_max = float(vgs_max)
        self.phif_min = float(phif_min)
        self.phif_max = float(phif_max)
        self.normalize_inputs = bool(normalize_inputs)
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.net[2].bias)

    def _normalize(self, Vgs, phi_f):
        # Map the two PINN inputs to approximately [-1, 1].
        Vgs_n = 2.0 * (Vgs - self.vgs_min) / (self.vgs_max - self.vgs_min) - 1.0
        phi_f_n = 2.0 * (phi_f - self.phif_min) / (self.phif_max - self.phif_min) - 1.0
        return Vgs_n, phi_f_n

    def forward(self, Vgs, phi_f):
        Vgs = Vgs.reshape(-1, 1)
        phi_f = phi_f.reshape(-1, 1)
        if self.normalize_inputs:
            Vgs, phi_f = self._normalize(Vgs, phi_f)
        return self.net(torch.cat([Vgs, phi_f], dim=1))


def surface_potential(
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
):
    # Add the PINN correction to the analytical initial surface potential.
    Vgs = Vgs.reshape(-1, 1)
    phi_f = phi_f.reshape(-1, 1)
    phis_ini = phi_init(
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
    return phis_ini + model(Vgs, phi_f)


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
    dtype=torch.float64,
    device="cpu",
):
    # Evaluate a trained surface-potential PINN.
    Vgs = torch.as_tensor(Vgs, dtype=dtype, device=device).reshape(-1, 1)
    phi_f = torch.as_tensor(phi_f, dtype=dtype, device=device).reshape(-1, 1)
    model.eval()
    return surface_potential(
        model,
        Vgs,
        phi_f,
        T=T,
        NA=NA,
        eps_sic=eps_sic,
        Cox=Cox,
        Vfbs0=Vfbs0,
        Dit_mid=Dit_mid,
        Dit_edge=Dit_edge,
        sigma_it=sigma_it,
        Eg=Eg,
    )


def save_surface_potential_pkl(
    path,
    model,
    *,
    physical_parameters: Optional[Dict[str, Any]] = None,
    training_metadata: Optional[Dict[str, Any]] = None,
):
    # Save the trained surface-potential PINN as a structured PyTorch checkpoint.
    path = Path(path)
    checkpoint = {
        "format_version": 2,
        "model_name": "SiC_MOSFET_Surface_Potential_PINN",
        "paper_role": "Delta_phi_s_PINN",
        "architecture": {
            "input_dim": 2,
            "hidden_dims": [8],
            "output_dim": 1,
            "activation": "tanh",
            "normalize_inputs": model.normalize_inputs,
            "vgs_range": [model.vgs_min, model.vgs_max],
            "phif_range": [model.phif_min, model.phif_max],
        },
        "input_names": ["Vgs", "phi_f"],
        "output_name": "delta_phi_s",
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "physical_parameters": physical_parameters or {},
        "training_metadata": training_metadata or {},
    }
    torch.save(checkpoint, path)


def load_surface_potential_pkl(path, *, device="cpu", dtype=torch.float64, freeze=False):
    # Reconstruct a surface-potential PINN from a structured checkpoint.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", {})
    vgs_range = architecture.get("vgs_range", [-10.0, 30.0])
    phif_range = architecture.get("phif_range", [0.0, 20.0])
    normalize_inputs = architecture.get("normalize_inputs", False)
    model = DeltaPhisPINN(
        vgs_min=vgs_range[0],
        vgs_max=vgs_range[1],
        phif_min=phif_range[0],
        phif_max=phif_range[1],
        normalize_inputs=normalize_inputs,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, checkpoint


def load_surface_potential_model(checkpoint, device, dtype, freeze=True):
    # Load either a structured .pkl checkpoint or a legacy raw state dictionary.
    checkpoint = Path(checkpoint)
    if checkpoint.suffix.lower() == ".pkl":
        model, _ = load_surface_potential_pkl(checkpoint, device=device, dtype=dtype, freeze=freeze)
        return model
    model = DeltaPhisPINN(normalize_inputs=False).to(device=device, dtype=dtype)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model
