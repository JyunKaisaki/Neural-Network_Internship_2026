from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value):
    value = torch.as_tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value))


class IVPhysicalParameters(nn.Module):
    def __init__(self, Lch_cm=1.0e-4, T_ref_K=298.15):
        super().__init__()
        self.Lch_cm = float(Lch_cm)
        self.T_ref_K = float(T_ref_K)
        self.log_K = nn.Parameter(torch.tensor(13.1))
        self.log_mu0 = nn.Parameter(torch.tensor(3.4))
        self.mu_t1 = nn.Parameter(torch.tensor(-0.30))
        self.mu_t2 = nn.Parameter(torch.tensor(0.00))
        self.log_lambda0 = nn.Parameter(torch.tensor(-4.6))
        self.lambda_t1 = nn.Parameter(torch.tensor(0.00))
        self.log_vsat = nn.Parameter(torch.tensor(16.8))
        self.log_Dit_edge0 = nn.Parameter(torch.log(torch.tensor(2.0e13)))
        self.Dit_edge_t1 = nn.Parameter(torch.tensor(-0.05))
        self.Dit_edge_t2 = nn.Parameter(torch.tensor(0.00))
        self.Vfbs0_ref = nn.Parameter(torch.tensor(-1.0))
        self.Vfbs0_t1 = nn.Parameter(torch.tensor(0.00))
        self.Vfbs0_t2 = nn.Parameter(torch.tensor(0.00))
        self.vth0 = nn.Parameter(torch.tensor(2.9))
        self.vth_t1 = nn.Parameter(torch.tensor(-0.25))
        self.raw_vdsat_scale = nn.Parameter(_inverse_softplus(0.65))
        self.raw_delta = nn.Parameter(_inverse_softplus(2.0))

    def temperature_coordinate(self, T_K):
        return (T_K - self.T_ref_K) / 100.0

    def values(self, Vgs, T_K):
        x = self.temperature_coordinate(T_K)
        K = torch.exp(self.log_K)
        mu_lf = torch.exp(self.log_mu0 + self.mu_t1 * x + self.mu_t2 * x**2)
        lambda_clm = torch.exp(self.log_lambda0 + self.lambda_t1 * x)
        vsat = torch.exp(self.log_vsat)
        Dit_edge0 = torch.exp(self.log_Dit_edge0)
        Dit_multiplier = torch.clamp(1.0 + self.Dit_edge_t1 * x + self.Dit_edge_t2 * x**2, min=0.1)
        Dit_edge = Dit_edge0 * Dit_multiplier
        Vfbs0 = self.Vfbs0_ref + self.Vfbs0_t1 * x + self.Vfbs0_t2 * x**2
        vth = self.vth0 + self.vth_t1 * x
        vdsat_scale = F.softplus(self.raw_vdsat_scale) + 1.0e-4
        Vdssat = 0.05 + vdsat_scale * F.softplus(Vgs - vth)
        delta = F.softplus(self.raw_delta) + 0.05
        return {
            "K": K,
            "mu_lf": mu_lf,
            "lambda_clm": lambda_clm,
            "vsat": vsat,
            "Dit_edge": Dit_edge,
            "Vfbs0": Vfbs0,
            "vth": vth,
            "Vdssat": Vdssat,
            "delta": delta,
            "Lch_cm": self.Lch_cm,
        }

    def report(self, temperatures_c=(-55.0, 25.0, 150.0), vgs=15.0):
        rows = []
        device = self.log_K.device
        dtype = self.log_K.dtype
        with torch.no_grad():
            for temp_c in temperatures_c:
                T_K = torch.tensor([[temp_c + 273.15]], dtype=dtype, device=device)
                Vgs = torch.tensor([[vgs]], dtype=dtype, device=device)
                values = self.values(Vgs, T_K)
                rows.append({
                    "T_C": float(temp_c),
                    "K": float(values["K"].item()),
                    "mu_lf_cm2_Vs": float(values["mu_lf"].item()),
                    "lambda_1_V": float(values["lambda_clm"].item()),
                    "vsat_cm_s": float(values["vsat"].item()),
                    "Dit_edge_cm-2_eV-1": float(values["Dit_edge"].item()),
                    "Vfbs0_V": float(values["Vfbs0"].item()),
                    "Vth_V": float(values["vth"].item()),
                    "Vdssat_V_at_report_Vgs": float(values["Vdssat"].item()),
                    "delta": float(values["delta"].item()),
                })
        return rows


def save_iv_physical_pkl(path, model, *, training_metadata: Optional[Dict[str, Any]] = None):
    checkpoint = {
        "format_version": 1,
        "model_name": "SiC_MOSFET_Physical_Ich",
        "paper_equations": [17, 18, 20, 21],
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "Lch_cm": model.Lch_cm,
        "T_ref_K": model.T_ref_K,
        "training_metadata": training_metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_iv_physical_pkl(path, *, device="cpu", dtype=torch.float32):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = IVPhysicalParameters(
        Lch_cm=checkpoint.get("Lch_cm", 1.0e-4),
        T_ref_K=checkpoint.get("T_ref_K", 298.15),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
