from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value):
    value = torch.as_tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value))


class IVPhysicalParameters(nn.Module):
    def __init__(self, Lch_cm=1.0e-4, T_ref_K=298.15, Vfbs0_init=-1.0):
        super().__init__()
        self.Lch_cm = float(Lch_cm)
        self.T_ref_K = float(T_ref_K)
        self.Vfbs0 = nn.Parameter(torch.tensor(float(Vfbs0_init), dtype=torch.float32))
        self.log_K = nn.Parameter(torch.tensor(13.1, dtype=torch.float32))
        self.log_mu0 = nn.Parameter(torch.tensor(3.4, dtype=torch.float32))
        self.log_lambda0 = nn.Parameter(torch.tensor(-4.6, dtype=torch.float32))
        self.log_vsat = nn.Parameter(torch.tensor(17.1, dtype=torch.float32))
        self.raw_delta = nn.Parameter(_inverse_softplus(2.0))
        
    def values(self, Vgs, T_K=None):
        # Convert trainable variables into room-temperature physical quantities.
        self._check_temperature(T_K)
        K = torch.exp(self.log_K)
        mu_lf = torch.exp(self.log_mu0)
        lambda_clm = torch.exp(self.log_lambda0)
        vsat = torch.exp(self.log_vsat)
        delta = F.softplus(self.raw_delta) + 0.05
        return {"Vfbs0": self.Vfbs0, "K": K, "mu_lf": mu_lf, "lambda_clm": lambda_clm, "vsat": vsat, "delta": delta, "Lch_cm": self.Lch_cm}

    def report(self):
        # Return the fitted room-temperature physical parameters.
        with torch.no_grad():
            return {"T_C": self.T_ref_K - 273.15, "Vfbs0_V": float(self.Vfbs0.item()), "K": float(torch.exp(self.log_K).item()), "mu_lf_cm2_Vs": float(torch.exp(self.log_mu0).item()), "lambda_1_V": float(torch.exp(self.log_lambda0).item()), "vsat_cm_s": float(torch.exp(self.log_vsat).item()), "delta": float((F.softplus(self.raw_delta) + 0.05).item()), "Lch_cm": float(self.Lch_cm)}


def save_iv_physical_pkl(path, model, *, training_metadata: Optional[Dict[str, Any]] = None):
    # Save the trained room-temperature Ich physical parameters.
    checkpoint = {"format_version": 4, "model_name": "SiC_MOSFET_Physical_Ich_25C_TwoStage", "paper_equations": [17, 18, 20, 21], "temperature_C": model.T_ref_K - 273.15, "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "Lch_cm": model.Lch_cm, "T_ref_K": model.T_ref_K, "Vfbs0_init": float(model.Vfbs0.detach().cpu().item()), "Vdssat_method": "stationarity_condition", "training_metadata": training_metadata or {}}
    torch.save(checkpoint, Path(path))


def load_iv_physical_pkl(path, *, device="cpu", dtype=torch.float32):
    # Load the trained room-temperature Ich physical parameters.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = IVPhysicalParameters(Lch_cm=checkpoint.get("Lch_cm", 1.0e-4), T_ref_K=checkpoint.get("T_ref_K", 298.15), Vfbs0_init=checkpoint.get("Vfbs0_init", -1.0)).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
