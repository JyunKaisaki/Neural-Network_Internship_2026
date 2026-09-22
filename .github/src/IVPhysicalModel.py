from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Convert a positive physical initial value into the corresponding unconstrained raw softplus parameter.
def _inverse_softplus(value):
    value = torch.as_tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value))


class IVPhysicalParameters(nn.Module):
    def __init__(self, Lch_cm=1.0e-4, T_ref_K=298.15):
        super().__init__()

        # Store the fixed channel length and room-temperature reference.
        self.Lch_cm = float(Lch_cm)
        self.T_ref_K = float(T_ref_K)

        # Define the positive current scaling factor K through logarithmic parameterization.
        self.log_K = nn.Parameter(torch.tensor(13.1, dtype=torch.float32))

        # Define the positive low-field mobility through logarithmic parameterization.
        self.log_mu0 = nn.Parameter(torch.tensor(3.4, dtype=torch.float32))

        # Define the positive channel-length modulation coefficient through logarithmic parameterization.
        self.log_lambda0 = nn.Parameter(torch.tensor(-4.6, dtype=torch.float32))

        # Define the positive carrier saturation velocity through logarithmic parameterization.
        self.log_vsat = nn.Parameter(torch.tensor(16.8, dtype=torch.float32))

        # Define the positive smoothing parameter used by the effective drain voltage.
        self.raw_delta = nn.Parameter(_inverse_softplus(2.0))


    # Ensure that the physical parameter model is used only at room temperature.
    def _check_temperature(self, T_K, tolerance_K=1.0e-3):
        if T_K is None:
            return

        T_K = torch.as_tensor(T_K)

        if not torch.all(torch.abs(T_K - self.T_ref_K) <= tolerance_K):
            raise ValueError("IVPhysicalParameters only supports T = 298.15 K (25 degC).")


    # Convert raw trainable variables into physical quantities.
    def values(self, Vgs, T_K=None):
        self._check_temperature(T_K)

        # Calculate the positive current scaling factor.
        K = torch.exp(self.log_K)

        # Calculate the positive low-field mobility.
        mu_lf = torch.exp(self.log_mu0)

        # Calculate the positive channel-length modulation coefficient.
        lambda_clm = torch.exp(self.log_lambda0)

        # Calculate the positive saturation velocity.
        vsat = torch.exp(self.log_vsat)

        # Calculate the positive Vdseff transition smoothing factor.
        delta = F.softplus(self.raw_delta) + 0.05

        return {
            "K": K,
            "mu_lf": mu_lf,
            "lambda_clm": lambda_clm,
            "vsat": vsat,
            "delta": delta,
            "Lch_cm": self.Lch_cm,
        }


    # Return the fitted room-temperature physical parameters in a readable dictionary.
    def report(self):
        with torch.no_grad():
            return {
                "T_C": self.T_ref_K - 273.15,
                "K": float(torch.exp(self.log_K).item()),
                "mu_lf_cm2_Vs": float(torch.exp(self.log_mu0).item()),
                "lambda_1_V": float(torch.exp(self.log_lambda0).item()),
                "vsat_cm_s": float(torch.exp(self.log_vsat).item()),
                "delta": float((F.softplus(self.raw_delta) + 0.05).item()),
                "Lch_cm": float(self.Lch_cm),
            }


# Save the trained room-temperature Ich physical parameters.
def save_iv_physical_pkl(path, model, *, training_metadata: Optional[Dict[str, Any]] = None):
    checkpoint = {
        "format_version": 3,
        "model_name": "SiC_MOSFET_Physical_Ich_25C",
        "paper_equations": [17, 18, 20, 21],
        "temperature_C": model.T_ref_K - 273.15,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "Lch_cm": model.Lch_cm,
        "T_ref_K": model.T_ref_K,
        "Vdssat_method": "stationarity_condition",
        "training_metadata": training_metadata or {},
    }

    torch.save(checkpoint, Path(path))


# Load a trained room-temperature Ich physical model.
def load_iv_physical_pkl(path, *, device="cpu", dtype=torch.float32):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = IVPhysicalParameters(Lch_cm=checkpoint.get("Lch_cm", 1.0e-4), T_ref_K=checkpoint.get("T_ref_K", 298.15)).to(device=device, dtype=dtype)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return model, checkpoint