from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CVPhysicalParameters(nn.Module):
    def __init__(self, ND=5.0e16, Agd_init=3.06e-3, Ads_init=4.1e-2, Coxgd_init=1.0e-9, Cgs_init=1100e-12, Vpt_init=1000.0):
        super().__init__()
        self.log_ND = nn.Parameter(torch.log(torch.tensor(float(ND))))
        self.log_Agd = nn.Parameter(torch.log(torch.tensor(float(Agd_init))))
        self.log_Ads = nn.Parameter(torch.log(torch.tensor(float(Ads_init))))
        self.log_Coxgd = nn.Parameter(torch.log(torch.tensor(float(Coxgd_init))))
        self.log_Cgs = nn.Parameter(torch.log(torch.tensor(float(Cgs_init))))
        self.log_Vpt = nn.Parameter(torch.log(torch.tensor(float(Vpt_init))))

    @property
    def ND(self):
        return float(torch.exp(self.log_ND.detach()).cpu().item())

    def values(self, Cox_density=None):
        # Recover trainable and fixed C-V physical values.
        ND = torch.exp(self.log_ND)
        Agd = torch.exp(self.log_Agd)
        Ads = torch.exp(self.log_Ads)
        Coxgd = torch.exp(self.log_Coxgd)
        Cgs_const = torch.exp(self.log_Cgs)
        Vpt = torch.exp(self.log_Vpt)
        return {"ND": ND, "Agd": Agd, "Ads": Ads, "Coxgd": Coxgd, "Cgs_const": Cgs_const, "Vpt": Vpt}

    def report(self, Cox_density=None):
        values = self.values(Cox_density)
        return {key: float(value.item()) if torch.is_tensor(value) else float(value) for key, value in values.items()}


def save_cv_physical_pkl(path, model, *, training_metadata: Optional[Dict[str, Any]] = None):
    # Save the fitted C-V physical parameters.
    checkpoint = {"format_version": 4, "model_name": "SiC_MOSFET_Physical_CV_25C", "paper_equations": [22, 23, 24, 25], "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "ND": model.ND, "training_metadata": training_metadata or {}}
    torch.save(checkpoint, Path(path))


def load_cv_physical_pkl(path, *, device="cpu", dtype=torch.float32):
    # Load the fitted C-V physical parameters.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CVPhysicalParameters(ND=checkpoint.get("ND", 5.0e16)).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
