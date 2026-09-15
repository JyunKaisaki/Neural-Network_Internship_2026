from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CVPhysicalParameters(nn.Module):
    def __init__(self, ND=5.0e16):
        super().__init__()
        self.ND = float(ND)
        self.log_Agd = nn.Parameter(torch.log(torch.tensor(3.15e-2)))
        self.log_Ads = nn.Parameter(torch.log(torch.tensor(2.97e-2)))
        self.log_Coxgd_scale = nn.Parameter(torch.tensor(0.0))
        self.log_Cgs = nn.Parameter(torch.log(torch.tensor(1100e-12)))
        self.log_Vpt = nn.Parameter(torch.log(torch.tensor(450.0)))

    def values(self, Cox_density):
        Agd = torch.exp(self.log_Agd)
        Ads = torch.exp(self.log_Ads)
        Coxgd = Agd * Cox_density * torch.exp(self.log_Coxgd_scale)
        Cgs_const = torch.exp(self.log_Cgs)
        Vpt = torch.exp(self.log_Vpt)
        return {
            "ND": self.ND,
            "Agd": Agd,
            "Ads": Ads,
            "Coxgd": Coxgd,
            "Cgs_const": Cgs_const,
            "Vpt": Vpt,
        }


def save_cv_physical_pkl(path, model, *, training_metadata: Optional[Dict[str, Any]] = None):
    checkpoint = {
        "format_version": 1,
        "model_name": "SiC_MOSFET_Physical_CV",
        "paper_equations": [22, 23, 24, 25],
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "ND": model.ND,
        "training_metadata": training_metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_cv_physical_pkl(path, *, device="cpu", dtype=torch.float32):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CVPhysicalParameters(ND=checkpoint.get("ND", 5.0e16)).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
