from pathlib import Path
import torch


def build_model_bundle(base_dir=".", output_name="SiC_MOSFET_Model.pkl"):
    base_dir = Path(base_dir)
    files = {
        "surface_potential_PINN": "phis_PINN_IV_refined.pkl",
        "surface_potential_PINN_pretrained": "phis_PINN.pkl",
        "Ich_physical": "Ich_physical.pkl",
        "Cgd_surface_PINN": "phis_Cgd_PINN_refined.pkl",
        "Cgd_surface_PINN_pretrained": "phis_Cgd_PINN.pkl",
        "CV_physical": "CV_physical.pkl",
        "body_diode_NN": "Ibd_NN.pkl",
    }
    bundle = {
        "format_version": 1,
        "model_name": "SiC_MOSFET_Surface_Potential_PINN_Model",
        "components": {},
    }
    for key, filename in files.items():
        path = base_dir / filename
        if path.exists():
            bundle["components"][key] = torch.load(path, map_location="cpu", weights_only=False)
    output_path = base_dir / output_name
    torch.save(bundle, output_path)
    return output_path, sorted(bundle["components"].keys())


def load_model_bundle(path="SiC_MOSFET_Model.pkl", *, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


if __name__ == "__main__":
    output_path, components = build_model_bundle()
    print("Saved =", output_path)
    print("Components =", components)
