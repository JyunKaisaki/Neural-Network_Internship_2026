from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Normalization:
    Vgs_mean: float
    Vgs_std: float
    Vds_mean: float
    Vds_std: float
    Ibd_mean: float
    Ibd_std: float

    @classmethod
    def from_tensors(cls, Vgs, Vds, Ibd, eps=1.0e-12):
        return cls(
            Vgs_mean=float(Vgs.mean().item()),
            Vgs_std=float(Vgs.std().item() + eps),
            Vds_mean=float(Vds.mean().item()),
            Vds_std=float(Vds.std().item() + eps),
            Ibd_mean=float(Ibd.mean().item()),
            Ibd_std=float(Ibd.std().item() + eps),
        )

    def normalize_inputs(self, Vgs, Vds):
        return torch.cat(
            [
                (Vgs - self.Vgs_mean) / self.Vgs_std,
                (Vds - self.Vds_mean) / self.Vds_std,
            ],
            dim=1,
        )

    def normalize_output(self, Ibd):
        return (Ibd - self.Ibd_mean) / self.Ibd_std

    def denormalize_output(self, Ibd_n):
        return Ibd_n * self.Ibd_std + self.Ibd_mean


class BodyDiodeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_ibd_csv(csv_path, vgs_value=None, dtype=torch.float32):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None, names=["Vds", "Ibd"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if vgs_value is None:
        raise ValueError("vgs_value is required for a two-column body-diode CSV file.")
    df["Vgs"] = float(vgs_value)
    Vgs = torch.tensor(df["Vgs"].to_numpy(np.float32), dtype=dtype).reshape(-1, 1)
    Vds = torch.tensor(df["Vds"].to_numpy(np.float32), dtype=dtype).reshape(-1, 1)
    Ibd = torch.tensor(df["Ibd"].to_numpy(np.float32), dtype=dtype).reshape(-1, 1)
    return df, Vgs, Vds, Ibd


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    best_val_loss: float


def _ibd_loss_terms(
    pred_n,
    target_n,
    xb,
    norm,
    current_scale,
    *,
    sign_weight,
    zero_weight,
    zero_width,
    tail_gain,
    tail_center,
    tail_width,
):
    vds_batch = (
        xb[:, 1:2]
        * norm.Vds_std
        + norm.Vds_mean
    )

    tail_weight = (
        1.0
        + tail_gain
        * torch.sigmoid(
            (-vds_batch - tail_center)
            / tail_width
        )
    )

    data_loss = torch.mean(
        tail_weight
        * (pred_n - target_n) ** 2
    )

    pred_current = norm.denormalize_output(
        pred_n
    )

    positive_current = torch.relu(
        pred_current
    )

    sign_loss = torch.mean(
        (
            positive_current
            / current_scale
        ) ** 2
    )

    zero_gate = torch.exp(
        -(
            vds_batch
            / zero_width
        ) ** 2
    )

    zero_loss = torch.mean(
        zero_gate
        * (
            pred_current
            / current_scale
        ) ** 2
    )

    total_loss = (
        data_loss
        + sign_weight
        * sign_loss
        + zero_weight
        * zero_loss
    )

    return (
        total_loss,
        data_loss,
        sign_loss,
        zero_loss,
    )


def train_ibd_network(
    Vgs,
    Vds,
    Ibd,
    *,
    device=None,
    epochs=20000,
    learning_rate=1.0e-4,
    batch_size=128,
    val_fraction=0.20,
    seed=42,
    print_every=500,
    sign_weight=1.0,
    zero_weight=1.0,
    zero_width=0.50,
    tail_gain=2.0,
    tail_center=5.3,
    tail_width=0.30,
):
    set_seed(seed)

    device = get_device() if device is None else device

    norm = Normalization.from_tensors(
        Vgs,
        Vds,
        Ibd,
    )

    X = norm.normalize_inputs(
        Vgs,
        Vds,
    )

    y = norm.normalize_output(
        Ibd,
    )

    dataset = TensorDataset(
        X,
        y,
    )

    n_total = len(dataset)

    n_val = max(
        1,
        int(
            round(
                n_total
                * val_fraction
            )
        ),
    )

    n_train = (
        n_total
        - n_val
    )

    if n_train < 1:
        raise ValueError(
            "No training samples remain."
        )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_set, val_set = random_split(
        dataset,
        [
            n_train,
            n_val,
        ],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=min(
            batch_size,
            n_train,
        ),
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=min(
            batch_size,
            n_val,
        ),
        shuffle=False,
    )

    model = BodyDiodeNN().to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    train_history = []
    val_history = []

    best_val_loss = float(
        "inf"
    )

    best_state = None

    current_scale = max(
        float(
            torch.max(
                torch.abs(Ibd)
            ).item()
        ),
        1.0,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        train_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred_n = model(
                xb
            )

            loss, _, _, _ = _ibd_loss_terms(
                pred_n,
                yb,
                xb,
                norm,
                current_scale,
                sign_weight=sign_weight,
                zero_weight=zero_weight,
                zero_width=zero_width,
                tail_gain=tail_gain,
                tail_center=tail_center,
                tail_width=tail_width,
            )

            loss.backward()

            optimizer.step()

            train_sum += (
                loss.item()
                * len(xb)
            )

            train_count += len(
                xb
            )

        train_loss = (
            train_sum
            / train_count
        )

        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )

                pred_n = model(
                    xb
                )

                val_loss_batch, _, _, _ = _ibd_loss_terms(
                    pred_n,
                    yb,
                    xb,
                    norm,
                    current_scale,
                    sign_weight=sign_weight,
                    zero_weight=zero_weight,
                    zero_width=zero_width,
                    tail_gain=tail_gain,
                    tail_center=tail_center,
                    tail_width=tail_width,
                )

                val_sum += (
                    val_loss_batch.item()
                    * len(xb)
                )

                val_count += len(
                    xb
                )

        val_loss = (
            val_sum
            / val_count
        )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {
                key:
                value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:5d}/{epochs} | "
                f"train = {train_loss:.6e} | "
                f"val = {val_loss:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

        model.to(
            device
        )

    model.eval()

    history = TrainingHistory(
        train_loss=train_history,
        val_loss=val_history,
        best_val_loss=best_val_loss,
    )

    return (
        model,
        norm,
        history,
    )


def predict_ibd(
    model,
    norm,
    Vgs,
    Vds,
    *,
    device=None,
):
    device = (
        next(
            model.parameters()
        ).device
        if device is None
        else device
    )

    Vgs_t = torch.as_tensor(
        Vgs,
        dtype=torch.float32,
        device=device,
    ).reshape(
        -1,
        1,
    )

    Vds_t = torch.as_tensor(
        Vds,
        dtype=torch.float32,
        device=device,
    ).reshape(
        -1,
        1,
    )

    X = norm.normalize_inputs(
        Vgs_t,
        Vds_t,
    )

    model.eval()

    with torch.no_grad():
        Ibd = norm.denormalize_output(
            model(
                X
            )
        )

        Ibd = torch.minimum(
            Ibd,
            torch.zeros_like(
                Ibd
            ),
        )

        Ibd = torch.where(
            Vds_t >= 0.0,
            torch.zeros_like(
                Ibd
            ),
            Ibd,
        )

    return Ibd


def save_ibd_pkl(path, model, norm, *, training_metadata=None):
    checkpoint = {
        "format_version": 4,
        "model_name": "SiC_MOSFET_Body_Diode_ANN",
        "paper_equation": "Ibd = fNN(Vgs, Vds), Eq. (29)",
        "architecture": {
            "input_dim": 2,
            "hidden_dims": [6, 6],
            "output_dim": 1,
            "activation": "tanh",
        },
        "input_names": ["Vgs", "Vds"],
        "output_name": "Ibd",
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "normalization": asdict(norm),
        "training_metadata": training_metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_ibd_pkl(path, *, device=None):
    device = get_device() if device is None else device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("input_dim") != 2 or architecture.get("hidden_dims") != [6, 6] or architecture.get("output_dim") != 1:
        raise ValueError("The checkpoint architecture does not match the room-temperature 2-6-6-1 network.")
    model = BodyDiodeNN().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    norm = Normalization(**checkpoint["normalization"])
    return model, norm, checkpoint


def extract_ibd_ann_parameters(checkpoint):
    state = checkpoint["state_dict"]
    return {
        "W1": state["network.0.weight"].detach().cpu().numpy().copy(),
        "b1": state["network.0.bias"].detach().cpu().numpy().copy(),
        "W2": state["network.2.weight"].detach().cpu().numpy().copy(),
        "b2": state["network.2.bias"].detach().cpu().numpy().copy(),
        "W3": state["network.4.weight"].detach().cpu().numpy().copy(),
        "b3": state["network.4.bias"].detach().cpu().numpy().copy(),
    }


@dataclass
class LumpedChargeParameters:
    tau: float
    TM: float

    def validate(self):
        if self.tau <= 0.0 or self.TM <= 0.0:
            raise ValueError("tau and TM must be positive.")


def simulate_hybrid_body_diode(
    model,
    norm,
    time_s,
    Vgs_waveform,
    Vds_waveform,
    params,
    *,
    device=None,
    initial_qM=None,
):
    params.validate()
    time_s = np.asarray(time_s, dtype=float)
    Vgs_waveform = np.asarray(Vgs_waveform, dtype=float)
    Vds_waveform = np.asarray(Vds_waveform, dtype=float)
    if not (len(time_s) == len(Vgs_waveform) == len(Vds_waveform)):
        raise ValueError("All transient arrays must have the same length.")
    Ibd = predict_ibd(model, norm, Vgs_waveform, Vds_waveform, device=device).cpu().numpy().reshape(-1)
    qE = params.tau * Ibd
    qM = np.zeros_like(qE)
    qM[0] = qE[0] if initial_qM is None else float(initial_qM)
    I_transient = np.zeros_like(qE)
    I_transient[0] = (qE[0] - qM[0]) / params.TM
    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        rhs = -qM[i - 1] / params.tau + (qE[i - 1] - qM[i - 1]) / params.TM
        qM[i] = qM[i - 1] + dt * rhs
        I_transient[i] = (qE[i] - qM[i]) / params.TM
    return {
        "time_s": time_s,
        "qE_C": qE,
        "qM_C": qM,
        "Ibd_static_A": Ibd,
        "I_transient_A": I_transient,
    }
