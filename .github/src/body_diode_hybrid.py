from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Normalization
# ============================================================

@dataclass
class Normalization:
    Vgs_mean: float
    Vgs_std: float
    Vds_mean: float
    Vds_std: float
    Ibd_mean: float
    Ibd_std: float

    @classmethod
    def from_tensors(
        cls,
        Vgs: torch.Tensor,
        Vds: torch.Tensor,
        Ibd: torch.Tensor,
        eps: float = 1e-12,
    ) -> "Normalization":
        return cls(
            Vgs_mean=float(Vgs.mean().item()),
            Vgs_std=float(Vgs.std().item() + eps),
            Vds_mean=float(Vds.mean().item()),
            Vds_std=float(Vds.std().item() + eps),
            Ibd_mean=float(Ibd.mean().item()),
            Ibd_std=float(Ibd.std().item() + eps),
        )

    def normalize_inputs(
        self,
        Vgs: torch.Tensor,
        Vds: torch.Tensor,
    ) -> torch.Tensor:
        Vgs_n = (Vgs - self.Vgs_mean) / self.Vgs_std
        Vds_n = (Vds - self.Vds_mean) / self.Vds_std
        return torch.cat((Vgs_n, Vds_n), dim=1)

    def normalize_output(self, Ibd: torch.Tensor) -> torch.Tensor:
        return (Ibd - self.Ibd_mean) / self.Ibd_std

    def denormalize_output(self, Ibd_n: torch.Tensor) -> torch.Tensor:
        return Ibd_n * self.Ibd_std + self.Ibd_mean


# ============================================================
# Eq. (29): static body-diode ANN
# ============================================================

class BodyDiodeNN(nn.Module):
    """
    Fully connected network required by the paper excerpt:

        [Vgs, Vds]
             |
          Linear(2, 6)
             |
            Tanh
             |
          Linear(6, 6)
             |
            Tanh
             |
          Linear(6, 1)
             |
            Ibd
    """

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 6),
            nn.Tanh(),
            nn.Linear(6, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# Data
# ============================================================

def load_ibd_csv(
    csv_path: str,
    vgs_col: str = "Vgs",
    vds_col: str = "Vds",
    ibd_col: str = "Ibd",
    dtype: torch.dtype = torch.float32,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load measured static third-quadrant data.

    Required columns:
        Vgs, Vds, Ibd
    """
    df = pd.read_csv(csv_path)

    required = [vgs_col, vds_col, ibd_col]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Required columns are {required}."
        )

    df = df[required].dropna().copy()

    if len(df) < 8:
        raise ValueError(
            "Too few data points. The network can technically run, "
            "but a useful measured-data fit needs more points."
        )

    Vgs = torch.tensor(
        df[vgs_col].to_numpy(dtype=np.float32),
        dtype=dtype,
    ).reshape(-1, 1)

    Vds = torch.tensor(
        df[vds_col].to_numpy(dtype=np.float32),
        dtype=dtype,
    ).reshape(-1, 1)

    Ibd = torch.tensor(
        df[ibd_col].to_numpy(dtype=np.float32),
        dtype=dtype,
    ).reshape(-1, 1)

    return df, Vgs, Vds, Ibd


# ============================================================
# Training
# ============================================================

@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    best_val_loss: float


def train_ibd_network(
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Ibd: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    epochs: int = 5000,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    val_fraction: float = 0.2,
    seed: int = 42,
    print_every: int = 250,
) -> Tuple[BodyDiodeNN, Normalization, TrainingHistory]:
    """
    Train Eq. (29): Ibd = f_NN(Vgs, Vds).
    """
    set_seed(seed)
    device = device or get_device()

    norm = Normalization.from_tensors(Vgs, Vds, Ibd)

    X = norm.normalize_inputs(Vgs, Vds)
    y = norm.normalize_output(Ibd)

    dataset = TensorDataset(X, y)

    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_train = len(dataset) - n_val

    if n_train < 1:
        raise ValueError("No training samples remain after validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=min(batch_size, n_train),
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_size, n_val),
        shuffle=False,
    )

    model = BodyDiodeNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_hist = []
    val_hist = []

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        train_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            train_sum += loss.item() * len(xb)
            train_count += len(xb)

        train_loss = train_sum / train_count

        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                val_sum += loss.item() * len(xb)
                val_count += len(xb)

        val_loss = val_sum / val_count

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:5d}/{epochs} | "
                f"train={train_loss:.6e} | "
                f"val={val_loss:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    history = TrainingHistory(
        train_loss=train_hist,
        val_loss=val_hist,
        best_val_loss=best_val_loss,
    )

    return model, norm, history


# ============================================================
# Static prediction
# ============================================================

def predict_ibd(
    model: BodyDiodeNN,
    norm: Normalization,
    Vgs,
    Vds,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Evaluate Eq. (29): Ibd = f_NN(Vgs, Vds).

    No sign clipping or quadrant masking is imposed here because the paper
    excerpt does not specify such an extra equation. Therefore, the model
    should only be trusted in the voltage domain covered by measured data.
    """
    device = device or next(model.parameters()).device

    Vgs_t = torch.as_tensor(
        Vgs,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    Vds_t = torch.as_tensor(
        Vds,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    X = norm.normalize_inputs(Vgs_t, Vds_t)

    model.eval()

    with torch.no_grad():
        Ibd_n = model(X)
        Ibd = norm.denormalize_output(Ibd_n)

    return Ibd


# ============================================================
# Checkpoint
# ============================================================

def save_ibd_model(
    path: str,
    model: BodyDiodeNN,
    norm: Normalization,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization": asdict(norm),
            "architecture": "2-6-6-1",
            "activation": "tanh",
        },
        path,
    )


def load_ibd_model(
    path: str,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[BodyDiodeNN, Normalization]:
    device = device or get_device()

    checkpoint = torch.load(path, map_location=device)

    model = BodyDiodeNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm = Normalization(**checkpoint["normalization"])

    return model, norm

@dataclass
class LumpedChargeParameters:
    tau: float
    TM: float

    def validate(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be > 0.")
        if self.TM <= 0:
            raise ValueError("TM must be > 0.")


def junction_charge(
    Ibd: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Eq. (28):
        q_E = tau * I_bd
    """
    return tau * Ibd


def base_charge_derivative(
    qM: torch.Tensor,
    qE: torch.Tensor,
    tau: float,
    TM: float,
) -> torch.Tensor:
    """
    Eq. (27):
        0 = dq_M/dt + q_M/tau - (q_E-q_M)/T_M

    Rearranged:
        dq_M/dt = (q_E-q_M)/T_M - q_M/tau
    """
    return (qE - qM) / TM - qM / tau


def transient_diode_current(
    qE: torch.Tensor,
    qM: torch.Tensor,
    TM: float,
) -> torch.Tensor:
    """
    Eq. (26):
        I(t) = (q_E-q_M)/T_M
    """
    return (qE - qM) / TM


def steady_state_base_charge(
    qE: torch.Tensor,
    tau: float,
    TM: float,
) -> torch.Tensor:
    """
    Steady-state solution of Eq. (27), used only to initialize a transient:

        q_M,ss = q_E * tau / (tau + T_M)
    """
    return qE * tau / (tau + TM)


# ============================================================
# Full hybrid transient simulation
# ============================================================

def simulate_hybrid_body_diode(
    model: BodyDiodeNN,
    norm: Normalization,
    time_s,
    Vgs_waveform,
    Vds_waveform,
    params: LumpedChargeParameters,
    *,
    device: Optional[torch.device] = None,
    initial_qM: Optional[float] = None,
    integration: str = "rk4",
) -> Dict[str, np.ndarray]:
    """
    Hybrid model:

        Ibd(Vgs,Vds)          <- Eq. (29)
        qE = tau*Ibd          <- Eq. (28)
        dqM/dt = ...          <- Eq. (27)
        I(t) = ...            <- Eq. (26)

    The supplied Vgs/Vds waveform must use the same voltage convention and
    should remain within the domain for which the static network is valid.
    """
    params.validate()

    device = device or next(model.parameters()).device

    t = np.asarray(time_s, dtype=np.float64)
    Vgs = np.asarray(Vgs_waveform, dtype=np.float64)
    Vds = np.asarray(Vds_waveform, dtype=np.float64)

    if not (len(t) == len(Vgs) == len(Vds)):
        raise ValueError("time_s, Vgs_waveform, Vds_waveform must have equal length.")

    if len(t) < 2:
        raise ValueError("At least two time points are required.")

    dt_array = np.diff(t)
    if np.any(dt_array <= 0):
        raise ValueError("time_s must be strictly increasing.")

    Ibd_static = predict_ibd(
        model,
        norm,
        Vgs,
        Vds,
        device=device,
    ).reshape(-1)

    qE = junction_charge(
        Ibd_static,
        params.tau,
    )

    qM_hist = torch.zeros(
        len(t),
        dtype=torch.float32,
        device=device,
    )

    I_hist = torch.zeros(
        len(t),
        dtype=torch.float32,
        device=device,
    )

    if initial_qM is None:
        qM = steady_state_base_charge(
            qE[0],
            params.tau,
            params.TM,
        )
    else:
        qM = torch.tensor(
            initial_qM,
            dtype=torch.float32,
            device=device,
        )

    qM_hist[0] = qM

    I_hist[0] = transient_diode_current(
        qE[0],
        qM,
        params.TM,
    )

    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])

        qE_0 = qE[i]
        qE_1 = qE[i + 1]
        qE_mid = 0.5 * (qE_0 + qE_1)

        if integration.lower() == "euler":
            k1 = base_charge_derivative(
                qM,
                qE_0,
                params.tau,
                params.TM,
            )
            qM = qM + dt * k1

        elif integration.lower() == "rk4":
            k1 = base_charge_derivative(
                qM,
                qE_0,
                params.tau,
                params.TM,
            )

            k2 = base_charge_derivative(
                qM + 0.5 * dt * k1,
                qE_mid,
                params.tau,
                params.TM,
            )

            k3 = base_charge_derivative(
                qM + 0.5 * dt * k2,
                qE_mid,
                params.tau,
                params.TM,
            )

            k4 = base_charge_derivative(
                qM + dt * k3,
                qE_1,
                params.tau,
                params.TM,
            )

            qM = qM + (dt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )

        else:
            raise ValueError("integration must be 'euler' or 'rk4'.")

        qM_hist[i + 1] = qM

        I_hist[i + 1] = transient_diode_current(
            qE_1,
            qM,
            params.TM,
        )

    return {
        "time_s": t,
        "Vgs_V": Vgs,
        "Vds_V": Vds,
        "Ibd_static_A": Ibd_static.detach().cpu().numpy(),
        "qE_C": qE.detach().cpu().numpy(),
        "qM_C": qM_hist.detach().cpu().numpy(),
        "I_transient_A": I_hist.detach().cpu().numpy(),
    }
