from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOM_TEMPERATURE_C = 25.0
ROOM_TEMPERATURE_K = ROOM_TEMPERATURE_C + 273.15


def _read_two_column_csv(path, names):
    df = pd.read_csv(path, header=None, names=names, skipinitialspace=True)
    for name in names:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def load_output_characteristics(template_dir):
    template_dir = Path(template_dir)
    folder = template_dir / "OutputChara25"
    frames = []
    for path in sorted(folder.glob("Vgs_*V.csv")):
        match = re.search(r"Vgs_(-?\d+(?:\.\d+)?)V", path.stem)
        if match is None:
            continue
        vgs = float(match.group(1))
        df = _read_two_column_csv(path, ["Vds", "Ids"])
        df["Vgs"] = vgs
        df["T_C"] = ROOM_TEMPERATURE_C
        df["T_K"] = ROOM_TEMPERATURE_K
        df["curve_id"] = f"output_25C_Vgs{vgs:g}V"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No output-characteristic CSV files were found in {folder}.")
    return pd.concat(frames, ignore_index=True)[["T_C", "T_K", "Vgs", "Vds", "Ids", "curve_id"]]


def load_transfer_characteristics(template_dir, vds_value=20.0):
    template_dir = Path(template_dir)
    folder = template_dir / "TransferChara25"
    files = sorted(folder.glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one transfer CSV in {folder}, found {len(files)}.")
    df = _read_two_column_csv(files[0], ["Vgs", "Ids"])
    df["Vds"] = float(vds_value)
    df["T_C"] = ROOM_TEMPERATURE_C
    df["T_K"] = ROOM_TEMPERATURE_K
    df["curve_id"] = f"transfer_25C_Vds{vds_value:g}V"
    return df[["T_C", "T_K", "Vgs", "Vds", "Ids", "curve_id"]]


def load_iv_characteristics(template_dir, transfer_vds=20.0):
    output = load_output_characteristics(template_dir).copy()
    transfer = load_transfer_characteristics(template_dir, vds_value=transfer_vds).copy()
    output["characteristic"] = "output"
    transfer["characteristic"] = "transfer"
    combined = pd.concat([output, transfer], ignore_index=True)
    return combined[["T_C", "T_K", "Vgs", "Vds", "Ids", "characteristic", "curve_id"]]


def load_body_diode_characteristics(template_dir):
    template_dir = Path(template_dir)
    folder = template_dir / "Ibd25"
    frames = []
    for path in sorted(folder.glob("Vgs_*V.csv")):
        match = re.search(r"Vgs_(-?\d+(?:\.\d+)?)V", path.stem)
        if match is None:
            continue
        vgs = float(match.group(1))
        df = _read_two_column_csv(path, ["Vds", "Ibd"])
        df["Vgs"] = vgs
        df["T_C"] = ROOM_TEMPERATURE_C
        df["T_K"] = ROOM_TEMPERATURE_K
        df["curve_id"] = f"ibd_25C_Vgs{vgs:g}V"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No body-diode CSV files were found in {folder}.")
    return pd.concat(frames, ignore_index=True)[["T_C", "T_K", "Vgs", "Vds", "Ibd", "curve_id"]]


def load_cv_characteristics(template_dir):
    template_dir = Path(template_dir)
    folders = {
        "low": template_dir / "CV_Vds0-200V",
        "high": template_dir / "CV_Vds0-1000V",
    }
    file_map = {
        "Ciss": ("CissLowV.csv", "CissHighV.csv"),
        "Coss": ("CossLowV.csv", "CossHighV.csv"),
        "Crss": ("CrssLowV.csv", "CrssHighV.csv"),
    }
    frames = []
    for capacitance, (low_name, high_name) in file_map.items():
        for range_name, file_name in [("low", low_name), ("high", high_name)]:
            path = folders[range_name] / file_name
            df = _read_two_column_csv(path, ["Vds", "Capacitance_pF"])
            df["capacitance"] = capacitance
            df["range"] = range_name
            df["T_C"] = ROOM_TEMPERATURE_C
            df["T_K"] = ROOM_TEMPERATURE_K
            frames.append(df)
    return pd.concat(frames, ignore_index=True)[["T_C", "T_K", "Vds", "Capacitance_pF", "capacitance", "range"]]
