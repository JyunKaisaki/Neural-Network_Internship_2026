from pathlib import Path
import re

import numpy as np
import pandas as pd


TEMPERATURE_FOLDERS = {
    -55.0: "-55",
    25.0: "25",
    150.0: "150",
}


def _read_two_column_csv(path, names):
    df = pd.read_csv(path, header=None, names=names, skipinitialspace=True)
    for name in names:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def load_output_characteristics(template_dir):
    template_dir = Path(template_dir)
    folder_map = {
        -55.0: template_dir / "OutputChara-55",
        25.0: template_dir / "OutputChara25",
        150.0: template_dir / "OutputChara150",
    }
    frames = []
    for temp_c, folder in folder_map.items():
        for path in sorted(folder.glob("Vgs_*V.csv")):
            match = re.search(r"Vgs_(-?\d+(?:\.\d+)?)V", path.stem)
            if match is None:
                continue
            vgs = float(match.group(1))
            df = _read_two_column_csv(path, ["Vds", "Ids"])
            df["Vgs"] = vgs
            df["T_C"] = temp_c
            df["T_K"] = temp_c + 273.15
            df["curve_id"] = f"output_{temp_c:g}C_Vgs{vgs:g}V"
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No output-characteristic CSV files were found.")
    return pd.concat(frames, ignore_index=True)[["T_C", "T_K", "Vgs", "Vds", "Ids", "curve_id"]]


def load_transfer_characteristics(template_dir, vds_value=20.0):
    template_dir = Path(template_dir)
    folder_map = {
        -55.0: template_dir / "TransferChara-55",
        25.0: template_dir / "TransferChara25",
        150.0: template_dir / "TransferChara150",
    }
    frames = []
    for temp_c, folder in folder_map.items():
        files = sorted(folder.glob("*.csv"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one transfer CSV in {folder}, found {len(files)}.")
        df = _read_two_column_csv(files[0], ["Vgs", "Ids"])
        df["Vds"] = float(vds_value)
        df["T_C"] = temp_c
        df["T_K"] = temp_c + 273.15
        df["curve_id"] = f"transfer_{temp_c:g}C_Vds{vds_value:g}V"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[["T_C", "T_K", "Vgs", "Vds", "Ids", "curve_id"]]


def load_iv_characteristics(template_dir, transfer_vds=20.0):
    output = load_output_characteristics(template_dir).copy()
    transfer = load_transfer_characteristics(template_dir, vds_value=transfer_vds).copy()
    output["characteristic"] = "output"
    transfer["characteristic"] = "transfer"
    combined = pd.concat([output, transfer], ignore_index=True)
    return combined[["T_C", "T_K", "Vgs", "Vds", "Ids", "characteristic", "curve_id"]]


def load_body_diode_characteristics(template_dir):
    template_dir = Path(template_dir)
    folder_map = {
        -55.0: template_dir / "Ibd-55",
        25.0: template_dir / "Ibd25",
        150.0: template_dir / "Ibd150",
    }
    frames = []
    for temp_c, folder in folder_map.items():
        for path in sorted(folder.glob("Vgs_*V.csv")):
            match = re.search(r"Vgs_(-?\d+(?:\.\d+)?)V", path.stem)
            if match is None:
                continue
            vgs = float(match.group(1))
            df = _read_two_column_csv(path, ["Vds", "Ibd"])
            df["Vgs"] = vgs
            df["T_C"] = temp_c
            df["T_K"] = temp_c + 273.15
            df["curve_id"] = f"ibd_{temp_c:g}C_Vgs{vgs:g}V"
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No body-diode CSV files were found.")
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
            frames.append(df)
    return pd.concat(frames, ignore_index=True)[["Vds", "Capacitance_pF", "capacitance", "range"]]
