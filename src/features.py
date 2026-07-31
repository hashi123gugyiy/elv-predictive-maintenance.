"""Feature sets for the AI4I 2020 dataset.

Two variants are exposed so the baseline script can compare them head-to-head:
- RAW: the 5 sensor columns + one-hot encoded machine Type.
- ENGINEERED: RAW plus `power` (torque x rotational speed) and `temp_diff`
  (process temp - air temp), which mirror the actual threshold rules AI4I 2020
  uses to define the PWF and HDF failure modes.

The 5 failure-mode flags (TWF/HDF/PWF/OSF/RNF) are never used as features —
they are components of the target label and including them would be leakage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

RAW_NUMERIC = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["power"] = df["Torque"] * df["Rotational speed"]
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]
    return df


def build_xy(df: pd.DataFrame, engineered: bool) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=LEAKAGE_COLUMNS)
    if engineered:
        df = add_engineered_features(df)

    y = df[TARGET]
    x = df.drop(columns=[TARGET])
    x = pd.get_dummies(x, columns=["Type"], drop_first=True)
    return x, y
