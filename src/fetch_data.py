"""Fetch the AI4I 2020 Predictive Maintenance dataset (UCI id=601) and cache it locally as CSV."""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "ai4i2020.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {out_path}")
    return df


if __name__ == "__main__":
    fetch()
