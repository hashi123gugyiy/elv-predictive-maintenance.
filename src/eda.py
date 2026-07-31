"""First exploratory pass over the AI4I 2020 dataset: shape, failure rate, feature distributions."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isna().sum())

    print("\n" + "=" * 60)
    print("MACHINE TYPE COUNTS")
    print("=" * 60)
    print(df["Type"].value_counts())

    print("\n" + "=" * 60)
    print("OVERALL FAILURE RATE")
    print("=" * 60)
    rate = df["Machine failure"].mean()
    print(f"{rate:.4%} ({df['Machine failure'].sum()} / {len(df)} rows)")

    print("\n" + "=" * 60)
    print("FAILURE MODE BREAKDOWN (counts, can overlap)")
    print("=" * 60)
    print(df[FAILURE_MODES].sum())

    print("\n" + "=" * 60)
    print("NUMERIC FEATURE DISTRIBUTIONS")
    print("=" * 60)
    print(df[NUMERIC_FEATURES].describe().T)

    print("\n" + "=" * 60)
    print("FEATURE MEANS: FAILURE vs NO FAILURE")
    print("=" * 60)
    print(df.groupby("Machine failure")[NUMERIC_FEATURES].mean().T)


if __name__ == "__main__":
    main()
