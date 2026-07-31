# Predictive Maintenance / Anomaly Detection — AI4I 2020

Applied AI project using the [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(synthetic industrial machine sensor data: air temp, process temp, rotational speed, torque, tool wear, and failure labels).

Background: ELV/fire alarm & BMS field engineering, learning applied ML/AI.

## Project structure

```
.
├── data/           # cached dataset CSV (gitignored, regenerate with src/fetch_data.py)
├── notebooks/      # exploratory notebooks
├── src/            # scripts
│   ├── fetch_data.py   # pulls dataset via ucimlrepo, caches to data/ai4i2020.csv
│   └── eda.py          # first exploratory pass: shape, failure rate, distributions
├── reports/        # generated figures/reports
├── requirements.txt
└── venv/           # local virtual environment (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# fetch the dataset (writes data/ai4i2020.csv)
python src/fetch_data.py

# run the first EDA pass
python src/eda.py
```

## Dataset notes

- 10,000 rows, 12 columns, no missing values.
- Machine `Type`: L (low), M (medium), H (high) quality variants.
- Binary target `Machine failure`, plus 5 failure-mode flags: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- Overall failure rate: ~3.4% (imbalanced — plan for stratified splits / class weighting).

## Roadmap

- [x] Environment + data pipeline + first EDA
- [ ] Feature engineering & train/test split (stratified on `Machine failure`)
- [ ] Baseline models (logistic regression, random forest) via scikit-learn
- [ ] PyTorch model using MPS backend (Apple Silicon GPU) for comparison
- [ ] Streamlit dashboard for interactive exploration / model demo
