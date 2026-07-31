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
│   ├── fetch_data.py         # pulls dataset via ucimlrepo, caches to data/ai4i2020.csv
│   ├── eda.py                # first exploratory pass: shape, failure rate, distributions
│   ├── features.py           # raw vs. engineered (power, temp_diff) feature sets
│   ├── train_baseline.py     # Logistic Regression / Random Forest baselines
│   └── threshold_analysis.py # RF threshold sweep + expected-cost analysis
├── reports/        # generated figures (precision/recall & expected-cost plots)
├── requirements.txt
└── venv/           # local virtual environment (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Streamlit app

```bash
streamlit run app.py
```

Two tabs:
- **Failure Risk Predictor** — sliders for the 5 raw sensor readings (+ machine
  Type), run through the trained RF (engineered features), showing failure
  probability and a HIGH/LOW RISK label at the 0.3 threshold.
- **Expected-Cost Threshold Explorer** — drag the missed-failure/false-alarm
  cost ratio and watch the expected-cost curve and cost-minimizing threshold
  update live (same math as `expected_cost_analysis()`).

## Usage

```bash
# fetch the dataset (writes data/ai4i2020.csv)
python src/fetch_data.py

# run the first EDA pass
python src/eda.py

# train baseline models (raw vs. engineered features, LR vs. RF)
python src/train_baseline.py

# RF threshold sweep + expected-cost analysis (writes plots to reports/)
python src/threshold_analysis.py
```

## Dataset notes

- 10,000 rows, 12 columns, no missing values.
- Machine `Type`: L (low), M (medium), H (high) quality variants.
- Binary target `Machine failure`, plus 5 failure-mode flags: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- Overall failure rate: ~3.4% (imbalanced — plan for stratified splits / class weighting).

## Modeling notes

- `TWF`/`HDF`/`PWF`/`OSF`/`RNF` are components of `Machine failure`, not
  inputs — they're dropped before training to avoid label leakage.
- Splits are stratified on `Machine failure` (3.39% positive rate): with
  only 339 failure rows total, a plain random split can easily land very
  different failure counts in train vs. test purely by chance, making
  recall/precision estimates unstable run to run.
- `features.py` compares a **raw** feature set (5 sensor columns + one-hot
  `Type`) against an **engineered** one that adds `power` (torque ×
  rotational speed) and `temp_diff` (process temp − air temp) — these mirror
  the actual threshold rules AI4I 2020 uses to define the PWF/HDF failure
  modes, not a speculative guess. Engineered features improved every model
  on every metric in `train_baseline.py`.
- Both baselines use `class_weight='balanced'` rather than SMOTE: it
  reweights the loss for the ~3.4% minority class with no synthetic data
  and no leakage risk from oversampling before/across CV folds.
- Accuracy is not used as a headline metric — a model that always predicts
  "no failure" scores ~96.6% accuracy while catching zero real failures.

### Threshold selection: an expected-cost framework, not a single answer

Rather than picking one classification threshold for the Random Forest
model, `threshold_analysis.py::expected_cost_analysis()` turns the
precision/recall sweep into an **expected cost curve**:

```
expected_cost = (false_negatives * cost_per_missed_failure) + (false_positives * cost_per_false_alarm)
```

A false negative here means a machine fails without a maintenance trigger
(unplanned downtime, possible collateral damage); a false positive sends a
technician to inspect a machine that turns out fine. The right threshold
depends entirely on the ratio between those two costs, which is a
maintenance/business input, not something a model can determine on its own.

Swept from a 2x to 20x missed-failure-to-false-alarm cost ratio, the
cost-minimizing threshold is **not stable** — it steps down across four
distinct bands:

| cost ratio | optimal threshold |
|---|---|
| ~2x – 4.5x | 0.31 |
| ~5x – 10.5x | 0.26 |
| ~11x – 14x | 0.19 |
| ~15x – 20x | 0.06 (brief transitional step at 0.10 around 14.5x) |

The transitions cluster tightly between **~11x and ~15x**, meaning that
band is where the deployment decision is most sensitive to the cost
assumption — a small change in the estimated cost ratio there flips the
recommended threshold multiple times.

**Illustrative costs used to build and demonstrate this framework:
5,000 SAR per missed failure, 250 SAR per false alarm (20x ratio) — these
are planning placeholders, not audited figures, and must be replaced with
real maintenance/finance numbers before this threshold is used
operationally.** At that illustrative 20x ratio the framework picks 0.06,
which sits at the edge of the swept range (0.05–0.5) — the cost curve was
still declining as it approached that edge, so the true optimum at a real
20x ratio may fall below 0.05 and the sweep range should be widened before
trusting that specific number.

See `reports/rf_expected_cost_vs_threshold.png` and
`reports/rf_optimal_threshold_vs_cost_ratio.png` for the underlying plots.

## Roadmap

- [x] Environment + data pipeline + first EDA
- [x] Feature engineering (raw vs. engineered) & stratified train/test split
- [x] Baseline models (logistic regression, random forest) via scikit-learn
- [x] Threshold selection via expected-cost framework (illustrative costs)
- [x] Streamlit app: risk predictor + interactive expected-cost explorer
- [ ] Replace illustrative SAR costs with real maintenance/finance figures
- [ ] PyTorch model using MPS backend (Apple Silicon GPU) for comparison
- [ ] Streamlit dashboard for interactive exploration / model demo
