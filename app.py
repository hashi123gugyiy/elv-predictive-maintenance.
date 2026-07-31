"""Streamlit app for the AI4I 2020 predictive maintenance project.

Two sections:
1. Failure Risk Predictor -- sliders for the 5 raw sensor readings, run
   through the trained Random Forest (engineered features), showing failure
   probability and a risk label at the 0.3 threshold settled on in
   src/threshold_analysis.py.
2. Expected-Cost Threshold Explorer -- an interactive version of
   expected_cost_analysis(): drag the missed-failure/false-alarm cost ratio
   and watch the expected-cost curve and optimal threshold update live.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from features import build_xy, load_raw  # noqa: E402
from threshold_analysis import compute_threshold_sweep  # noqa: E402

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.3  # threshold settled on for the RF (engineered) model
DEFAULT_COST_PER_FALSE_ALARM_SAR = 250.0
DEFAULT_COST_RATIO = 20.0  # matches the illustrative 5000/250 SAR default

st.set_page_config(page_title="AI4I Predictive Maintenance", layout="wide")


@st.cache_resource
def train_model():
    df = load_raw()
    x, y = build_xy(df, engineered=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_test)[:, 1]
    return model, x_train.columns.tolist(), y_test.to_numpy(), y_proba


model, feature_columns, y_test, y_proba = train_model()

thresholds = np.arange(0.05, 0.505, 0.01)
_, _, false_negatives, false_positives = compute_threshold_sweep(y_test, y_proba, thresholds)

st.title("AI4I 2020 Predictive Maintenance")

tab_predict, tab_cost = st.tabs(
    ["Failure Risk Predictor", "Expected-Cost Threshold Explorer"]
)

with tab_predict:
    st.header("Failure Risk Predictor")
    st.caption(
        "Random Forest, engineered features (power, temp_diff), "
        f"flagged HIGH RISK at probability >= {DECISION_THRESHOLD:.0%}."
    )

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
        rot_speed = st.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
    with col2:
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
        tool_wear = st.slider("Tool wear (min)", 0, 255, 100, 1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=0)

    power = torque * rot_speed
    temp_diff = process_temp - air_temp

    input_row = pd.DataFrame(
        [
            {
                "Air temperature": air_temp,
                "Process temperature": process_temp,
                "Rotational speed": rot_speed,
                "Torque": torque,
                "Tool wear": tool_wear,
                "power": power,
                "temp_diff": temp_diff,
                "Type_L": 1 if machine_type == "L" else 0,
                "Type_M": 1 if machine_type == "M" else 0,
            }
        ]
    ).reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(input_row)[0, 1]
    is_high_risk = proba >= DECISION_THRESHOLD

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Predicted failure probability", f"{proba:.1%}")
    with col_b:
        if is_high_risk:
            st.error(f"HIGH RISK — at/above the {DECISION_THRESHOLD:.0%} threshold. Recommend inspection.")
        else:
            st.success(f"LOW RISK — below the {DECISION_THRESHOLD:.0%} threshold.")
    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("Derived features used by the model"):
        st.write(f"power (torque x rotational speed) = {power:,.0f}")
        st.write(f"temp_diff (process temp - air temp) = {temp_diff:.1f} K")

with tab_cost:
    st.header("Expected-Cost Threshold Explorer")
    st.caption(
        "Illustrative SAR costs, not audited figures -- see README. "
        "Drag the cost ratio to see the cost-minimizing threshold shift."
    )

    col1, col2 = st.columns(2)
    with col1:
        cost_ratio = st.slider(
            "Cost ratio (missed failure / false alarm)",
            2.0, 20.0, DEFAULT_COST_RATIO, 0.5,
        )
    with col2:
        cost_per_false_alarm = st.number_input(
            "Cost per false alarm (SAR)",
            value=DEFAULT_COST_PER_FALSE_ALARM_SAR,
            step=50.0,
        )

    cost_per_missed_failure = cost_ratio * cost_per_false_alarm
    expected_cost = (
        false_negatives * cost_per_missed_failure + false_positives * cost_per_false_alarm
    )
    best_idx = int(np.argmin(expected_cost))
    best_threshold = thresholds[best_idx]
    best_cost = expected_cost[best_idx]

    st.metric(
        "Optimal threshold at this cost ratio",
        f"{best_threshold:.2f}",
        help=f"Expected cost = {best_cost:,.0f} SAR (missed failure = {cost_per_missed_failure:,.0f} SAR)",
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(thresholds, expected_cost, color="tab:red", marker="o", markersize=3)
    ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1)
    ax.scatter([best_threshold], [best_cost], color="black", zorder=5)
    ax.annotate(
        f"min cost\nthreshold={best_threshold:.2f}",
        xy=(best_threshold, best_cost),
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Expected cost (SAR)")
    ax.set_title(
        f"Expected cost vs. threshold "
        f"(missed failure={cost_per_missed_failure:,.0f} SAR, false alarm={cost_per_false_alarm:,.0f} SAR)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    if best_idx in (0, len(thresholds) - 1):
        st.warning(
            "The optimal threshold sits at the edge of the swept range (0.05-0.5) — "
            "the true optimum at this cost ratio may lie outside what's been tested."
        )
