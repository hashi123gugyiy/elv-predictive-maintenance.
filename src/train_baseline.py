"""Baseline models for AI4I 2020 failure prediction.

Compares Logistic Regression and Random Forest, each on the RAW and
ENGINEERED feature sets, using a stratified train/test split and
class_weight='balanced' to handle the ~3.4% failure rate.

Metrics: precision, recall, F1, and ROC-AUC on the failure class.
Accuracy is intentionally not the headline metric -- a model that always
predicts "no failure" scores ~96.6% accuracy while catching zero failures.
"""
import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_xy, load_raw

RANDOM_STATE = 42

# Apple's Accelerate BLAS backend (used by NumPy on Apple Silicon) emits
# spurious "divide by zero"/"overflow" RuntimeWarnings on some matmul shapes
# during LogisticRegression fit/predict even when every value involved is
# finite -- verified by inspecting coef_ and the transformed inputs directly.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)


@dataclass
class Result:
    feature_set: str
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_feature_set(df: pd.DataFrame, engineered: bool) -> list[Result]:
    feature_set = "engineered" if engineered else "raw"
    x, y = build_xy(df, engineered=engineered)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        results.append(Result(feature_set=feature_set, model_name=name, **metrics))

    return results


def main() -> None:
    df = load_raw()

    all_results = []
    for engineered in (False, True):
        all_results.extend(run_feature_set(df, engineered=engineered))

    print(f"{'feature_set':<12} {'model':<20} {'precision':>9} {'recall':>7} "
          f"{'f1':>6} {'roc_auc':>8} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>5}")
    for r in all_results:
        print(
            f"{r.feature_set:<12} {r.model_name:<20} {r.precision:>9.3f} {r.recall:>7.3f} "
            f"{r.f1:>6.3f} {r.roc_auc:>8.3f} {r.tp:>4} {r.fn:>4} {r.fp:>4} {r.tn:>5}"
        )


if __name__ == "__main__":
    main()
