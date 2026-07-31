"""Threshold analysis for the Random Forest baseline (engineered features).

Sweeps the classification threshold from 0.05 to 0.5, plots precision and
recall against threshold, prints the confusion matrix at a few candidate
thresholds, and (via `expected_cost_analysis`) turns those confusion matrices
into an expected-cost curve so the threshold choice can be justified against
an explicit cost assumption instead of eyeballing the precision/recall plot.

This script does not pick a threshold for you -- `expected_cost_analysis`
shows where the cost curve is minimized *given* the cost inputs you provide,
and how sensitive that answer is to those inputs. The final call on which
cost assumption to trust is a business/maintenance-team decision.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from features import build_xy, load_raw

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

RANDOM_STATE = 42
CANDIDATE_THRESHOLDS = [0.3, 0.2, 0.1]

# --------------------------------------------------------------------------
# IMPORTANT: these are illustrative planning assumptions for exercising the
# cost-analysis machinery below, NOT audited or quoted figures. Before this
# threshold choice is used operationally, replace them with real numbers
# from maintenance/finance (technician callout cost for a false alarm;
# unplanned-downtime + repair + safety-risk cost for a missed failure).
# --------------------------------------------------------------------------
DEFAULT_COST_PER_MISSED_FAILURE_SAR = 5000.0
DEFAULT_COST_PER_FALSE_ALARM_SAR = 250.0


def compute_threshold_sweep(y_test, y_proba, thresholds):
    """Return per-threshold precision, recall, false negatives, and false positives."""
    precisions, recalls, false_negatives, false_positives = [], [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        false_negatives.append(fn)
        false_positives.append(fp)
    return (
        np.array(precisions),
        np.array(recalls),
        np.array(false_negatives),
        np.array(false_positives),
    )


def plot_precision_recall(thresholds, precisions, recalls, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", marker="o", markersize=3)
    ax.plot(thresholds, recalls, label="Recall", marker="o", markersize=3)
    for t in CANDIDATE_THRESHOLDS:
        ax.axvline(t, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("RandomForest (engineered features): Precision & Recall vs. Threshold")
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def print_confusion_matrices(y_test, y_proba):
    print("\ndefault threshold = 0.5 (for reference)")
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"  TP={tp:>3} FN={fn:>3} FP={fp:>3} TN={tn:>4}  "
          f"precision={precision_score(y_test, y_pred):.3f} recall={recall_score(y_test, y_pred):.3f}")

    for t in CANDIDATE_THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred)
        print(f"\nthreshold = {t}")
        print(f"  TP={tp:>3} FN={fn:>3} FP={fp:>3} TN={tn:>4}  precision={p:.3f} recall={r:.3f}")


def expected_cost_analysis(
    thresholds,
    false_negatives,
    false_positives,
    cost_per_missed_failure=DEFAULT_COST_PER_MISSED_FAILURE_SAR,
    cost_per_false_alarm=DEFAULT_COST_PER_FALSE_ALARM_SAR,
    ratio_range=(2, 20),
    out_dir="reports",
):
    """Turn the threshold sweep into an expected-cost curve and report the
    cost-minimizing threshold, plus its sensitivity to the cost assumption.

    Parameters
    ----------
    thresholds, false_negatives, false_positives : arrays from
        `compute_threshold_sweep`, aligned by index (same threshold grid).
    cost_per_missed_failure, cost_per_false_alarm : SAR cost assumptions.
        THESE ARE ILLUSTRATIVE PLANNING FIGURES, NOT AUDITED COSTS -- see the
        module-level comment above. Swap in real numbers before using this
        analysis to justify an operational decision.
    ratio_range : (min, max) missed-failure-to-false-alarm cost ratio to
        sweep, holding cost_per_false_alarm fixed and scaling
        cost_per_missed_failure = ratio * cost_per_false_alarm. Shows how
        much the "optimal" threshold moves as the cost assumption changes,
        rather than presenting one fixed answer as if it were certain.

    Returns
    -------
    dict with the default-cost optimal threshold, its expected cost, and the
    ratio -> optimal-threshold table (also saved as a plot).
    """
    thresholds = np.asarray(thresholds)
    false_negatives = np.asarray(false_negatives)
    false_positives = np.asarray(false_positives)

    # --- 1. Expected cost at the default cost assumption ---
    expected_cost = false_negatives * cost_per_missed_failure + false_positives * cost_per_false_alarm
    best_idx = int(np.argmin(expected_cost))
    best_threshold = thresholds[best_idx]
    best_cost = expected_cost[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, expected_cost, marker="o", markersize=3, color="tab:red")
    ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1)
    ax.scatter([best_threshold], [best_cost], color="black", zorder=5)
    ax.annotate(
        f"min cost\nthreshold={best_threshold:.2f}\ncost={best_cost:,.0f} SAR",
        xy=(best_threshold, best_cost),
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Expected cost (SAR)")
    ax.set_title(
        f"Expected cost vs. threshold\n"
        f"(illustrative: missed failure={cost_per_missed_failure:.0f} SAR, "
        f"false alarm={cost_per_false_alarm:.0f} SAR)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    cost_plot_path = f"{out_dir}/rf_expected_cost_vs_threshold.png"
    fig.savefig(cost_plot_path, dpi=150)
    print(f"Saved plot to {cost_plot_path}")

    # --- 2. Sensitivity: how does the optimal threshold move as the cost
    # ratio (missed failure / false alarm) changes, holding false-alarm cost
    # fixed at cost_per_false_alarm and scaling the missed-failure cost? ---
    ratios = np.arange(ratio_range[0], ratio_range[1] + 0.5, 0.5)
    optimal_thresholds = []
    for ratio in ratios:
        cost = false_negatives * (ratio * cost_per_false_alarm) + false_positives * cost_per_false_alarm
        optimal_thresholds.append(thresholds[int(np.argmin(cost))])
    optimal_thresholds = np.array(optimal_thresholds)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.step(ratios, optimal_thresholds, where="post", marker="o", markersize=3, color="tab:purple")
    ax2.set_xlabel("Cost ratio (missed failure / false alarm)")
    ax2.set_ylabel("Optimal threshold")
    ax2.set_title("Sensitivity: optimal threshold vs. cost ratio assumption")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    sensitivity_plot_path = f"{out_dir}/rf_optimal_threshold_vs_cost_ratio.png"
    fig2.savefig(sensitivity_plot_path, dpi=150)
    print(f"Saved plot to {sensitivity_plot_path}")

    # --- 3. Find the ratio(s) at which the optimal threshold flips ---
    flips = []
    for i in range(1, len(ratios)):
        if optimal_thresholds[i] != optimal_thresholds[i - 1]:
            flips.append((ratios[i], optimal_thresholds[i - 1], optimal_thresholds[i]))

    # --- 4. Print summary ---
    default_ratio = cost_per_missed_failure / cost_per_false_alarm
    print("\n" + "=" * 60)
    print("EXPECTED COST SUMMARY (illustrative cost assumptions)")
    print("=" * 60)
    print(f"cost_per_missed_failure = {cost_per_missed_failure:,.0f} SAR")
    print(f"cost_per_false_alarm    = {cost_per_false_alarm:,.0f} SAR")
    print(f"implied cost ratio      = {default_ratio:.1f}x")
    print(f"\n-> optimal threshold at default costs: {best_threshold:.2f} "
          f"(expected cost = {best_cost:,.0f} SAR)")

    print(f"\nsensitivity: optimal threshold across cost ratios "
          f"{ratio_range[0]}x - {ratio_range[1]}x")
    print(f"{'ratio':>6}  {'optimal_threshold':>18}")
    for r, opt_t in zip(ratios, optimal_thresholds):
        marker = " <-- default" if np.isclose(r, default_ratio, atol=0.25) else ""
        print(f"{r:>6.1f}  {opt_t:>18.2f}{marker}")

    if flips:
        print("\nthreshold flips as the cost ratio increases:")
        for ratio, before, after in flips:
            print(f"  at ratio ~{ratio:.1f}x: optimal threshold moves from {before:.2f} to {after:.2f}")
    else:
        print("\nno threshold flips detected across the swept ratio range "
              f"({ratio_range[0]}x-{ratio_range[1]}x) -- the optimal threshold "
              "was stable throughout.")

    return {
        "default_optimal_threshold": best_threshold,
        "default_optimal_cost": best_cost,
        "ratios": ratios,
        "optimal_thresholds": optimal_thresholds,
        "flips": flips,
    }


def main() -> None:
    df = load_raw()
    x, y = build_xy(df, engineered=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
    )
    rf.fit(x_train, y_train)
    y_proba = rf.predict_proba(x_test)[:, 1]

    thresholds = np.arange(0.05, 0.505, 0.01)
    precisions, recalls, false_negatives, false_positives = compute_threshold_sweep(
        y_test, y_proba, thresholds
    )

    plot_precision_recall(
        thresholds, precisions, recalls, "reports/rf_precision_recall_vs_threshold.png"
    )
    print_confusion_matrices(y_test, y_proba)

    expected_cost_analysis(thresholds, false_negatives, false_positives)


if __name__ == "__main__":
    main()
