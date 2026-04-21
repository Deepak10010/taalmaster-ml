"""
DRIFT DETECTION
=================
Compare today's feature distributions to the distribution the model was
trained on. If they differ enough, the model may be making predictions in
a world it no longer understands — a signal to retrain.

THIS IS THE "DRIFT" HALF OF THE RETRAIN TRIGGER
  The architecture diagram shows two retrain triggers: "Scheduled" and
  "Drift". Scheduled retraining is handled by Windows Task Scheduler
  invoking pipeline.py daily at 10am. This script is the drift half —
  run it on whatever cadence you like (hourly, daily, pre-deploy), and
  act on the exit code if `--fail-on-drift` is set.

HOW DRIFT IS MEASURED
  Per feature:
    - Continuous features → Kolmogorov-Smirnov 2-sample test
    - Binary (0/1) features → chi-square test on a 2×2 contingency table
  A feature is flagged as "drifted" when its p-value < --threshold
  (default 0.01, meaning <1% chance the distributions are the same).

  Why KS + chi-square instead of a single test?
    KS works on continuous distributions. For a binary feature like
    `is_trial_expired`, KS still runs but gives degenerate results.
    Chi-square is the principled test for binary.

WHAT DRIFT MEANS IN PRACTICE
  Some drift is expected (it's a living product). But a sudden shift in
  e.g. `days_since_last_activity` distribution is a red flag — it may
  mean a feature rolled out, a campaign drove re-engagement, or a bug
  is skewing activity data.

  The output shows:
    - How many features drifted (and at what p-value)
    - Per-feature: reference mean vs. current mean, and % delta

Usage:
    python -m evaluation.drift_check
    python -m evaluation.drift_check --threshold 0.05
    python -m evaluation.drift_check --fail-on-drift     # exit 1 on drift
    python -m evaluation.drift_check --no-mlflow         # skip MLflow logging

Cron chaining example (retrain only if drift detected):
    python -m evaluation.drift_check --fail-on-drift && echo "no drift, skipping" \\
        || python pipeline.py
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from scipy import stats

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from ingestion.data_extraction import extract_all
from transformation.feature_engineering import build_feature_matrix, get_feature_columns


FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"


def load_reference_features():
    """
    The reference distribution is the most recent training snapshot.
    Good enough when the pipeline runs daily (the reference is never stale
    by more than a day). For a stricter definition, pull the exact features
    that trained the current Production model via MLflow run_id.
    """
    candidates = sorted(FEATURES_DIR.glob("features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No training snapshot found in {FEATURES_DIR}. "
            f"Run `python pipeline.py` first to produce one."
        )
    path = candidates[-1]
    df = pd.read_parquet(path)
    return df, path


def compute_current_features():
    """Fresh extract + feature build with ref_date = now, no label."""
    now = datetime.utcnow()
    raw = extract_all()
    features = build_feature_matrix(raw, ref_date=now, compute_label=False)
    return features


def _is_binary(values: np.ndarray) -> bool:
    unique = set(np.unique(values).tolist())
    return unique.issubset({0, 1, 0.0, 1.0, True, False})


def test_drift(ref: np.ndarray, cur: np.ndarray) -> tuple[str, float, float]:
    """Pick the right test for the feature type and return (test, stat, p_value)."""
    if _is_binary(ref):
        ref_pos = int((ref == 1).sum())
        ref_neg = int((ref == 0).sum())
        cur_pos = int((cur == 1).sum())
        cur_neg = int((cur == 0).sum())
        contingency = [[ref_pos, ref_neg], [cur_pos, cur_neg]]
        # chi-square breaks if any margin is zero (constant feature); fall back
        # to "no drift detected" rather than raising.
        row_ok = all(sum(row) > 0 for row in contingency)
        col_ok = (ref_pos + cur_pos) > 0 and (ref_neg + cur_neg) > 0
        if row_ok and col_ok:
            try:
                chi2, p, *_ = stats.chi2_contingency(contingency)
                return "chi2", float(chi2), float(p)
            except ValueError:
                return "chi2", 0.0, 1.0
        return "chi2", 0.0, 1.0

    stat, p = stats.ks_2samp(ref, cur)
    return "ks", float(stat), float(p)


def analyze_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  feature_cols: list, threshold: float) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        ref_vals = reference[feat].values
        cur_vals = current[feat].values
        test, stat, p = test_drift(ref_vals, cur_vals)

        ref_mean = float(np.mean(ref_vals))
        cur_mean = float(np.mean(cur_vals))
        delta_pct = (cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

        rows.append({
            "feature": feat,
            "test": test,
            "statistic": stat,
            "p_value": p,
            "drifted": p < threshold,
            "ref_mean": ref_mean,
            "cur_mean": cur_mean,
            "mean_delta_pct": delta_pct,
        })
    return pd.DataFrame(rows)


def print_report(drift_df: pd.DataFrame, threshold: float, ref_path: Path) -> int:
    drifted = drift_df[drift_df["drifted"]].sort_values("p_value")
    n_drifted = len(drifted)
    n_total = len(drift_df)

    print("\n" + "=" * 72)
    print(f"DRIFT REPORT   {n_drifted} / {n_total} features drifted   "
          f"(threshold p < {threshold})")
    print(f"Reference snapshot: {ref_path.name}")
    print("=" * 72)

    if n_drifted == 0:
        print("  [OK] No significant drift detected.")
    else:
        print(f"  {'feature':<32s}{'test':>8s}{'p_value':>14s}{'ref_mean':>14s}{'cur_mean':>14s}{'delta':>10s}")
        print(f"  {'-' * 92}")
        for _, row in drifted.iterrows():
            print(f"  {row['feature']:<32s}{row['test']:>8s}"
                  f"{row['p_value']:>14.2e}{row['ref_mean']:>14.3g}"
                  f"{row['cur_mean']:>14.3g}{row['mean_delta_pct']:>+9.1f}%")

    return n_drifted


def log_to_mlflow(drift_df: pd.DataFrame, threshold: float,
                  ref_path: Path, n_ref: int, n_cur: int) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift-check-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tag("purpose", "drift_check")
        mlflow.log_params({
            "threshold": threshold,
            "reference_snapshot": ref_path.name,
            "n_features": len(drift_df),
            "n_ref_rows": n_ref,
            "n_cur_rows": n_cur,
        })
        n_drifted = int(drift_df["drifted"].sum())
        mlflow.log_metrics({
            "n_drifted": n_drifted,
            "drift_rate": n_drifted / max(len(drift_df), 1),
            "min_p_value": float(drift_df["p_value"].min()),
            "max_abs_delta_pct": float(drift_df["mean_delta_pct"].abs().max()),
        })
        mlflow.log_dict(
            {"rows": drift_df.to_dict(orient="records")},
            "drift_report.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-distribution drift check")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="p-value cutoff (default 0.01)")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit code 1 if any feature drifted (useful in cron chains)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    args = parser.parse_args()

    print("=" * 72)
    print("Feature drift check")
    print("=" * 72)

    print("  Loading reference features...")
    reference, ref_path = load_reference_features()
    print(f"    {ref_path.name}  ({len(reference):,} rows)")

    print("  Building current features (extract + feature engineering)...")
    current = compute_current_features()
    print(f"    current snapshot  ({len(current):,} rows)")

    feature_cols = get_feature_columns()
    print(f"  Analyzing {len(feature_cols)} features (p-threshold = {args.threshold})...")
    drift_df = analyze_drift(reference, current, feature_cols, args.threshold)

    n_drifted = print_report(drift_df, args.threshold, ref_path)

    if not args.no_mlflow:
        print("\n  Logging drift report to MLflow...")
        log_to_mlflow(drift_df, args.threshold, ref_path, len(reference), len(current))

    if args.fail_on_drift and n_drifted > 0:
        print(f"\nExiting with code 1 (--fail-on-drift set; {n_drifted} features drifted)")
        sys.exit(1)


if __name__ == "__main__":
    main()
