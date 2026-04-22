"""
REAL-WORLD ACCURACY MEASUREMENT
=================================
Cross-validation and backtest give you estimates of how the model
*should* perform. This script tells you how it actually *did* perform
on real predictions already served in production.

THE LOOP THIS CLOSES
  1. serving/api.py serves a prediction for user_id=42 on April 22
  2. inference/prediction_logger.py records it in ml_prediction_log
  3. Time passes. The user either does something or they don't.
  4. This script queries ml_prediction_log joined with study_streaks to
     check what actually happened in the 7 days after each prediction.
  5. The difference between what we predicted and what happened is the
     model's production accuracy.

WHY THIS MATTERS
  CV and backtest measure the model on data that LOOKS LIKE production
  but isn't. Production accuracy is the only number that matters to the
  business. A model that scored 0.97 F1 in CV but only 0.72 F1 in
  production tells you:
    - Something about your features is shifting (drift → drift_check.py)
    - OR users behave differently in the wild than in your synthetic data
    - OR the label definition doesn't match what "dropout" actually
      looks like operationally

WHAT THIS SCRIPT DOES
  1. Find every prediction in ml_prediction_log that's older than
     --window-days (so the outcome window has fully elapsed and we can
     check the truth).
  2. For each one, join with study_streaks to see if the user was
     active in (logged_at, logged_at + window_days].
  3. Compute accuracy, precision, recall, F1, ROC AUC against the
     predicted probabilities and risk levels.
  4. Break metrics down by model_version (lets you compare v3 vs v4
     on real production data, not just CV).
  5. Log everything to MLflow under purpose=real_world_accuracy.

Usage:
    python -m evaluation.measure_accuracy
    python -m evaluation.measure_accuracy --window-days 7
    python -m evaluation.measure_accuracy --threshold 0.5
    python -m evaluation.measure_accuracy --no-mlflow
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from config import DATABASE_URL, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


# SQL that does the whole join in one round trip. Using `::date` truncation
# because study_streaks.activity_date is a DATE, not TIMESTAMPTZ.
_QUERY = text("""
    WITH evaluatable AS (
        SELECT
            mpl.user_id,
            mpl.logged_at,
            mpl.dropout_probability,
            mpl.risk_level,
            mpl.model_version
        FROM ml_prediction_log mpl
        WHERE mpl.logged_at < NOW() - (:window_days || ' days')::interval
    )
    SELECT
        e.user_id,
        e.logged_at,
        e.dropout_probability,
        e.risk_level,
        e.model_version,
        -- 1 if the user had NO activity in the forward window = actual dropout
        CASE WHEN EXISTS (
            SELECT 1
            FROM study_streaks ss
            WHERE ss.user_id = e.user_id
              AND ss.activity_date >= e.logged_at::date
              AND ss.activity_date <  e.logged_at::date + (:window_days || ' days')::interval
        ) THEN 0 ELSE 1 END AS actual_dropout
    FROM evaluatable e
""")


def fetch_outcomes(window_days: int) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        return pd.read_sql(_QUERY, conn, params={"window_days": window_days})


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["actual_dropout"].values.astype(int)
    y_prob = df["dropout_probability"].values.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "n_predictions": int(len(df)),
        "actual_dropout_rate": float(y_true.mean()),
        "predicted_dropout_rate": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC AUC only defined when both classes are present in ground truth
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def print_report(metrics: dict, df: pd.DataFrame, threshold: float) -> None:
    print("\n" + "=" * 72)
    print("REAL-WORLD ACCURACY REPORT")
    print("=" * 72)
    print(f"  Evaluated predictions:      {metrics['n_predictions']:,}")
    print(f"  Threshold for 'predicted dropout':  probability >= {threshold}")
    print(f"  Actual dropout rate:        {metrics['actual_dropout_rate']:.1%}")
    print(f"  Predicted dropout rate:     {metrics['predicted_dropout_rate']:.1%}")
    print()
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}   (of predicted dropouts, fraction that really were)")
    print(f"  Recall:     {metrics['recall']:.4f}   (of actual dropouts, fraction we caught)")
    print(f"  F1:         {metrics['f1']:.4f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"  ROC AUC:    {metrics['roc_auc']:.4f}")

    # Confusion matrix
    y_true = df["actual_dropout"].values.astype(int)
    y_pred = (df["dropout_probability"].values >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print()
    print("  Confusion matrix")
    print(f"                    predicted active   predicted dropout")
    print(f"    actual active   {cm[0, 0]:>8d}            {cm[0, 1]:>8d}")
    print(f"    actual dropout  {cm[1, 0]:>8d}            {cm[1, 1]:>8d}")

    # Per-version breakdown (useful once v5, v6 exist)
    if df["model_version"].nunique() > 1:
        print("\n  Per model-version breakdown:")
        print(f"    {'version':<28s}{'n':>8s}{'precision':>12s}{'recall':>10s}{'f1':>10s}")
        for version, grp in df.groupby("model_version"):
            m = compute_metrics(grp, threshold)
            print(f"    {version:<28s}{m['n_predictions']:>8d}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}")

    # Per-risk-level breakdown (tells you how well each bucket maps to outcome)
    print("\n  Per risk-level breakdown:")
    print(f"    {'risk_level':<14s}{'n':>8s}{'actual_dropout_rate':>24s}")
    for level, grp in df.groupby("risk_level"):
        rate = float(grp["actual_dropout"].mean())
        print(f"    {level:<14s}{len(grp):>8d}{rate:>24.1%}")


def log_to_mlflow(metrics: dict, df: pd.DataFrame, threshold: float, window_days: int) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"real-world-accuracy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tag("purpose", "real_world_accuracy")
        mlflow.log_params({
            "threshold": threshold,
            "window_days": window_days,
            "n_predictions": metrics["n_predictions"],
            "distinct_model_versions": int(df["model_version"].nunique()),
        })
        loggable = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)}
        mlflow.log_metrics(loggable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the model's accuracy on real production predictions.")
    parser.add_argument("--window-days", type=int, default=7,
                        help="Size of the forward window used when the prediction was made (default: 7)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold above which we say the model predicted dropout (default: 0.5)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging (report-only mode)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"Fetching predictions logged before  NOW - {args.window_days} days  ...")
    print("=" * 72)
    df = fetch_outcomes(args.window_days)

    if df.empty:
        # Totally expected on day 1 — the log table exists but nothing in it
        # is old enough to have a ground-truth answer yet.
        print(f"\nNo predictions old enough yet. Come back in {args.window_days} days.")
        print(f"(The ml_prediction_log table needs entries with logged_at older than")
        print(f" the forward window before we can join with actual activity.)")
        return

    metrics = compute_metrics(df, args.threshold)
    print_report(metrics, df, args.threshold)

    if not args.no_mlflow:
        print("\n  Logging to MLflow under purpose=real_world_accuracy...")
        log_to_mlflow(metrics, df, args.threshold, args.window_days)


if __name__ == "__main__":
    main()
