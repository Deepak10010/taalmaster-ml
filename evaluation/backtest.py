"""
TEMPORAL BACKTEST
===================
Cross-validation reuses the same slice of time — every fold's features AND
label come from the same ref_date. A model that only works at THAT date can
still score well in CV.

A temporal backtest is harder: train on one date, test on a date the model
has never seen. This simulates what actually happens in production
(you train Monday, deploy, run against next week's data).

WHAT THIS SCRIPT DOES:
    1. Pull data from Neon once.
    2. Build the TRAIN matrix at ref_date = T - 21 days
           features: data before T-21d
           label:    activity in [T-21d, T-14d]
    3. Build the TEST matrix at ref_date = T - 14 days
           features: data before T-14d
           label:    activity in [T-14d, T-7d]
    4. Train XGBoost on the TRAIN matrix.
    5. Score the TEST matrix.  The test week was never touched during
       training, so this is a clean generalization check.

INTERPRETATION:
    CV F1 ≈ 0.97  and  holdout F1 ≈ 0.97  → model genuinely generalizes.
    CV F1 ≈ 0.97  and  holdout F1 ≈ 0.70  → your data (or features) are
                                             easier in-sample than out.
"""
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, DROPOUT_DAYS_THRESHOLD
from ingestion.data_extraction import extract_all
from transformation.feature_engineering import build_feature_matrix, get_feature_columns

warnings.filterwarnings("ignore")


def build_xgb(pos_weight: float) -> XGBClassifier:
    """Same hyperparameters as train.py so the backtest is apples-to-apples."""
    return XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42,
    )


def metrics_from(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal holdout backtest")
    parser.add_argument("--train-offset-days", type=int, default=21,
                        help="How far back (from now) is the TRAIN ref_date")
    parser.add_argument("--test-offset-days", type=int, default=14,
                        help="How far back (from now) is the TEST ref_date")
    parser.add_argument("--window-days", type=int, default=DROPOUT_DAYS_THRESHOLD,
                        help="Forward window for the label")
    args = parser.parse_args()

    now = datetime.utcnow()
    train_ref = now - timedelta(days=args.train_offset_days)
    test_ref = now - timedelta(days=args.test_offset_days)

    # Sanity check: training window must not overlap with the test window.
    train_label_end = train_ref + timedelta(days=args.window_days)
    if train_label_end > test_ref:
        raise ValueError(
            f"Training label window ends {train_label_end.date()} but test ref_date is "
            f"{test_ref.date()} — they overlap. Pick --train-offset-days >= "
            f"--test-offset-days + {args.window_days}."
        )

    print("=" * 68)
    print("TaalMaster Temporal Backtest")
    print("=" * 68)
    print(f"TRAIN ref_date : {train_ref.date()}   features < this date")
    print(f"TRAIN label    : [{train_ref.date()}, {train_label_end.date()})")
    print(f"TEST ref_date  : {test_ref.date()}    features < this date")
    test_label_end = test_ref + timedelta(days=args.window_days)
    print(f"TEST label     : [{test_ref.date()}, {test_label_end.date()})")
    print("=" * 68)

    print("\n[1/4] Extracting data from Neon...")
    raw = extract_all()
    print(f"      {len(raw['students']):,} students, {len(raw['streaks']):,} streak rows")

    print(f"\n[2/4] Building TRAIN matrix (ref_date = {train_ref.date()})...")
    train_df = build_feature_matrix(raw, train_ref, label_window_days=args.window_days)
    print(f"      shape = {train_df.shape[0]:,} x {train_df.shape[1]}   "
          f"dropout rate = {train_df['is_dropout'].mean():.1%}")

    print(f"\n[3/4] Building TEST matrix (ref_date = {test_ref.date()})...")
    test_df = build_feature_matrix(raw, test_ref, label_window_days=args.window_days)
    print(f"      shape = {test_df.shape[0]:,} x {test_df.shape[1]}   "
          f"dropout rate = {test_df['is_dropout'].mean():.1%}")

    feature_cols = get_feature_columns()
    X_train, y_train = train_df[feature_cols].values, train_df["is_dropout"].values
    X_test, y_test = test_df[feature_cols].values, test_df["is_dropout"].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    print("\n[4/4] Training on TRAIN, scoring on TEST, and running CV on TRAIN for reference...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"backtest-{now.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tag("purpose", "temporal_backtest")
        mlflow.log_params({
            "train_ref_date": train_ref.isoformat(),
            "test_ref_date": test_ref.isoformat(),
            "window_days": args.window_days,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "train_dropout_rate": float(y_train.mean()),
            "test_dropout_rate": float(y_test.mean()),
        })

        # ── 5-fold CV on the training matrix (for reference) ──
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = build_xgb(pos_weight)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            y_p = m.predict(X_train[va_idx])
            y_pr = m.predict_proba(X_train[va_idx])[:, 1]
            for k, v in metrics_from(y_train[va_idx], y_p, y_pr).items():
                cv[k].append(v)
        cv_summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in cv.items()}

        for metric, (mean, std) in cv_summary.items():
            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

        # ── Train on ALL of TRAIN, score TEST (the real holdout) ──
        model = build_xgb(pos_weight)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        holdout = metrics_from(y_test, y_test_pred, y_test_proba)

        for metric, val in holdout.items():
            mlflow.log_metric(f"holdout_{metric}", val)

        print("\n" + "=" * 68)
        print("RESULTS")
        print("=" * 68)
        header = f"{'metric':<12s}{'CV on TRAIN':>20s}{'HOLDOUT on TEST':>20s}{'delta':>14s}"
        print(header)
        print("-" * 68)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            cv_m, cv_s = cv_summary[metric]
            ho = holdout[metric]
            delta = ho - cv_m
            print(f"{metric:<12s}{cv_m:>14.4f} +/- {cv_s:>.3f}{ho:>20.4f}{delta:>+14.4f}")
        print("=" * 68)

        print(f"\nConfusion matrix on TEST (rows = actual, cols = predicted):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"              predicted Active   predicted Dropout")
        print(f"actual Active    {cm[0,0]:>8d}           {cm[0,1]:>8d}")
        print(f"actual Dropout   {cm[1,0]:>8d}           {cm[1,1]:>8d}")

        # Interpretation hint
        f1_cv = cv_summary["f1"][0]
        f1_ho = holdout["f1"]
        drop = f1_cv - f1_ho
        print(f"\nF1 drop from CV to holdout: {drop:+.4f}")
        if drop < 0.02:
            verdict = "Model generalizes cleanly to an unseen week. The strong CV score is real."
        elif drop < 0.08:
            verdict = "Modest generalization gap. Expected for real data; still trustworthy."
        else:
            verdict = ("Large generalization gap. The model is fitting patterns specific to "
                       "the training week. Investigate features or gather more data.")
        print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
