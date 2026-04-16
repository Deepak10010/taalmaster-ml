"""
MODEL COMPARISON
==================
Train Logistic Regression, Random Forest, and XGBoost on the SAME features
with the SAME 5-fold CV, so the only thing that changes is the algorithm.
Every model is logged as its own MLflow run — open the UI and tick all
three to see them side-by-side.

WHY COMPARE AT ALL?
  - XGBoost might be overkill. If logistic regression scores 0.95 and XGB
    scores 0.97, maybe the simpler model is the right choice (faster,
    easier to explain, fewer moving parts).
  - If LR scores 0.60 and XGB scores 0.97, you know non-linear patterns
    and feature interactions are doing real work — not just "the signal
    is obvious."

THE THREE MODELS:
  LR  → linear baseline. If this already wins, your features are very
        clean and mostly linear in their effect on dropout.
  RF  → bagged trees. Averages many independent trees, each trained on a
        random subset. Different bias/variance profile than boosted trees.
  XGB → gradient-boosted trees. Trees are built SEQUENTIALLY, each one
        correcting the last one's mistakes. Usually the strongest on
        tabular data, but can overfit without care.

Usage:
    python compare_models.py
    python compare_models.py --features data/features/features_20260416_193734.parquet
"""
import argparse
import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from transformation.feature_engineering import get_feature_columns


FEATURES_DIR = Path(__file__).resolve().parent / "data" / "features"


def latest_features_path() -> Path:
    candidates = sorted(FEATURES_DIR.glob("features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No feature snapshot found in {FEATURES_DIR}. "
            f"Run `python pipeline.py` first."
        )
    return candidates[-1]


def evaluate(name: str, model_factory, X: np.ndarray, y: np.ndarray,
             feature_cols: list) -> dict:
    """Run one model through 5-fold CV and log to its own MLflow run."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    run_name = f"compare-{name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", name)
        mlflow.set_tag("purpose", "model_comparison")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_students", len(y))
        mlflow.log_param("n_dropout", int(y.sum()))

        for train_idx, val_idx in skf.split(X, y):
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            y_proba = model.predict_proba(X[val_idx])[:, 1]

            metrics["accuracy"].append(accuracy_score(y[val_idx], y_pred))
            metrics["precision"].append(precision_score(y[val_idx], y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y[val_idx], y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y[val_idx], y_pred, zero_division=0))
            metrics["roc_auc"].append(
                roc_auc_score(y[val_idx], y_proba) if len(np.unique(y[val_idx])) > 1 else 0.0
            )

        summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
        for metric, (mean, std) in summary.items():
            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare LR, RF, XGBoost on the same features")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to a features parquet file (default: most recent)")
    args = parser.parse_args()

    features_path = Path(args.features) if args.features else latest_features_path()
    print(f"Loading features from: {features_path}")
    features = pd.read_parquet(features_path)

    feature_cols = get_feature_columns()
    X = features[feature_cols].values
    y = features["is_dropout"].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    print(f"Students: {len(y):,}  Dropout: {n_pos:,}  Active: {n_neg:,}  (pos_weight={pos_weight:.2f})")

    # Model zoo. Each is wrapped in a factory so every fold gets a fresh instance.
    candidates = {
        # Linear baseline. Features must be scaled for LR to behave.
        "logistic_regression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )),
        ]),
        # Bagged trees. Independent trees, averaged.
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        # Boosted trees. Same hyperparameters as train.py so the comparison is fair.
        "xgboost": lambda: XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}
    for name, factory in candidates.items():
        print(f"\n>> Evaluating {name}...")
        results[name] = evaluate(name, factory, X, y, feature_cols)
        f1_mean, f1_std = results[name]["f1"]
        print(f"    CV F1 = {f1_mean:.4f} (+/- {f1_std:.4f})")

    # ── Side-by-side table ──
    print("\n" + "=" * 78)
    print("COMPARISON (5-fold CV, higher is better)")
    print("=" * 78)
    header = f"{'model':<22s}" + "".join(f"{m:>11s}" for m in ["accuracy", "precision", "recall", "f1", "roc_auc"])
    print(header)
    print("-" * 78)
    for name, summary in results.items():
        row = f"{name:<22s}"
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean, _ = summary[metric]
            row += f"{mean:>11.4f}"
        print(row)
    print("=" * 78)

    # Pick the winner on F1 (our primary metric).
    winner = max(results.items(), key=lambda kv: kv[1]["f1"][0])
    print(f"\nBest F1: {winner[0]} ({winner[1]['f1'][0]:.4f})")
    print(f"\nOpen the MLflow UI, tick all three `compare-*` runs, and click Compare.")


if __name__ == "__main__":
    main()
