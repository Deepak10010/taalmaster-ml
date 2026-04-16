"""
DVC STAGE 3 — Train
=====================
Loads the feature matrix from STAGE 2, calls the training routine, and
writes two things DVC knows how to track:
  - model artifacts  → artifacts/dropout_model.joblib, feature_columns.joblib
  - metrics          → metrics.json  (cv_f1_mean, cv_roc_auc_mean, etc.)

`dvc metrics show` and `dvc metrics diff` read metrics.json, so every
DVC run records a reproducible performance snapshot.
"""
import json
from pathlib import Path

import pandas as pd

from training.train import train_model
from transformation.feature_engineering import get_feature_columns


ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "dvc" / "features.parquet"
METRICS_PATH = ROOT / "metrics.json"


def main() -> None:
    print(f"[train] loading features from {FEATURES_PATH}")
    features = pd.read_parquet(FEATURES_PATH)

    feature_cols = get_feature_columns()
    results = train_model(features, feature_cols)

    # The model + feature_columns files are written inside train_model()
    # to artifacts/. MLflow also captured them. We additionally write
    # a flat metrics.json for DVC's metrics view.
    metrics = {
        "cv_f1_mean": results["cv_metrics"]["f1"]["mean"],
        "cv_f1_std": results["cv_metrics"]["f1"]["std"],
        "cv_roc_auc_mean": results["cv_metrics"]["roc_auc"]["mean"],
        "cv_precision_mean": results["cv_metrics"]["precision"]["mean"],
        "cv_recall_mean": results["cv_metrics"]["recall"]["mean"],
        "cv_accuracy_mean": results["cv_metrics"]["accuracy"]["mean"],
        "n_students": int(len(features)),
        "dropout_rate": float(features["is_dropout"].mean()),
        "mlflow_run_id": results["run_id"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] wrote {METRICS_PATH}")
    print(f"        CV F1 = {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
    print(f"        MLflow run: {metrics['mlflow_run_id']}")


if __name__ == "__main__":
    main()
