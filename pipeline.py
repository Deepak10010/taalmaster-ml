"""
END-TO-END PIPELINE
===================
Orchestrates the full ML flow shown in the architecture diagram:

    Neon DB ──▶ Extract ──▶ Feature Engineering ──▶ Train (MLflow)
                                                        │
                                                        ▼
    Predictions (JSON) ◀── Predict ◀── MLflow Registry (Production stage)

WHY A SEPARATE PIPELINE SCRIPT?
  - train.py trains a model once; predict.py scores once; api.py serves.
  - pipeline.py is what you actually schedule / trigger (the "Retrain
    trigger" box in the diagram). It extracts, versions the data, trains,
    evaluates, promotes the best model, and writes fresh predictions.
  - Each stage saves a timestamped snapshot to data/, so any run is
    fully reproducible (and ready for DVC versioning later).

Usage:
    python pipeline.py                      # Full pipeline (default)
    python pipeline.py --skip-train         # Only refresh predictions
    python pipeline.py --no-register        # Train, but don't touch the registry
    python pipeline.py --ref-date 2026-04-16  # Evaluate "as of" a past date
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    MODEL_PATH, FEATURE_COLS_PATH, DROPOUT_DAYS_THRESHOLD,
)
from ingestion.data_extraction import extract_all
from transformation.feature_engineering import build_feature_matrix, get_feature_columns
from training.train import train_model
from inference.predict import predict_all_students


# ---------------------------------------------------------------------------
# Where each stage's output lands. Timestamped so we never overwrite history.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

for d in (RAW_DIR, FEATURES_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

REGISTERED_MODEL_NAME = "taalmaster-dropout"

# F1 is our primary metric — if a new model beats the Production model on
# cv_f1_mean, we promote it. Lower bar than in production but sensible for
# a teaching repo where the dataset is small.
PROMOTION_METRIC = "cv_f1_mean"


# ---------------------------------------------------------------------------
# STAGE 1 — Extract from Neon DB and snapshot to disk
# ---------------------------------------------------------------------------
def stage_extract(run_stamp: str) -> dict:
    print("\n[1/5] Extracting from Neon DB...")
    raw = extract_all()

    snapshot_dir = RAW_DIR / run_stamp
    snapshot_dir.mkdir(exist_ok=True)

    # Save each DataFrame as parquet — compact, typed, DVC-friendly.
    for key, df in raw.items():
        if isinstance(df, pd.DataFrame):
            df.to_parquet(snapshot_dir / f"{key}.parquet", index=False)
    # content_counts is a dict, save as json
    with open(snapshot_dir / "content_counts.json", "w") as f:
        json.dump(raw["content_counts"], f, indent=2)

    print(f"      Students:         {len(raw['students']):,}")
    print(f"      Streak records:   {len(raw['streaks']):,}")
    print(f"      Quiz attempts:    {len(raw['quizzes']):,}")
    print(f"      Word progress:    {len(raw['words']):,}")
    print(f"      Writing subs:     {len(raw['writing']):,}")
    print(f"      Audio listened:   {len(raw['audio']):,}")
    print(f"      Snapshot:         {snapshot_dir}")
    return raw


# ---------------------------------------------------------------------------
# STAGE 2 — Engineer features and snapshot the matrix
# ---------------------------------------------------------------------------
def stage_features(raw: dict, ref_date: datetime, label_window_days: int, run_stamp: str) -> pd.DataFrame:
    print("\n[2/5] Engineering features...")
    features = build_feature_matrix(raw, ref_date, label_window_days=label_window_days, compute_label=True)
    feature_cols = get_feature_columns()

    out = FEATURES_DIR / f"features_{run_stamp}.parquet"
    features.to_parquet(out, index=False)

    dropout_rate = features["is_dropout"].mean()
    window_end = ref_date + timedelta(days=label_window_days)
    print(f"      ref_date:         {ref_date.isoformat()}  (features use data BEFORE this)")
    print(f"      label window:     [{ref_date.date()}, {window_end.date()})  (look forward {label_window_days}d)")
    print(f"      Matrix shape:     {features.shape[0]:,} students x {len(feature_cols)} features")
    print(f"      Dropout rate:     {dropout_rate:.1%}  (went silent in forward window)")
    print(f"      Snapshot:         {out}")
    return features


# ---------------------------------------------------------------------------
# STAGE 3 — Train and log to MLflow
# ---------------------------------------------------------------------------
def stage_train(features: pd.DataFrame) -> dict:
    print("\n[3/5] Training XGBoost (5-fold CV + MLflow tracking)...")
    feature_cols = get_feature_columns()
    results = train_model(features, feature_cols)

    cv = results["cv_metrics"]
    print(f"      MLflow run:       {results['run_id']}")
    print(f"      CV F1:            {cv['f1']['mean']:.4f} (+/- {cv['f1']['std']:.4f})")
    print(f"      CV ROC AUC:       {cv['roc_auc']['mean']:.4f} (+/- {cv['roc_auc']['std']:.4f})")
    print(f"      Model saved:      {MODEL_PATH}")
    return results


# ---------------------------------------------------------------------------
# STAGE 4 — Register in the MLflow model registry
# ---------------------------------------------------------------------------
# WHY A REGISTRY?
# MLflow logs every run, but the registry answers "which model is serving
# predictions RIGHT NOW?" It tracks stages (None → Staging → Production)
# so the API always loads a known-good version.
def stage_register(train_results: dict) -> str | None:
    print("\n[4/5] Registering model in MLflow registry...")
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1) Register this run's model as a new version
    model_uri = f"runs:/{train_results['run_id']}/model"
    new_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    new_f1 = train_results["cv_metrics"]["f1"]["mean"]
    print(f"      Registered:       {REGISTERED_MODEL_NAME} v{new_version.version}")
    print(f"      New CV F1:        {new_f1:.4f}")

    # 2) Compare against the current Production version (if any)
    try:
        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    except Exception:
        prod_versions = []

    should_promote = True
    if prod_versions:
        prod = prod_versions[0]
        prod_run = client.get_run(prod.run_id)
        prod_f1 = prod_run.data.metrics.get(PROMOTION_METRIC)
        print(f"      Current Prod:     v{prod.version} (F1: {prod_f1:.4f})" if prod_f1 else f"      Current Prod:     v{prod.version}")
        if prod_f1 is not None and new_f1 <= prod_f1:
            should_promote = False
            print(f"      Decision:         KEEP current Production (new model did not beat it)")

    # 3) Promote if it wins (or if there's no incumbent)
    if should_promote:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=new_version.version,
            stage="Production",
            archive_existing_versions=True,  # demote old Production to Archived
        )
        print(f"      Decision:         PROMOTED v{new_version.version} to Production")
        return new_version.version

    return None


# ---------------------------------------------------------------------------
# STAGE 5 — Score all students and save predictions
# ---------------------------------------------------------------------------
def stage_predict(raw: dict, run_stamp: str) -> list[dict]:
    print("\n[5/5] Generating predictions for all students...")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model found at {MODEL_PATH}. Run training first.")

    predictions = predict_all_students(raw)

    out = PREDICTIONS_DIR / f"predictions_{run_stamp}.json"
    with open(out, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "total_students": len(predictions),
            "predictions": predictions,
        }, f, indent=2, default=str)

    high = sum(1 for p in predictions if p["risk_level"] == "high")
    medium = sum(1 for p in predictions if p["risk_level"] == "medium")
    low = sum(1 for p in predictions if p["risk_level"] == "low")
    print(f"      Total scored:     {len(predictions):,}")
    print(f"      High risk:        {high:,}")
    print(f"      Medium risk:      {medium:,}")
    print(f"      Low risk:         {low:,}")
    print(f"      Saved to:         {out}")
    return predictions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(ref_date: datetime | None = None,
                 label_window_days: int = DROPOUT_DAYS_THRESHOLD,
                 skip_train: bool = False,
                 register: bool = True) -> dict:
    now = datetime.utcnow()
    run_stamp = now.strftime("%Y%m%d_%H%M%S")

    # TEMPORAL SPLIT: by default we train on a past snapshot so the forward
    # label window is FULLY observed. Picking ref_date = now would make every
    # label = 1 (no future activity has happened yet) — useless for training.
    # ref_date = now - label_window - 1d buffer.
    if ref_date is None:
        ref_date = now - timedelta(days=label_window_days + 1)

    print("=" * 64)
    print(f"TaalMaster ML Pipeline — run {run_stamp}")
    print(f"Training ref_date: {ref_date.isoformat()}")
    print(f"Forward window:    {label_window_days} days")
    print("=" * 64)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Stage 1 + 2 always run — we need fresh data to score
    raw = stage_extract(run_stamp)
    features = stage_features(raw, ref_date, label_window_days, run_stamp)

    train_results = None
    promoted_version = None
    if not skip_train:
        train_results = stage_train(features)
        if register:
            promoted_version = stage_register(train_results)
    else:
        print("\n[3/5] Skipped training (--skip-train).")
        print("[4/5] Skipped registry promotion.")

    predictions = stage_predict(raw, run_stamp)

    print("\n" + "=" * 64)
    print("PIPELINE COMPLETE")
    print("=" * 64)
    print(f"Run stamp:       {run_stamp}")
    if train_results:
        print(f"MLflow run ID:   {train_results['run_id']}")
    if promoted_version:
        print(f"New Production:  {REGISTERED_MODEL_NAME} v{promoted_version}")
    print(f"MLflow UI:       mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    return {
        "run_stamp": run_stamp,
        "train_results": train_results,
        "promoted_version": promoted_version,
        "n_predictions": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="TaalMaster ML end-to-end pipeline")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; only refresh predictions using the current model")
    parser.add_argument("--no-register", action="store_true",
                        help="Train but skip MLflow model registry promotion")
    parser.add_argument("--ref-date", type=str, default=None,
                        help="Evaluate 'as of' this date (ISO format, e.g. 2026-04-16)")
    args = parser.parse_args()

    ref_date = datetime.fromisoformat(args.ref_date) if args.ref_date else None

    run_pipeline(
        ref_date=ref_date,
        skip_train=args.skip_train,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
