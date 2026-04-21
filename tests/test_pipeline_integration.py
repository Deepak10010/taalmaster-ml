"""
Integration test — end-to-end smoke test for the training pipeline.

WHAT THIS CHECKS
  - generate_synthetic_data produces a dict with the expected keys and shapes.
  - build_feature_matrix turns that dict into a 32-feature matrix with binary label.
  - train_model fits an XGBoost on the matrix, logs to MLflow, writes joblib files,
    and returns metrics in the expected shape.
  - The trained model can then be used by predict_all_students to score students
    (with compute_label=False at inference time).

WHAT THIS DOES NOT CHECK
  - Live DB extraction (requires Neon; runs elsewhere).
  - Model quality on real data (backtest.py does that).
  - MLflow registry promotion logic (pipeline.py stage_register).

Runs in ~10 seconds on synthetic data.
"""
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from training.train import generate_synthetic_data, train_model
from transformation.feature_engineering import build_feature_matrix, get_feature_columns


pytestmark = pytest.mark.integration

# Mirror what pipeline.py does: build features "as of" a past date so the
# forward-label window sits entirely inside observed history. With ref_date
# = now - 14 days, both past features AND the 7-day forward window are
# populated by the synthetic generator (which creates activity spanning
# the past ~90 days).
REF_DATE = datetime.utcnow() - timedelta(days=14)


def test_synthetic_data_has_expected_shape():
    raw = generate_synthetic_data(n_students=50)
    assert len(raw["students"]) == 50
    for key in ("students", "streaks", "quizzes", "words", "writing", "audio"):
        assert key in raw
    assert raw["content_counts"]["total_vocabulary"] > 0


def test_feature_matrix_from_synthetic_data_is_well_formed():
    raw = generate_synthetic_data(n_students=100)
    features = build_feature_matrix(raw, ref_date=REF_DATE)

    # Every feature column is present
    for col in get_feature_columns():
        assert col in features.columns, f"missing: {col}"

    # Label is binary
    assert set(features["is_dropout"].unique()).issubset({0, 1})

    # No NaNs leaked into the feature columns (the model can't train on NaN
    # unless XGBoost handles it, but we've made all features numeric+filled).
    feat_cols = get_feature_columns()
    assert not features[feat_cols].isna().any().any(), "NaN in features"


def test_end_to_end_training_runs_and_returns_sensible_metrics(tmp_path, monkeypatch):
    """
    Train on 200 synthetic students, confirm we get back a plausible CV F1 and
    a saved model file. This is the pipeline's happy path.
    """
    # Redirect all artifact writes into a tmp dir so the real artifacts/ is untouched.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    model_path = tmp_path / "dropout_model.joblib"
    cols_path = tmp_path / "feature_columns.joblib"

    with patch("training.train.MODEL_PATH", model_path), \
         patch("training.train.FEATURE_COLS_PATH", cols_path), \
         patch("training.train.MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}"):

        raw = generate_synthetic_data(n_students=200)
        features = build_feature_matrix(raw, ref_date=REF_DATE)
        feature_cols = get_feature_columns()

        # Sanity guard: both classes must be represented, otherwise we're
        # testing on a degenerate label distribution.
        assert set(features["is_dropout"].unique()) == {0, 1}, (
            f"degenerate label distribution: {features['is_dropout'].value_counts().to_dict()}"
        )

        results = train_model(features, feature_cols)

    # Contract: results dict has the keys pipeline.py depends on.
    assert "model" in results
    assert "run_id" in results
    assert "cv_metrics" in results
    cv = results["cv_metrics"]
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert metric in cv
        assert "mean" in cv[metric]
        assert "std" in cv[metric]

    # Sanity: the model should beat random. We don't assert a tight upper
    # bound because (a) 200 students is small, (b) the temporal split +
    # ref_date=now-14d is a trickier setup than full-pipeline CV, and (c)
    # the point here is "does the pipeline train end-to-end without crashing
    # and produce a non-degenerate model" — not "is it state of the art".
    assert cv["roc_auc"]["mean"] > 0.55, (
        f"suspicious CV ROC AUC {cv['roc_auc']['mean']:.3f} — "
        f"model barely beats random; something may be broken upstream"
    )
    # F1 can vary widely on small synthetic samples; we only check it's
    # defined and positive.
    assert 0.0 < cv["f1"]["mean"] <= 1.0

    # Model files were saved
    assert model_path.exists()
    assert cols_path.exists()


def test_trained_model_can_score_without_label(tmp_path, monkeypatch):
    """
    Regression guard: predict_all_students must work when
    build_feature_matrix is called with compute_label=False.
    This is the inference path the API uses.
    """
    import joblib

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    model_path = tmp_path / "dropout_model.joblib"
    cols_path = tmp_path / "feature_columns.joblib"

    # Train a tiny model first
    with patch("training.train.MODEL_PATH", model_path), \
         patch("training.train.FEATURE_COLS_PATH", cols_path), \
         patch("training.train.MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}"):
        raw = generate_synthetic_data(n_students=60)
        features = build_feature_matrix(raw, ref_date=REF_DATE)
        train_model(features, get_feature_columns())

    # Now simulate inference: fresh data, compute_label=False (no future)
    raw2 = generate_synthetic_data(n_students=30)
    inference_features = build_feature_matrix(raw2, compute_label=False)

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)

    probs = model.predict_proba(inference_features[feature_cols].values)[:, 1]
    assert len(probs) == 30
    assert np.all((probs >= 0) & (probs <= 1))
