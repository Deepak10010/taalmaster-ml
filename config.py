"""
Configuration
=============
Loads BigQuery + auth settings from .env. As of step 8 of the data platform
refactor, ML reads from BigQuery marts instead of Neon directly. See
`data_platform/RUNBOOK.md` for the architectural rationale.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── BigQuery: where to read source data from ──────────────────────────────
# google-cloud-bigquery picks up the service-account key from
# GOOGLE_APPLICATION_CREDENTIALS automatically.
BQ_PROJECT         = os.getenv("BQ_PROJECT", "taalmaster-data-dev")
BQ_RAW_DATASET     = os.getenv("BQ_RAW_DATASET",     "taalmaster_raw_dev")
BQ_STAGING_DATASET = os.getenv("BQ_STAGING_DATASET", "taalmaster_staging_dev")
BQ_MARTS_DATASET   = os.getenv("BQ_MARTS_DATASET",   "taalmaster_marts_dev")

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    raise RuntimeError(
        "GOOGLE_APPLICATION_CREDENTIALS not set. Point it at your service-account "
        "JSON key file (e.g. ../taalmaster/data_platform/airflow/include/gcp-key.json). "
        "Copy .env.example to .env and edit accordingly."
    )


# ── Neon Postgres (kept for prediction logging only) ──────────────────────
# DATABASE_URL is no longer used by data_extraction. It IS still used by
# inference/prediction_logger.py to write back into the operational DB so the
# admin dashboard can show predictions.
DATABASE_URL = os.getenv("DATABASE_URL")


# ── MLflow ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT      = "taalmaster-dropout-prediction"
REGISTERED_MODEL_NAME  = "taalmaster-dropout"


# ── Trained model artifacts ───────────────────────────────────────────────
MODEL_DIR         = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH        = MODEL_DIR / "dropout_model.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.joblib"


# ── THE KEY BUSINESS DECISION ──
# A student inactive for 7+ days is labeled "dropout".
DROPOUT_DAYS_THRESHOLD = 7
