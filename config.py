"""
Configuration
Loads the DATABASE_URL from .env (same Neon DB as TaalMaster).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set – copy .env.example to .env and fill it in")

# MLflow stores experiment runs locally in ./mlruns
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT = "taalmaster-dropout-prediction"

# Trained model artifacts
MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dropout_model.joblib"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.joblib"

# ── THE KEY BUSINESS DECISION ──
# A student inactive for 7+ days is labeled "dropout".
# Changing this changes what the model learns:
#   - 3 days  → aggressive, catches people on vacation
#   - 7 days  → balanced (our choice)
#   - 14 days → conservative, misses slow faders
DROPOUT_DAYS_THRESHOLD = 7
