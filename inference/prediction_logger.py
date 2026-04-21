"""
Prediction logging
====================
Every prediction served by the API is recorded in a dedicated Neon table
so we can measure real-world accuracy later.

WHY LOG PREDICTIONS AT ALL?
  Cross-validation and backtest tell us how the model does on HISTORICAL
  data. Neither tells us how the model does on PRODUCTION predictions
  after we've shipped them. The only way to measure that is to log what
  we predicted, then wait for the truth to arrive, then compare.

WHAT WE LOG
  For each prediction: who it was about, what we said, what model version
  said it, what endpoint served it, and whether it was freshly computed
  or served from cache. Everything you'd need to retrospectively audit a
  specific prediction, six months from now, without any other context.

HOW WE'LL MEASURE ACCURACY LATER
  After ~1-2 weeks of predictions have accumulated, join ml_prediction_log
  with study_streaks on user_id. For each logged prediction, ask:
     "In the 7 days after the prediction was logged, did this user have
      any activity?"
  That answers the binary question the model was trying to predict.
  Compare logged `risk_level` / `dropout_probability` to the truth ->
  real-world precision, recall, F1. (See evaluation/measure_accuracy.py
  once that's built.)

SCHEMA
  Created automatically on first write via `ensure_log_table_exists()`.
  Idempotent — safe to call at every API startup.

PERFORMANCE NOTES
  - Batch endpoints (/predictions, /summary) log 2500 rows per call.
    We batch them into one INSERT statement and run it in a FastAPI
    BackgroundTask so the HTTP response returns without waiting.
  - Single-user endpoints log one row synchronously (negligible cost).
"""
import logging
from datetime import datetime
from typing import Iterable

from sqlalchemy import create_engine, text

from config import DATABASE_URL


_logger = logging.getLogger("ml_prediction_log")
_engine = None


def _get_engine():
    """Lazily create the SQLAlchemy engine. Pool size 1 because this runs
    inside the already-multi-threaded FastAPI process — we don't want to
    blow the Neon connection limit."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_size=1, max_overflow=2)
    return _engine


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_prediction_log (
    id                         BIGSERIAL PRIMARY KEY,
    logged_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id                    INTEGER     NOT NULL,
    dropout_probability        DOUBLE PRECISION NOT NULL,
    risk_level                 TEXT        NOT NULL,
    model_version              TEXT        NOT NULL,
    source                     TEXT        NOT NULL,
    endpoint                   TEXT        NOT NULL,
    days_since_last_activity   INTEGER,
    has_subscription           BOOLEAN
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_mpl_user_id    ON ml_prediction_log (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_mpl_logged_at  ON ml_prediction_log (logged_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_mpl_version    ON ml_prediction_log (model_version);",
]


def ensure_log_table_exists() -> None:
    """Idempotent — creates the table and indexes if they don't exist."""
    try:
        with _get_engine().begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
            for idx_sql in _CREATE_INDEXES_SQL:
                conn.execute(text(idx_sql))
    except Exception as exc:
        _logger.warning("ensure_log_table_exists failed (logging disabled): %s", exc)


_INSERT_SQL = text("""
    INSERT INTO ml_prediction_log
        (user_id, dropout_probability, risk_level,
         model_version, source, endpoint,
         days_since_last_activity, has_subscription)
    VALUES
        (:user_id, :dropout_probability, :risk_level,
         :model_version, :source, :endpoint,
         :days_since_last_activity, :has_subscription)
""")


def _pred_to_row(p: dict, model_version: str, source: str, endpoint: str) -> dict:
    return {
        "user_id": int(p["user_id"]),
        "dropout_probability": float(p["dropout_probability"]),
        "risk_level": p["risk_level"],
        "model_version": model_version,
        "source": source,
        "endpoint": endpoint,
        "days_since_last_activity": int(p.get("days_since_last_activity", 0)),
        "has_subscription": bool(p.get("has_subscription", False)),
    }


def log_prediction(p: dict, model_version: str, source: str, endpoint: str) -> None:
    """Synchronously write one prediction. Never raises — logging failures
    don't take down the API."""
    try:
        with _get_engine().begin() as conn:
            conn.execute(_INSERT_SQL, _pred_to_row(p, model_version, source, endpoint))
    except Exception as exc:
        _logger.warning("log_prediction failed: %s", exc)


def log_predictions_batch(
    predictions: Iterable[dict],
    model_version: str,
    source: str,
    endpoint: str,
) -> None:
    """Bulk-insert many predictions in one round trip.

    Intended for FastAPI BackgroundTasks so the HTTP response returns
    before the log write commits. Logging failures are swallowed so
    monitoring problems don't become availability problems.
    """
    rows = [_pred_to_row(p, model_version, source, endpoint) for p in predictions]
    if not rows:
        return
    try:
        with _get_engine().begin() as conn:
            conn.execute(_INSERT_SQL, rows)
    except Exception as exc:
        _logger.warning("log_predictions_batch failed (%d rows): %s", len(rows), exc)
