"""
Prediction cache for the serving layer.

WHY
  /summary and /predictions re-run extract_all() + predict_all_students()
  end-to-end on every request — roughly 20-30 seconds on the current
  dataset. That was fine as a proof of concept but it makes the admin
  dashboard unusable and wastes compute.

  This module serves the latest predictions JSON that pipeline.py already
  writes to data/predictions/ on every scheduled run. The file is one day
  old at most (the scheduled task runs daily at 10am), which is a perfectly
  acceptable lag for a dropout-risk dashboard. Operators who need the very
  latest numbers can force a live recompute by passing ?fresh=true.

CONTRACT
  - `get_predictions_for_serving(force_fresh=False)` returns a tuple
    `(predictions: list[dict], source: "cache" | "live")`.
  - `source` is surfaced in API responses so operators can tell which
    request paid the 30-second cost and which was served instantly.
  - When no fresh snapshot exists on disk, this falls back to live compute
    so the API still works on a brand-new deployment.

CONFIGURATION
  PREDICTION_CACHE_MAX_AGE_MINUTES  (default: 1440 = 24h)
      A snapshot older than this is considered stale and triggers a live
      recompute. Set to 0 to disable caching entirely.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "predictions"
_DEFAULT_MAX_AGE_MINUTES = 1440  # 24 hours


def _max_age_minutes() -> int:
    try:
        return int(os.getenv("PREDICTION_CACHE_MAX_AGE_MINUTES", _DEFAULT_MAX_AGE_MINUTES))
    except ValueError:
        return _DEFAULT_MAX_AGE_MINUTES


# In-memory cache of the parsed JSON so repeated requests don't re-parse
# the file. Keyed by file path so the cache invalidates automatically
# when pipeline.py writes a new predictions_*.json.
_mem_cache: dict = {"path": None, "data": None, "loaded_at": None}


def _latest_predictions_file() -> Optional[Path]:
    if not PREDICTIONS_DIR.exists():
        return None
    candidates = sorted(PREDICTIONS_DIR.glob("predictions_*.json"))
    return candidates[-1] if candidates else None


def _load_cached(max_age_minutes: int) -> Optional[list[dict]]:
    """Returns the latest on-disk predictions if fresh enough, else None."""
    if max_age_minutes <= 0:
        return None

    path = _latest_predictions_file()
    if path is None:
        return None

    # Check file mtime for freshness (not load time — the file could have been
    # sitting on disk for a week before the API started).
    age_seconds = datetime.utcnow().timestamp() - path.stat().st_mtime
    if age_seconds > max_age_minutes * 60:
        return None

    # Reuse the parsed JSON if we've already loaded this exact file.
    if _mem_cache["path"] == str(path) and _mem_cache["data"] is not None:
        return _mem_cache["data"]

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    predictions = payload.get("predictions", [])
    _mem_cache["path"] = str(path)
    _mem_cache["data"] = predictions
    _mem_cache["loaded_at"] = datetime.utcnow()
    return predictions


def _run_live() -> list[dict]:
    """Fresh extract + score. Slow (~30s) but always authoritative."""
    # Imports here so importing this module doesn't drag the DB driver
    # and the model into memory before it's needed.
    from ingestion.data_extraction import extract_all
    from inference.predict import predict_all_students

    raw = extract_all()
    return predict_all_students(raw)


def get_predictions_for_serving(force_fresh: bool = False) -> tuple[list[dict], str]:
    """
    The entry point every API endpoint should use.

    Returns (predictions, source) where source is 'cache' or 'live'.
    """
    if not force_fresh:
        cached = _load_cached(_max_age_minutes())
        if cached is not None:
            return cached, "cache"
    return _run_live(), "live"


def invalidate_cache() -> None:
    """Drop the in-memory copy. Useful in tests or after manual refresh."""
    _mem_cache["path"] = None
    _mem_cache["data"] = None
    _mem_cache["loaded_at"] = None


def cache_info() -> dict:
    """Shape: {'path': str|None, 'age_seconds': float|None, 'max_age_seconds': int}"""
    path = _mem_cache["path"]
    age = None
    if path and Path(path).exists():
        age = datetime.utcnow().timestamp() - Path(path).stat().st_mtime
    return {
        "path": path,
        "age_seconds": age,
        "max_age_seconds": _max_age_minutes() * 60,
    }
