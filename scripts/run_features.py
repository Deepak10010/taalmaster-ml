"""
DVC STAGE 2 — Feature engineering
===================================
Reads the raw parquet snapshot written by STAGE 1, builds the 32-feature
matrix with the temporal split, and writes a single parquet file at a
stable path. Parameters come from params.yaml so DVC can track them.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from transformation.feature_engineering import build_feature_matrix


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "dvc" / "raw"
OUT_PATH = ROOT / "data" / "dvc" / "features.parquet"
PARAMS_PATH = ROOT / "params.yaml"


def load_raw() -> dict:
    """Rebuild the dict shape that build_feature_matrix expects.

    Strips tz from any tz-aware datetime columns. Older parquet snapshots were
    written with `datetime64[..., UTC]` (BQ TIMESTAMP), and feature_engineering
    compares those against tz-naive `pd.Timestamp(ref_date)`, which raises
    `Cannot compare tz-naive and tz-aware datetime-like objects`. The newer
    extract path strips tz upstream, but normalize here too so a stale parquet
    can't break the build.
    """
    raw = {}
    for key in ("students", "streaks", "quizzes", "words", "writing", "audio"):
        df = pd.read_parquet(RAW_DIR / f"{key}.parquet")
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        raw[key] = df
    with open(RAW_DIR / "content_counts.json") as f:
        raw["content_counts"] = json.load(f)
    return raw


def main() -> None:
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    ref_offset_days = params["features"]["ref_offset_days"]
    window_days = params["features"]["window_days"]
    ref_date = datetime.utcnow() - timedelta(days=ref_offset_days)

    print(f"[features] ref_date={ref_date.date()}  window={window_days}d")
    raw = load_raw()

    features = build_feature_matrix(raw, ref_date=ref_date, label_window_days=window_days)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)

    dropout_rate = features["is_dropout"].mean()
    print(f"[features] wrote {OUT_PATH}")
    print(f"           shape = {features.shape[0]:,} x {features.shape[1]}   "
          f"dropout_rate = {dropout_rate:.1%}")


if __name__ == "__main__":
    main()
