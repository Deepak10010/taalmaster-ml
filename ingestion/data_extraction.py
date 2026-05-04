"""
STEP 1 — Data Extraction (BigQuery)
====================================
Pull student engagement data from the TaalMaster data platform's BigQuery
marts and staging layer — NOT directly from Neon Postgres.

Why BigQuery instead of Neon?
  - dbt has already cleaned, deduped, and tested the data (75 quality checks).
  - Pre-computed features like score_pct, accuracy_rate, account_age_days save
    work downstream.
  - ML training queries don't compete with the live app for Neon connection slots.
  - BQ is columnar — analytical queries are 10-100× faster than the same query
    on Postgres.

  See `data_platform/RUNBOOK.md` -> "Architectural decision: why ML reads from
  BigQuery, not Neon" for the full rationale.

WHAT WE PULL (and from which mart/staging table):
  - get_students             -> dim_users  (joined + flags pre-computed)
  - get_study_streaks        -> stg_study_streaks
  - get_quiz_attempts        -> fct_quiz_attempts (with score_pct, performance_tier)
  - get_word_progress        -> stg_word_progress (with accuracy_rate)
  - get_writing_submissions  -> stg_writing_submissions (with word_count)
  - get_audio_progress       -> raw.audio_progress (filtered to listened rows)
  - get_total_content_counts -> raw.{vocabulary, audio_files, writing_prompts, sections}

This module's function signatures (names, return-DataFrame columns) match the
previous Neon-based implementation — feature_engineering.py is unchanged.
"""
from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from config import BQ_PROJECT, BQ_RAW_DATASET, BQ_STAGING_DATASET, BQ_MARTS_DATASET


_client: bigquery.Client | None = None


def _bq() -> bigquery.Client:
    """Lazy singleton — avoids creating a client at import time."""
    global _client
    if _client is None:
        _client = bigquery.Client(project=BQ_PROJECT)
    return _client


def _q(sql: str) -> pd.DataFrame:
    """Run a query, return a DataFrame.

    Two normalizations at the BQ → pandas boundary:
      * `db-dtypes` extension columns (dbdate, dbtime) → native pandas dtypes.
        Without this, parquet round-trips fail with `TypeError: data type
        'dbdate' not understood`.
      * tz-aware `datetime64[..., UTC]` columns → tz-naive. BQ TIMESTAMPs land
        as tz-aware, but the rest of the feature pipeline uses tz-naive
        `datetime.utcnow()`, which would otherwise raise
        `Cannot compare tz-naive and tz-aware datetime-like objects`.
    """
    df = _bq().query(sql).to_dataframe()
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        if dtype_name == "dbdate":
            df[col] = pd.to_datetime(df[col])
        elif dtype_name == "dbtime":
            df[col] = df[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Fact / dim mart reads
# ──────────────────────────────────────────────────────────────────────────

def get_students() -> pd.DataFrame:
    """
    All student accounts. Reads from dim_users (which already joins users with
    pre-computed flags from dbt).

    Returns columns matching the legacy Neon shape so downstream code is
    unchanged. Subscription columns will be NULL until subscriptions data
    flows through (table is empty in Neon as of this refactor).
    """
    sql = f"""
        SELECT
            user_id,
            user_name                                  AS name,
            email,
            created_at                                 AS account_created,
            trial_ends_at,
            -- Subscription columns sourced from raw when available (empty today).
            CAST(NULL AS STRING)                       AS subscription_status,
            CAST(NULL AS STRING)                       AS stripe_price_id,
            CAST(NULL AS TIMESTAMP)                    AS subscription_expires,
            CAST(NULL AS BOOL)                         AS cancel_at_period_end
        FROM `{BQ_PROJECT}.{BQ_MARTS_DATASET}.dim_users`
        WHERE role = 'student'
        ORDER BY user_id
    """
    return _q(sql)


def get_study_streaks() -> pd.DataFrame:
    sql = f"""
        SELECT user_id, activity_date
        FROM `{BQ_PROJECT}.{BQ_STAGING_DATASET}.stg_study_streaks`
        ORDER BY user_id, activity_date
    """
    return _q(sql)


def get_quiz_attempts() -> pd.DataFrame:
    """
    Quiz attempts with per-attempt enrichment from dbt (score_pct,
    performance_tier). Returns the legacy column set + the new derived ones —
    feature_engineering.py can use them for free.
    """
    sql = f"""
        SELECT
            user_id,
            section_id,
            quiz_mode AS mode,
            score,
            total_questions,
            completed_at,
            -- New columns from dbt enrichment (NULL-safe to use in features):
            score_pct,
            performance_tier
        FROM `{BQ_PROJECT}.{BQ_MARTS_DATASET}.fct_quiz_attempts`
        ORDER BY user_id, completed_at
    """
    return _q(sql)


def get_word_progress() -> pd.DataFrame:
    """
    Per (user, vocabulary) row. dbt's stg_word_progress added an `accuracy_rate`
    column that's cheaper to read than recomputing in pandas.
    """
    sql = f"""
        SELECT
            user_id,
            vocabulary_id,
            status,
            attempts,
            correct_attempts,
            last_seen,
            accuracy_rate
        FROM `{BQ_PROJECT}.{BQ_STAGING_DATASET}.stg_word_progress`
    """
    return _q(sql)


def get_writing_submissions() -> pd.DataFrame:
    sql = f"""
        SELECT
            user_id,
            prompt_id,
            score,
            submitted_at,
            word_count
        FROM `{BQ_PROJECT}.{BQ_STAGING_DATASET}.stg_writing_submissions`
        ORDER BY user_id, submitted_at
    """
    return _q(sql)


def get_audio_progress() -> pd.DataFrame:
    """
    Audio listening events. We ingest with `listened_at` as the watermark, which
    means rows with NULL listened_at (i.e. not listened yet) are naturally
    excluded. So `WHERE listened = true` is implicit.
    """
    sql = f"""
        SELECT user_id, audio_id, listened, listened_at
        FROM `{BQ_PROJECT}.{BQ_RAW_DATASET}.audio_progress`
        WHERE listened = true
    """
    return _q(sql)


# ──────────────────────────────────────────────────────────────────────────
# Content totals — for "coverage %" features (e.g. "what fraction of all
# vocabulary has this user mastered?")
# ──────────────────────────────────────────────────────────────────────────

def get_total_content_counts() -> dict:
    """
    How much content exists on the platform. We don't have dbt models for the
    content tables (they're nearly static reference data) — read counts straight
    from the raw layer.
    """
    sql = f"""
        SELECT
          (SELECT COUNT(*) FROM `{BQ_PROJECT}.{BQ_RAW_DATASET}.vocabulary`)      AS total_vocabulary,
          (SELECT COUNT(*) FROM `{BQ_PROJECT}.{BQ_RAW_DATASET}.audio_files`)     AS total_audio,
          (SELECT COUNT(*) FROM `{BQ_PROJECT}.{BQ_RAW_DATASET}.writing_prompts`) AS total_prompts,
          (SELECT COUNT(*) FROM `{BQ_PROJECT}.{BQ_RAW_DATASET}.sections`)        AS total_sections
    """
    row = _q(sql).iloc[0]
    return {
        "total_vocabulary": int(row["total_vocabulary"]),
        "total_audio":      int(row["total_audio"]),
        "total_prompts":    int(row["total_prompts"]),
        "total_sections":   int(row["total_sections"]),
    }


# ──────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────

def extract_all() -> dict:
    """Pull everything needed for feature engineering. One call, all data."""
    return {
        "students":       get_students(),
        "streaks":        get_study_streaks(),
        "quizzes":        get_quiz_attempts(),
        "words":          get_word_progress(),
        "writing":        get_writing_submissions(),
        "audio":          get_audio_progress(),
        "content_counts": get_total_content_counts(),
    }
