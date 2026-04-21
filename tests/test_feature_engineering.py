"""
Unit tests for transformation/feature_engineering.py.

WHY THESE SPECIFIC TESTS?
  The #1 bug we hit in this project was target leakage — the temporal cut
  being missing on one feature builder let XGBoost memorize the rule.
  These tests lock in:
    1. The temporal cut IS applied (data at/after ref_date is ignored).
    2. The forward label is computed only from data in the forward window.
    3. Edge cases (empty DataFrames, single students) don't explode.
    4. Features degrade gracefully when upstream tables are empty.

  If any of these start failing, something about the temporal semantics has
  drifted and the model may start leaking again.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from transformation.feature_engineering import (
    build_activity_features,
    build_quiz_features,
    build_feature_matrix,
    compute_forward_label,
    get_feature_columns,
)


REF_DATE = datetime(2026, 4, 8)


def _one_student(user_id=1):
    return pd.DataFrame([{
        "user_id": user_id,
        "name": "Test",
        "email": "t@example.com",
        "account_created": REF_DATE - timedelta(days=60),
        "trial_ends_at": REF_DATE - timedelta(days=57),
        "subscription_status": "active",
        "stripe_price_id": "price_monthly",
        "subscription_expires": REF_DATE + timedelta(days=30),
        "cancel_at_period_end": False,
    }])


def _streaks(user_id=1, days_ago_list=()):
    """Build streaks DataFrame with activity_date = REF_DATE - d days for d in days_ago_list."""
    return pd.DataFrame([
        {"user_id": user_id, "activity_date": (REF_DATE - timedelta(days=d)).date()}
        for d in days_ago_list
    ])


# ---------------------------------------------------------------------------
# TEMPORAL CUT — the most important invariant
# ---------------------------------------------------------------------------

def test_activity_features_ignore_data_at_or_after_ref_date():
    """
    Activity on or after ref_date must NOT affect the feature values.
    This is the leakage guard: features look strictly backward.
    """
    students = _one_student()
    # Mix of past and future-relative-to-ref_date activity.
    streaks = _streaks(days_ago_list=[1, 2, 3, 0, -1, -2])
    # days_ago=0 is ref_date itself. days_ago=-1, -2 are after ref_date.

    features = build_activity_features(students, streaks, REF_DATE)
    row = features.iloc[0]

    # Only the 3 genuine past days should count.
    assert row["total_active_days"] == 3, (
        f"expected 3 pre-ref-date active days, got {row['total_active_days']}"
    )
    assert row["active_days_last_7"] == 3
    # last_activity should be ref_date - 1 day
    assert row["days_since_last_activity"] == 1


def test_forward_label_sees_only_future_window():
    """
    compute_forward_label must classify based on data in (ref_date, ref_date + window]
    and nothing else.
    """
    students = _one_student()
    streaks = _streaks(days_ago_list=[10, 5])  # both before ref_date
    # Student is ACTIVE before ref_date but absent in the future window → dropout.
    label = compute_forward_label(students, streaks, REF_DATE, window_days=7)
    assert label.iloc[0]["is_dropout"] == 1

    # Now add one activity inside the forward window.
    streaks = pd.concat([streaks, _streaks(days_ago_list=[-3])], ignore_index=True)
    label = compute_forward_label(students, streaks, REF_DATE, window_days=7)
    assert label.iloc[0]["is_dropout"] == 0


def test_past_activity_does_not_count_as_future():
    """
    A student who was active weeks ago but silent in the forward window
    should be labeled dropout.
    """
    students = _one_student()
    streaks = _streaks(days_ago_list=[30, 20, 10])  # all long before ref_date
    label = compute_forward_label(students, streaks, REF_DATE, window_days=7)
    assert label.iloc[0]["is_dropout"] == 1


# ---------------------------------------------------------------------------
# EMPTY-DATA EDGE CASES — shouldn't crash, should return sensible defaults
# ---------------------------------------------------------------------------

def test_activity_features_handle_empty_streaks():
    students = _one_student()
    empty = pd.DataFrame(columns=["user_id", "activity_date"])

    features = build_activity_features(students, empty, REF_DATE)
    row = features.iloc[0]

    assert row["days_since_last_activity"] == 999.0
    assert row["total_active_days"] == 0.0
    assert row["active_days_last_7"] == 0.0


def test_quiz_features_handle_empty_quizzes():
    students = _one_student()
    empty = pd.DataFrame(columns=["user_id", "section_id", "mode", "score", "total_questions", "completed_at"])

    features = build_quiz_features(students, empty, REF_DATE)
    row = features.iloc[0]

    for col in ("total_quizzes", "avg_quiz_score_pct", "quiz_score_trend"):
        assert row[col] == 0.0


def test_forward_label_handles_empty_streaks():
    """Nobody has activity → everyone is dropout."""
    students = _one_student()
    empty = pd.DataFrame(columns=["user_id", "activity_date"])
    label = compute_forward_label(students, empty, REF_DATE, window_days=7)
    assert label.iloc[0]["is_dropout"] == 1


# ---------------------------------------------------------------------------
# FEATURE MATRIX SHAPE — guards against silent schema drift
# ---------------------------------------------------------------------------

def test_get_feature_columns_returns_32():
    """
    The model is trained on a fixed schema. Adding or removing a feature
    without updating get_feature_columns() breaks everything downstream.
    """
    cols = get_feature_columns()
    assert len(cols) == 32
    assert "days_since_last_activity" in cols
    assert "is_dropout" not in cols, "label must not be in feature columns"
    assert "user_id" not in cols


def test_build_feature_matrix_has_expected_columns():
    """End-to-end: a minimal synthetic input produces a matrix with user_id,
    every feature column, and the label."""
    students = _one_student()
    raw = {
        "students": students,
        "streaks": _streaks(days_ago_list=[1, 3]),
        "quizzes": pd.DataFrame(columns=["user_id", "section_id", "mode", "score", "total_questions", "completed_at"]),
        "words": pd.DataFrame(columns=["user_id", "vocabulary_id", "status", "attempts", "correct_attempts", "last_seen"]),
        "writing": pd.DataFrame(columns=["user_id", "prompt_id", "score", "submitted_at"]),
        "audio": pd.DataFrame(columns=["user_id", "audio_id", "listened", "listened_at"]),
        "content_counts": {"total_vocabulary": 100, "total_audio": 20, "total_prompts": 10, "total_sections": 5},
    }

    features = build_feature_matrix(raw, REF_DATE, label_window_days=7)

    assert "user_id" in features.columns
    assert "is_dropout" in features.columns
    for col in get_feature_columns():
        assert col in features.columns, f"missing feature column: {col}"


def test_build_feature_matrix_excludes_students_created_after_ref_date():
    """
    A student whose account didn't exist on ref_date should not appear in
    the training matrix — they had no past to feature-engineer over.
    """
    # Student 1: created before ref_date (valid)
    # Student 2: created after ref_date (should be excluded)
    students = pd.DataFrame([
        {
            "user_id": 1, "name": "A", "email": "a@x.com",
            "account_created": REF_DATE - timedelta(days=30),
            "trial_ends_at": REF_DATE - timedelta(days=27),
            "subscription_status": "active", "stripe_price_id": "price_monthly",
            "subscription_expires": REF_DATE + timedelta(days=30),
            "cancel_at_period_end": False,
        },
        {
            "user_id": 2, "name": "B", "email": "b@x.com",
            "account_created": REF_DATE + timedelta(days=1),   # future
            "trial_ends_at": REF_DATE + timedelta(days=4),
            "subscription_status": None, "stripe_price_id": None,
            "subscription_expires": None, "cancel_at_period_end": None,
        },
    ])
    raw = {
        "students": students,
        "streaks": pd.DataFrame(columns=["user_id", "activity_date"]),
        "quizzes": pd.DataFrame(columns=["user_id", "section_id", "mode", "score", "total_questions", "completed_at"]),
        "words": pd.DataFrame(columns=["user_id", "vocabulary_id", "status", "attempts", "correct_attempts", "last_seen"]),
        "writing": pd.DataFrame(columns=["user_id", "prompt_id", "score", "submitted_at"]),
        "audio": pd.DataFrame(columns=["user_id", "audio_id", "listened", "listened_at"]),
        "content_counts": {"total_vocabulary": 100, "total_audio": 20, "total_prompts": 10, "total_sections": 5},
    }

    features = build_feature_matrix(raw, REF_DATE)
    assert set(features["user_id"]) == {1}, (
        f"student 2 was created after ref_date and should be excluded; got {set(features['user_id'])}"
    )


def test_label_in_feature_matrix_is_zero_or_one():
    """The label column must be binary."""
    students = _one_student()
    raw = {
        "students": students,
        "streaks": _streaks(days_ago_list=[5, 10]),
        "quizzes": pd.DataFrame(columns=["user_id", "section_id", "mode", "score", "total_questions", "completed_at"]),
        "words": pd.DataFrame(columns=["user_id", "vocabulary_id", "status", "attempts", "correct_attempts", "last_seen"]),
        "writing": pd.DataFrame(columns=["user_id", "prompt_id", "score", "submitted_at"]),
        "audio": pd.DataFrame(columns=["user_id", "audio_id", "listened", "listened_at"]),
        "content_counts": {"total_vocabulary": 100, "total_audio": 20, "total_prompts": 10, "total_sections": 5},
    }
    features = build_feature_matrix(raw, REF_DATE)
    assert set(features["is_dropout"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# ACTIVITY TREND — the single most valuable engineered feature
# ---------------------------------------------------------------------------

def test_activity_trend_is_zero_when_no_recent_activity():
    students = _one_student()
    # Only very old activity — nothing in last 30 days.
    streaks = _streaks(days_ago_list=[60, 50])
    features = build_activity_features(students, streaks, REF_DATE)
    assert features.iloc[0]["activity_trend"] == 0.0


def test_activity_trend_equals_one_when_all_activity_is_recent():
    students = _one_student()
    # All activity is in the last 7 days AND last 30 days → ratio = 1.0.
    streaks = _streaks(days_ago_list=[1, 2, 3, 4, 5])
    features = build_activity_features(students, streaks, REF_DATE)
    assert features.iloc[0]["activity_trend"] == pytest.approx(1.0)
