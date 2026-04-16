"""
STEP 2 — Feature Engineering
==============================
Transform raw database rows into meaningful signals the model can learn from.

THIS IS WHERE 80% OF ML VALUE LIVES.

WHY RAW DATA ISN'T ENOUGH:
  A row saying "user_id=5, activity_date=2026-04-10" means nothing to a model.
  But "user 5 has been inactive for 6 days and their activity is declining"
  is a strong dropout signal.

THE 6 FEATURE GROUPS (32 features total):

  1. ACTIVITY (9 features)
     Captures: recency, frequency, streaks, gaps, trend
     Key insight: "days_since_last_activity" is the strongest single predictor,
     but "activity_trend" (recent vs. older activity ratio) catches students
     who are FADING before they fully stop.

  2. QUIZ (7 features)
     Captures: how much they practice, how well they score, are they improving?
     Key insight: a declining score trend often predicts dropout before
     activity drops — frustration leads to quitting.

  3. VOCABULARY (6 features)
     Captures: breadth (how many words tried) and depth (mastery rate)
     Key insight: "vocab_coverage" (% of available words attempted) measures
     how much of the platform they've explored.

  4. WRITING (3 features)
     Captures: submission count, quality, improvement
     Key insight: writing is the highest-effort activity — students who
     still write are usually the most committed.

  5. AUDIO (2 features)
     Captures: listening engagement
     Key insight: passive activity — low effort but still signals engagement.

  6. ACCOUNT (5 features)
     Captures: account age, trial/subscription status
     Key insight: "is_trial_expired" without subscription is a strong
     dropout signal — the paywall is a natural churn point.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import DROPOUT_DAYS_THRESHOLD


def _safe_div(a, b, fill=0.0):
    """Divide without crashing on zero. Returns `fill` when denominator is 0."""
    return np.where(b > 0, a / b, fill)


# ---------------------------------------------------------------------------
# 1. ACTIVITY FEATURES — the most predictive group
# ---------------------------------------------------------------------------
def build_activity_features(students: pd.DataFrame, streaks: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    ref = pd.Timestamp(ref_date)
    user_ids = students[["user_id"]].copy()

    if streaks.empty:
        for col in [
            "days_since_last_activity", "total_active_days",
            "active_days_last_7", "active_days_last_14", "active_days_last_30",
            "longest_streak", "current_streak", "avg_gap_between_sessions",
            "activity_trend",
        ]:
            user_ids[col] = 0.0
        user_ids["days_since_last_activity"] = 999.0
        return user_ids

    streaks = streaks.copy()
    streaks["activity_date"] = pd.to_datetime(streaks["activity_date"])

    agg = streaks.groupby("user_id").agg(
        last_activity=("activity_date", "max"),
        total_active_days=("activity_date", "nunique"),
    ).reset_index()

    # How many days since they last did anything?
    agg["days_since_last_activity"] = (ref - agg["last_activity"]).dt.days

    # Windowed activity counts: how active were they recently?
    for window, col in [(7, "active_days_last_7"), (14, "active_days_last_14"), (30, "active_days_last_30")]:
        cutoff = ref - timedelta(days=window)
        windowed = streaks[streaks["activity_date"] >= cutoff].groupby("user_id")["activity_date"].nunique().reset_index()
        windowed.columns = ["user_id", col]
        agg = agg.merge(windowed, on="user_id", how="left")

    # Streak analysis: consecutive days of practice
    def _streak_stats(group):
        dates = sorted(group["activity_date"].dt.date.unique())
        if len(dates) == 0:
            return pd.Series({"longest_streak": 0, "current_streak": 0, "avg_gap_between_sessions": 0.0})

        streaks_list = []
        current = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days == 1:
                current += 1
            else:
                streaks_list.append(current)
                current = 1
        streaks_list.append(current)

        longest = max(streaks_list)

        # Current streak: how many consecutive days ending at today?
        cur = 0
        check = ref.date()
        for d in reversed(dates):
            if d == check:
                cur += 1
                check -= timedelta(days=1)
            elif d < check:
                break

        gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        avg_gap = np.mean(gaps) if gaps else 0.0

        return pd.Series({
            "longest_streak": longest,
            "current_streak": cur,
            "avg_gap_between_sessions": avg_gap,
        })

    streak_stats = streaks.groupby("user_id").apply(_streak_stats, include_groups=False).reset_index()
    agg = agg.merge(streak_stats, on="user_id", how="left")

    # ACTIVITY TREND: last_7 / last_30
    # This is the single most important feature. It catches students who
    # are FADING — still active, but less and less each week.
    # Ratio > 0.23 = steady (7/30). Ratio < 0.1 = declining fast.
    agg["activity_trend"] = _safe_div(
        agg["active_days_last_7"].fillna(0).values,
        agg["active_days_last_30"].fillna(0).values,
        fill=0.0,
    )

    result = user_ids.merge(agg.drop(columns=["last_activity"]), on="user_id", how="left")
    result["days_since_last_activity"] = result["days_since_last_activity"].fillna(999.0)
    result = result.fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# 2. QUIZ FEATURES — performance + effort
# ---------------------------------------------------------------------------
def build_quiz_features(students: pd.DataFrame, quizzes: pd.DataFrame) -> pd.DataFrame:
    user_ids = students[["user_id"]].copy()

    if quizzes.empty:
        for col in [
            "total_quizzes", "avg_quiz_score_pct", "quiz_score_trend",
            "distinct_quiz_modes", "distinct_sections_quizzed",
            "quizzes_last_7_days", "quizzes_last_14_days",
        ]:
            user_ids[col] = 0.0
        return user_ids

    q = quizzes.copy()
    q["score_pct"] = _safe_div(q["score"].values, q["total_questions"].values)
    q["completed_at"] = pd.to_datetime(q["completed_at"])

    agg = q.groupby("user_id").agg(
        total_quizzes=("score", "count"),
        avg_quiz_score_pct=("score_pct", "mean"),
        distinct_quiz_modes=("mode", "nunique"),
        distinct_sections_quizzed=("section_id", "nunique"),
    ).reset_index()

    # SCORE TREND: is the student improving or declining?
    # Positive slope = getting better. Negative = struggling → may quit.
    def _score_slope(group):
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group), dtype=float)
        y = group.sort_values("completed_at")["score_pct"].values
        return np.polyfit(x, y, 1)[0]

    trends = q.groupby("user_id").apply(_score_slope, include_groups=False).reset_index()
    trends.columns = ["user_id", "quiz_score_trend"]
    agg = agg.merge(trends, on="user_id", how="left")

    now = q["completed_at"].max()
    for window, col in [(7, "quizzes_last_7_days"), (14, "quizzes_last_14_days")]:
        cutoff = now - timedelta(days=window)
        recent = q[q["completed_at"] >= cutoff].groupby("user_id").size().reset_index(name=col)
        agg = agg.merge(recent, on="user_id", how="left")

    result = user_ids.merge(agg, on="user_id", how="left").fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# 3. VOCABULARY FEATURES — breadth and depth of learning
# ---------------------------------------------------------------------------
def build_vocabulary_features(students: pd.DataFrame, words: pd.DataFrame, total_vocab: int) -> pd.DataFrame:
    user_ids = students[["user_id"]].copy()

    if words.empty:
        for col in [
            "words_attempted", "words_mastered", "words_learning",
            "mastery_rate", "avg_accuracy", "vocab_coverage",
        ]:
            user_ids[col] = 0.0
        return user_ids

    w = words.copy()
    agg = w.groupby("user_id").agg(
        words_attempted=("vocabulary_id", "count"),
        words_mastered=("status", lambda x: (x == "mastered").sum()),
        words_learning=("status", lambda x: (x == "learning").sum()),
        total_attempts=("attempts", "sum"),
        total_correct=("correct_attempts", "sum"),
    ).reset_index()

    agg["mastery_rate"] = _safe_div(agg["words_mastered"].values, agg["words_attempted"].values)
    agg["avg_accuracy"] = _safe_div(agg["total_correct"].values, agg["total_attempts"].values)
    # What % of the platform's vocabulary has this student touched?
    agg["vocab_coverage"] = _safe_div(agg["words_attempted"].values, max(total_vocab, 1))

    agg = agg.drop(columns=["total_attempts", "total_correct"])
    result = user_ids.merge(agg, on="user_id", how="left").fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# 4. WRITING FEATURES — highest-effort activity
# ---------------------------------------------------------------------------
def build_writing_features(students: pd.DataFrame, writing: pd.DataFrame) -> pd.DataFrame:
    user_ids = students[["user_id"]].copy()

    if writing.empty:
        for col in ["total_writing_submissions", "avg_writing_score", "writing_score_trend"]:
            user_ids[col] = 0.0
        return user_ids

    wr = writing.copy()
    wr["submitted_at"] = pd.to_datetime(wr["submitted_at"])

    agg = wr.groupby("user_id").agg(
        total_writing_submissions=("score", "count"),
        avg_writing_score=("score", "mean"),
    ).reset_index()

    def _writing_slope(group):
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group), dtype=float)
        y = group.sort_values("submitted_at")["score"].values.astype(float)
        return np.polyfit(x, y, 1)[0]

    trends = wr.groupby("user_id").apply(_writing_slope, include_groups=False).reset_index()
    trends.columns = ["user_id", "writing_score_trend"]
    agg = agg.merge(trends, on="user_id", how="left")

    result = user_ids.merge(agg, on="user_id", how="left").fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# 5. AUDIO FEATURES — passive engagement
# ---------------------------------------------------------------------------
def build_audio_features(students: pd.DataFrame, audio: pd.DataFrame, total_audio: int) -> pd.DataFrame:
    user_ids = students[["user_id"]].copy()

    if audio.empty:
        user_ids["audio_listened_count"] = 0.0
        user_ids["audio_completion_rate"] = 0.0
        return user_ids

    agg = audio.groupby("user_id").agg(
        audio_listened_count=("audio_id", "nunique"),
    ).reset_index()

    agg["audio_completion_rate"] = _safe_div(
        agg["audio_listened_count"].values, max(total_audio, 1)
    )

    result = user_ids.merge(agg, on="user_id", how="left").fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# 6. ACCOUNT FEATURES — subscription is a churn signal
# ---------------------------------------------------------------------------
def build_account_features(students: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    ref = pd.Timestamp(ref_date)
    df = students.copy()

    df["account_created"] = pd.to_datetime(df["account_created"])
    df["trial_ends_at"] = pd.to_datetime(df["trial_ends_at"])
    df["subscription_expires"] = pd.to_datetime(df["subscription_expires"])

    df["account_age_days"] = (ref - df["account_created"]).dt.days

    # Trial expired + no subscription = high churn risk
    df["is_trial_expired"] = (
        (df["trial_ends_at"] < ref) &
        (~df["subscription_status"].isin(["active", "trialing"]))
    ).astype(int)

    df["has_subscription"] = df["subscription_status"].isin(["active", "trialing"]).astype(int)
    df["is_subscription_cancelling"] = df["cancel_at_period_end"].fillna(False).astype(int)
    df["days_until_sub_expires"] = (df["subscription_expires"] - ref).dt.days
    df["days_until_sub_expires"] = df["days_until_sub_expires"].fillna(-1)

    return df[["user_id", "account_age_days", "is_trial_expired",
               "has_subscription", "is_subscription_cancelling", "days_until_sub_expires"]]


# ---------------------------------------------------------------------------
# COMBINE — merge all feature groups + create the label
# ---------------------------------------------------------------------------
def build_feature_matrix(raw_data: dict, ref_date: datetime = None) -> pd.DataFrame:
    """
    Build the complete feature matrix.

    Each row = one student.
    Each column = one engineered feature.
    Last column = is_dropout (the label we're predicting).
    """
    if ref_date is None:
        ref_date = datetime.utcnow()

    students = raw_data["students"]
    counts = raw_data["content_counts"]

    activity = build_activity_features(students, raw_data["streaks"], ref_date)
    quiz = build_quiz_features(students, raw_data["quizzes"])
    vocab = build_vocabulary_features(students, raw_data["words"], counts["total_vocabulary"])
    writing = build_writing_features(students, raw_data["writing"])
    audio = build_audio_features(students, raw_data["audio"], counts["total_audio"])
    account = build_account_features(students, ref_date)

    features = activity
    for df in [quiz, vocab, writing, audio, account]:
        features = features.merge(df, on="user_id", how="left")

    # THE LABEL: inactive 7+ days = dropout (1), otherwise active (0)
    features["is_dropout"] = (features["days_since_last_activity"] >= DROPOUT_DAYS_THRESHOLD).astype(int)

    return features


def get_feature_columns() -> list:
    """The 32 features the model uses (excludes user_id and label)."""
    return [
        # Activity (9)
        "days_since_last_activity", "total_active_days",
        "active_days_last_7", "active_days_last_14", "active_days_last_30",
        "longest_streak", "current_streak", "avg_gap_between_sessions", "activity_trend",
        # Quiz (7)
        "total_quizzes", "avg_quiz_score_pct", "quiz_score_trend",
        "distinct_quiz_modes", "distinct_sections_quizzed",
        "quizzes_last_7_days", "quizzes_last_14_days",
        # Vocabulary (6)
        "words_attempted", "words_mastered", "words_learning",
        "mastery_rate", "avg_accuracy", "vocab_coverage",
        # Writing (3)
        "total_writing_submissions", "avg_writing_score", "writing_score_trend",
        # Audio (2)
        "audio_listened_count", "audio_completion_rate",
        # Account (5)
        "account_age_days", "is_trial_expired", "has_subscription",
        "is_subscription_cancelling", "days_until_sub_expires",
    ]
