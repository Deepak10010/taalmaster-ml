"""
STEP 4 — Prediction
=====================
Load the trained model and generate dropout risk scores.

THIS IS WHERE ML BECOMES ACTIONABLE:
  The model outputs a probability (0.0 to 1.0) per student.
  We translate that into:
    - Risk level: high (>0.7), medium (0.4-0.7), low (<0.4)
    - Top risk factors: WHY this student is at risk
    - Recommendation: WHAT the admin should do about it

WHY EXPLAINABILITY MATTERS:
  A probability alone isn't useful. "Student X has 87% dropout risk" raises
  the question "why?" and "what do I do?". We answer both by:
    1. Checking which features deviate from "healthy" thresholds
    2. Weighting them by the model's feature importances
    3. Generating a specific recommendation based on the top risk factor
"""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from config import MODEL_PATH, FEATURE_COLS_PATH
from feature_engineering import build_feature_matrix, get_feature_columns


def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


def predict_all_students(raw_data: dict) -> list[dict]:
    """
    Generate dropout risk predictions for all students.
    Returns a list sorted by risk (highest first).
    """
    model, feature_cols = load_model()
    ref_date = datetime.utcnow()
    features = build_feature_matrix(raw_data, ref_date)

    X = features[feature_cols].values
    probabilities = model.predict_proba(X)[:, 1]

    importances = dict(zip(feature_cols, model.feature_importances_))

    results = []
    for idx, row in features.iterrows():
        prob = float(probabilities[idx])

        if prob >= 0.7:
            risk_level = "high"
        elif prob >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        top_factors = _get_risk_factors(row, feature_cols, importances)
        recommendation = _generate_recommendation(row, top_factors, risk_level)

        student_info = raw_data["students"][raw_data["students"]["user_id"] == row["user_id"]].iloc[0]

        results.append({
            "user_id": int(row["user_id"]),
            "name": student_info["name"],
            "email": student_info["email"],
            "dropout_probability": round(prob, 4),
            "risk_level": risk_level,
            "top_risk_factors": top_factors[:5],
            "days_since_last_activity": int(row["days_since_last_activity"]),
            "total_quizzes": int(row["total_quizzes"]),
            "words_mastered": int(row["words_mastered"]),
            "current_streak": int(row["current_streak"]),
            "has_subscription": bool(row["has_subscription"]),
            "recommendation": recommendation,
        })

    results.sort(key=lambda x: x["dropout_probability"], reverse=True)
    return results


def predict_single_student(raw_data: dict, user_id: int) -> dict | None:
    all_preds = predict_all_students(raw_data)
    for pred in all_preds:
        if pred["user_id"] == user_id:
            return pred
    return None


def _get_risk_factors(row: pd.Series, feature_cols: list, importances: dict) -> list[dict]:
    """
    For each student, identify which features are "unhealthy" and how much
    the model cares about that feature (importance weighting).

    STRATEGY: compare each feature value against a "healthy" threshold.
    If below threshold AND the model considers this feature important,
    it's a risk factor.
    """
    healthy_thresholds = {
        "days_since_last_activity": {"threshold": 3, "direction": "lower_is_better"},
        "active_days_last_7": {"threshold": 3, "direction": "higher_is_better"},
        "active_days_last_14": {"threshold": 5, "direction": "higher_is_better"},
        "current_streak": {"threshold": 3, "direction": "higher_is_better"},
        "total_quizzes": {"threshold": 5, "direction": "higher_is_better"},
        "avg_quiz_score_pct": {"threshold": 0.6, "direction": "higher_is_better"},
        "words_mastered": {"threshold": 10, "direction": "higher_is_better"},
        "mastery_rate": {"threshold": 0.3, "direction": "higher_is_better"},
        "total_writing_submissions": {"threshold": 2, "direction": "higher_is_better"},
        "has_subscription": {"threshold": 1, "direction": "higher_is_better"},
        "is_trial_expired": {"threshold": 0, "direction": "lower_is_better"},
        "avg_gap_between_sessions": {"threshold": 3, "direction": "lower_is_better"},
        "activity_trend": {"threshold": 0.3, "direction": "higher_is_better"},
    }

    risk_factors = []
    for feat in feature_cols:
        if feat not in healthy_thresholds:
            continue

        info = healthy_thresholds[feat]
        value = row[feat]
        threshold = info["threshold"]
        imp = importances.get(feat, 0)

        if info["direction"] == "higher_is_better":
            if value < threshold:
                severity = (threshold - value) / max(threshold, 1) * imp
                risk_factors.append({
                    "feature": feat, "value": round(float(value), 2),
                    "threshold": threshold, "severity": round(float(severity), 4),
                    "message": _feature_message(feat, value),
                })
        else:
            if value > threshold:
                severity = (value - threshold) / max(threshold, 1) * imp
                risk_factors.append({
                    "feature": feat, "value": round(float(value), 2),
                    "threshold": threshold, "severity": round(float(severity), 4),
                    "message": _feature_message(feat, value),
                })

    risk_factors.sort(key=lambda x: x["severity"], reverse=True)
    return risk_factors


def _feature_message(feat: str, value: float) -> str:
    messages = {
        "days_since_last_activity": f"Inactive for {int(value)} days",
        "active_days_last_7": f"Only {int(value)} active days in the last week",
        "active_days_last_14": f"Only {int(value)} active days in the last 2 weeks",
        "current_streak": f"Current streak is only {int(value)} days",
        "total_quizzes": f"Only completed {int(value)} quizzes total",
        "avg_quiz_score_pct": f"Average quiz score is {value:.0%}",
        "words_mastered": f"Only {int(value)} words mastered",
        "mastery_rate": f"Mastery rate is only {value:.0%}",
        "total_writing_submissions": f"Only {int(value)} writing submissions",
        "has_subscription": "No active subscription",
        "is_trial_expired": "Trial has expired without subscribing",
        "avg_gap_between_sessions": f"Average {value:.1f} days between sessions",
        "activity_trend": f"Recent activity trend is declining ({value:.0%})",
    }
    return messages.get(feat, f"{feat}: {value}")


def _generate_recommendation(row: pd.Series, risk_factors: list, risk_level: str) -> str:
    if risk_level == "low":
        return "Student is engaged. Keep up current pace."

    if not risk_factors:
        return "Monitor student activity over the next week."

    top = risk_factors[0]["feature"]

    recommendations = {
        "days_since_last_activity": "Send a re-engagement email with new content highlights.",
        "active_days_last_7": "Suggest a short daily quiz challenge to rebuild habit.",
        "current_streak": "Encourage starting a new streak with a small goal (3 days).",
        "total_quizzes": "Recommend trying different quiz modes to find their preferred style.",
        "avg_quiz_score_pct": "Suggest reviewing vocabulary before quizzing. Consider easier sections.",
        "words_mastered": "Focus on flashcard mode to build core vocabulary first.",
        "has_subscription": "Consider offering a discount or extended trial.",
        "is_trial_expired": "Send a trial extension offer or highlight free Thema 1 content.",
        "avg_gap_between_sessions": "Set up a reminder notification for daily practice.",
        "activity_trend": "Check if the student is stuck on a difficult section.",
    }

    rec = recommendations.get(top, "Review student's learning path and provide personalized guidance.")
    if risk_level == "high":
        rec = "URGENT: " + rec
    return rec
