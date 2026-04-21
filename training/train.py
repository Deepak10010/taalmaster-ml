"""
STEP 3 — Model Training
=========================
Train an XGBoost classifier to predict student dropout.

WHY XGBOOST?
  - Handles tabular data better than neural networks for this size
  - Built-in handling of missing values
  - Feature importance is interpretable (you can explain WHY a student is at risk)
  - Fast to train, fast to predict

WHAT THIS SCRIPT DOES:
  1. Extract data from Neon DB (or generate synthetic data for testing)
  2. Engineer 32 features from raw engagement data
  3. Train XGBoost with 5-fold cross-validation
  4. Log everything to MLflow (parameters, metrics, model artifacts)
  5. Save the trained model to artifacts/

WHAT IS CROSS-VALIDATION?
  We split 2500 students into 5 groups. Train on 4 groups, test on the 5th.
  Repeat 5 times so every student gets tested once. This tells us how the
  model will perform on students it's never seen before.

WHAT IS MLflow?
  An experiment journal. Every training run logs:
    - Parameters: what settings we used
    - Metrics: how accurate the model was
    - Artifacts: the actual model file
  So you can compare runs and reproduce any result.

Usage:
    python train.py                    # Train on live Neon DB data
    python train.py --synthetic 500    # Train on synthetic data (for testing)
"""
import argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    MODEL_PATH, FEATURE_COLS_PATH, DROPOUT_DAYS_THRESHOLD,
)
from transformation.feature_engineering import build_feature_matrix, get_feature_columns


def generate_synthetic_data(n_students: int = 200) -> dict:
    """
    Generate realistic synthetic student data for model development.

    WHY SYNTHETIC DATA?
    When you're building the pipeline, you don't want to wait for real users.
    Synthetic data lets you:
      - Test the pipeline end-to-end
      - Verify the model can distinguish active vs. dropout patterns
      - Develop without needing database access

    The synthetic profiles mimic real behavior:
      - Active students: frequent activity, good scores, subscriptions
      - At-risk students: declining activity, mediocre scores
      - Dropout students: old activity, low scores, no subscription
    """
    rng = np.random.default_rng(42)
    now = datetime.utcnow()

    n_active = int(n_students * 0.4)
    n_at_risk = int(n_students * 0.3)
    n_dropout = n_students - n_active - n_at_risk

    user_ids = list(range(1, n_students + 1))
    students_records, streaks_records, quiz_records = [], [], []
    word_records, writing_records, audio_records = [], [], []

    for i, uid in enumerate(user_ids):
        if i < n_active:
            profile = "active"
        elif i < n_active + n_at_risk:
            profile = "at_risk"
        else:
            profile = "dropout"

        account_age = rng.integers(7, 91)
        created = now - timedelta(days=int(account_age))

        if profile == "active":
            sub_status = rng.choice(["active", "trialing"], p=[0.7, 0.3])
            cancel = False
        elif profile == "at_risk":
            sub_status = rng.choice(["active", "trialing", None], p=[0.3, 0.2, 0.5])
            cancel = rng.random() < 0.4
        else:
            sub_status = rng.choice([None, "canceled", "past_due"], p=[0.5, 0.3, 0.2])
            cancel = rng.random() < 0.6

        students_records.append({
            "user_id": uid, "name": f"Student {uid}", "email": f"s{uid}@test.com",
            "account_created": created, "trial_ends_at": created + timedelta(days=3),
            "subscription_status": sub_status,
            "stripe_price_id": rng.choice(["price_monthly", "price_6month", "price_yearly"]) if sub_status else None,
            "subscription_expires": (now + timedelta(days=int(rng.integers(1, 60)))) if sub_status in ["active", "trialing"] else None,
            "cancel_at_period_end": cancel if sub_status else None,
        })

        # Study streaks — active students have recent days, dropouts have old days
        if profile == "active":
            n_days = rng.integers(int(account_age * 0.5), account_age + 1)
            recent_bias = 0.7
        elif profile == "at_risk":
            n_days = rng.integers(int(account_age * 0.15), int(account_age * 0.5) + 1)
            recent_bias = 0.3
        else:
            n_days = rng.integers(1, max(int(account_age * 0.2), 2))
            recent_bias = 0.05

        possible_dates = [created.date() + timedelta(days=d) for d in range(account_age)]
        weights = np.array([recent_bias + (1 - recent_bias) * (j / len(possible_dates)) for j in range(len(possible_dates))])
        weights /= weights.sum()
        chosen = rng.choice(len(possible_dates), size=min(n_days, len(possible_dates)), replace=False, p=weights)
        for idx in chosen:
            streaks_records.append({"user_id": uid, "activity_date": possible_dates[idx]})

        # Quiz attempts
        if profile == "active":
            n_q, base = rng.integers(10, 40), 0.7
        elif profile == "at_risk":
            n_q, base = rng.integers(3, 15), 0.5
        else:
            n_q, base = rng.integers(0, 6), 0.35

        for _ in range(n_q):
            total_q = rng.integers(5, 21)
            score = min(total_q, max(0, int(total_q * (base + rng.normal(0, 0.15)))))
            quiz_records.append({
                "user_id": uid, "section_id": rng.integers(1, 16),
                "mode": rng.choice(["flashcard", "multiple_choice", "fill_blank", "translation"]),
                "score": score, "total_questions": total_q,
                "completed_at": created + timedelta(days=int(rng.integers(0, account_age))),
            })

        # Word progress
        if profile == "active":
            n_w, mastery_p = rng.integers(30, 100), 0.5
        elif profile == "at_risk":
            n_w, mastery_p = rng.integers(10, 40), 0.2
        else:
            n_w, mastery_p = rng.integers(0, 15), 0.1

        for _ in range(n_w):
            att = rng.integers(1, 15)
            corr = rng.integers(0, att + 1)
            status = "mastered" if (corr / max(att, 1) >= 0.8 and att >= 3 and rng.random() < mastery_p) else "learning" if att > 0 else "new"
            word_records.append({
                "user_id": uid, "vocabulary_id": rng.integers(1, 255),
                "status": status, "attempts": att, "correct_attempts": corr,
                "last_seen": created + timedelta(days=int(rng.integers(0, account_age))),
            })

        # Writing
        if profile == "active":
            n_wr, base_ws = rng.integers(3, 12), 7
        elif profile == "at_risk":
            n_wr, base_ws = rng.integers(0, 4), 5
        else:
            n_wr, base_ws = rng.integers(0, 2), 4

        for _ in range(n_wr):
            writing_records.append({
                "user_id": uid, "prompt_id": rng.integers(1, 31),
                "score": min(10, max(1, int(base_ws + rng.normal(0, 1.5)))),
                "submitted_at": created + timedelta(days=int(rng.integers(0, account_age))),
            })

        # Audio
        n_a = rng.integers(5, 20) if profile == "active" else rng.integers(1, 8) if profile == "at_risk" else rng.integers(0, 3)
        for _ in range(n_a):
            audio_records.append({
                "user_id": uid, "audio_id": rng.integers(1, 32),
                "listened": True, "listened_at": created + timedelta(days=int(rng.integers(0, account_age))),
            })

    return {
        "students": pd.DataFrame(students_records),
        "streaks": pd.DataFrame(streaks_records),
        "quizzes": pd.DataFrame(quiz_records),
        "words": pd.DataFrame(word_records),
        "writing": pd.DataFrame(writing_records),
        "audio": pd.DataFrame(audio_records),
        "content_counts": {"total_vocabulary": 254, "total_audio": 31, "total_prompts": 30, "total_sections": 15},
    }


def train_model(features: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train XGBoost with stratified 5-fold CV and log to MLflow.

    KEY PARAMETERS EXPLAINED:
      n_estimators=200    → Build 200 decision trees (more = better but slower)
      max_depth=5         → Each tree can be 5 levels deep (prevents overfitting)
      learning_rate=0.05  → Each tree only contributes a little (prevents overfitting)
      scale_pos_weight    → Compensates for class imbalance (if 30% dropout, 70% active)
      subsample=0.8       → Each tree only sees 80% of data (prevents overfitting)

    METRICS WE TRACK:
      Accuracy   → % of all predictions that are correct
      Precision  → of students we flagged as "dropout", how many really are?
      Recall     → of actual dropouts, how many did we catch?
      F1         → harmonic mean of precision & recall (our primary metric)
      ROC AUC    → overall ranking quality (1.0 = perfect, 0.5 = random guess)
    """
    X = features[feature_cols].values
    y = features["is_dropout"].values

    # Handle class imbalance: if 70% active / 30% dropout,
    # scale_pos_weight = 70/30 ≈ 2.3, so the model pays more attention to dropouts
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"xgb-dropout-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params(params)
        mlflow.log_param("n_students", len(features))
        mlflow.log_param("n_dropout", int(n_pos))
        mlflow.log_param("n_active", int(n_neg))
        mlflow.log_param("dropout_threshold_days", DROPOUT_DAYS_THRESHOLD)
        mlflow.log_param("n_features", len(feature_cols))

        # ── 5-FOLD CROSS-VALIDATION ──
        # Split data 5 ways. Train on 4, test on 1. Repeat 5 times.
        # "Stratified" means each fold keeps the same active/dropout ratio.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = XGBClassifier(**params)
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)[:, 1]

            cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            cv_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            cv_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            cv_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            cv_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.0)

        for metric, values in cv_metrics.items():
            mlflow.log_metric(f"cv_{metric}_mean", np.mean(values))
            mlflow.log_metric(f"cv_{metric}_std", np.std(values))

        # ── FINAL MODEL ── Train on ALL data for deployment
        final_model = XGBClassifier(**params)
        final_model.fit(X, y, verbose=False)

        y_final_pred = final_model.predict(X)
        y_final_proba = final_model.predict_proba(X)[:, 1]

        final_metrics = {
            "train_accuracy": accuracy_score(y, y_final_pred),
            "train_precision": precision_score(y, y_final_pred, zero_division=0),
            "train_recall": recall_score(y, y_final_pred, zero_division=0),
            "train_f1": f1_score(y, y_final_pred, zero_division=0),
            "train_roc_auc": roc_auc_score(y, y_final_proba) if len(np.unique(y)) > 1 else 0.0,
        }
        mlflow.log_metrics(final_metrics)

        # ── FEATURE IMPORTANCES ──
        # Which features does the model rely on most?
        # This is KEY for explainability: "why is this student at risk?"
        importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        mlflow.log_dict(sorted_importance, "feature_importances.json")

        report = classification_report(y, y_final_pred, target_names=["Active", "Dropout"])
        mlflow.log_text(report, "classification_report.txt")

        cm = confusion_matrix(y, y_final_pred)
        mlflow.log_dict({"matrix": cm.tolist(), "labels": ["Active", "Dropout"]}, "confusion_matrix.json")

        # Save model to disk
        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(feature_cols, FEATURE_COLS_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(FEATURE_COLS_PATH))
        mlflow.xgboost.log_model(final_model, "model")

        run_id = mlflow.active_run().info.run_id

    return {
        "model": final_model,
        "run_id": run_id,
        "cv_metrics": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in cv_metrics.items()},
        "final_metrics": final_metrics,
        "feature_importances": sorted_importance,
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Train dropout prediction model")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Generate N synthetic students instead of using live DB")
    args = parser.parse_args()

    print("=" * 60)
    print("TaalMaster Dropout Prediction — Training Pipeline")
    print("=" * 60)

    if args.synthetic > 0:
        print(f"\n[1/3] Generating {args.synthetic} synthetic students...")
        raw_data = generate_synthetic_data(args.synthetic)
    else:
        print("\n[1/3] Extracting data from Neon DB...")
        from ingestion.data_extraction import extract_all
        raw_data = extract_all()

    print(f"  Students: {len(raw_data['students'])}")
    print(f"  Streak records: {len(raw_data['streaks'])}")
    print(f"  Quiz attempts: {len(raw_data['quizzes'])}")
    print(f"  Word progress: {len(raw_data['words'])}")
    print(f"  Writing submissions: {len(raw_data['writing'])}")
    print(f"  Audio listened: {len(raw_data['audio'])}")

    print("\n[2/3] Engineering features...")
    feature_cols = get_feature_columns()
    features = build_feature_matrix(raw_data)
    print(f"  Feature matrix: {features.shape[0]} students x {len(feature_cols)} features")
    print(f"  Dropout rate: {features['is_dropout'].mean():.1%}")

    print("\n[3/3] Training XGBoost with 5-fold CV + MLflow tracking...")
    results = train_model(features, feature_cols)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMLflow Run ID: {results['run_id']}")
    print(f"\nCross-Validation Metrics:")
    for metric, vals in results["cv_metrics"].items():
        print(f"  {metric:>12s}: {vals['mean']:.4f} (+/- {vals['std']:.4f})")

    print(f"\nTop 10 Feature Importances:")
    for i, (feat, imp) in enumerate(list(results["feature_importances"].items())[:10]):
        print(f"  {i+1:>2d}. {feat:<35s} {imp:.4f}")

    print(f"\n{results['classification_report']}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
