# TaalMaster ML — Student Dropout Prediction

Predicts which TaalMaster students are at risk of dropping out, why, and what to do about it.

## The ML Strategy

### 1. Define the Problem

**Question:** Which students will stop using TaalMaster?

**Label:** A student inactive for 7+ days = "dropout" (configurable in `config.py`).
This is a **binary classification** problem: each student is either `active (0)` or `dropout (1)`.

### 2. Extract Data (`data_extraction.py`)

We pull 6 tables from the TaalMaster Neon PostgreSQL database:
- `users` + `subscriptions` — who they are, are they paying
- `study_streaks` — which days they logged in
- `quiz_attempts` — practice frequency and scores
- `word_progress` — vocabulary mastery depth
- `writing_submissions` — highest-effort activity
- `audio_progress` — passive engagement

### 3. Engineer Features (`feature_engineering.py`)

**This is where 80% of ML value lives.**

Raw database rows are meaningless to a model. We transform them into 32 **signals**:

| Group | # | Key Insight |
|-------|---|-------------|
| Activity | 9 | `activity_trend` (recent vs. old ratio) catches students **fading** before they fully stop |
| Quiz | 7 | `quiz_score_trend` — declining scores often predict dropout before activity drops (frustration → quit) |
| Vocabulary | 6 | `vocab_coverage` — what % of the platform they've explored |
| Writing | 3 | Writing is the highest-effort activity — students who still write are committed |
| Audio | 2 | Low-effort passive engagement signal |
| Account | 5 | `is_trial_expired` without subscription = natural churn point |

### 4. Train the Model (`train.py`)

**Algorithm:** XGBoost (gradient-boosted decision trees)

**Why XGBoost?**
- Best-in-class for tabular data at this scale
- Handles missing values natively
- Feature importances are interpretable ("this student is at risk **because**...")
- Fast to train and predict

**Validation:** 5-fold stratified cross-validation
- Split students into 5 groups
- Train on 4 groups, test on the 5th
- Repeat 5x so every student is tested once
- Gives honest performance estimate on unseen data

**Experiment Tracking:** MLflow logs every run (parameters, metrics, model artifacts).

### 5. Predict & Explain (`predict.py`)

For each student, the model outputs:
- **Probability** (0.0–1.0) of dropout
- **Risk level**: high (>0.7), medium (0.4–0.7), low (<0.4)
- **Top risk factors**: which features deviate from "healthy" thresholds, weighted by feature importance
- **Recommendation**: actionable admin guidance based on the top risk factor

### 6. Serve via API (`api.py`)

FastAPI exposes predictions for the TaalMaster admin dashboard:
- `GET /predictions` — all students ranked by risk
- `GET /predictions/{user_id}` — single student
- `GET /summary` — aggregate risk overview
- `GET /docs` — interactive Swagger documentation

## Quick Start

```bash
# 1. Setup
cp .env.example .env          # add your Neon DATABASE_URL
pip install -r requirements.txt

# 2. Seed test data (if needed)
python seed_students.py --count 2500

# 3. Train the model
python train.py                # live DB data
python train.py --synthetic 500  # or synthetic data

# 4. Start the API
uvicorn api:app --port 8000 --reload

# 5. Test it
curl http://localhost:8000/summary
curl http://localhost:8000/predictions?risk_level=high&limit=5

# 6. View experiment history
mlflow ui
```

## Project Structure

```
taalmaster-ml/
├── config.py               # DB connection, model paths, dropout threshold
├── data_extraction.py       # Step 1: SQL queries → raw DataFrames
├── feature_engineering.py   # Step 2: raw data → 32 ML features
├── train.py                 # Step 3: XGBoost training + MLflow tracking
├── predict.py               # Step 4: risk scoring + explainability
├── api.py                   # Step 5: FastAPI serving layer
├── seed_students.py         # Utility: seed realistic test data
├── requirements.txt
├── artifacts/               # Trained model files (git-ignored)
└── mlruns/                  # MLflow experiment logs (git-ignored)
```

## Connecting to TaalMaster Admin Dashboard

The TaalMaster Express server proxies requests from the React admin panel:

```
React (Dropout Risk tab) → Express /api/ml/* → FastAPI :8000/*
```

Add `ML_API_URL=http://localhost:8000` to the TaalMaster server `.env`,
create the proxy route in Express, and add the dashboard component in React.
