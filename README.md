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
python -m ingestion.seed_students --count 2500

# 3. Run the full pipeline (extract → features → train → register → predict)
python pipeline.py

# — OR — run individual stages:
python -m training.train                # train only
python -m training.compare_models       # LR vs RF vs XGB comparison
python -m evaluation.backtest           # temporal holdout check

# 4. Start the API
uvicorn serving.api:app --port 8000 --reload

# 5. Test it
curl http://localhost:8000/summary
curl http://localhost:8000/predictions?risk_level=high&limit=5

# 6. View experiment history
mlflow ui
```

## DVC (data + pipeline versioning)

The repo is DVC-enabled. The pipeline (`extract → features → train`) is
declared in `dvc.yaml` with parameters in `params.yaml`.

```bash
# View the stage DAG
dvc dag

# Run the whole pipeline — only re-runs what actually changed
dvc repro

# See current metrics
dvc metrics show

# Compare metrics to the last committed run
dvc metrics diff

# Push tracked data/models to the remote
dvc push

# Pull them back on another machine (after git clone + pip install)
dvc pull
```

**Two ways to run the pipeline, pick the right one for the task:**

- `python pipeline.py` — ad-hoc runs, timestamped snapshots under `data/raw/`, MLflow tracking + registry promotion.
- `dvc repro` — reproducible runs with stable outputs at `data/dvc/`, metric diffing across runs, smart caching (skips unchanged stages).

Both write models to `artifacts/` and log experiments to `mlruns/`. Pick
`pipeline.py` when you just want to retrain; pick `dvc repro` when you're
iterating on features or hyperparams and want to compare runs.

## Project Structure

```
taalmaster-ml/
├── config.py                     # DB connection, model paths, dropout threshold
├── pipeline.py                   # End-to-end orchestrator (extract → serve)
├── run_scheduled.bat             # Windows Task Scheduler wrapper
├── requirements.txt
├── ingestion/                    # Step 1 — pull raw data from the DB
│   ├── data_extraction.py
│   └── seed_students.py          # Utility: seed realistic test data
├── transformation/               # Step 2 — raw rows → 32 ML features
│   └── feature_engineering.py
├── training/                     # Step 3 — fit models, log to MLflow
│   ├── train.py                  # XGBoost + 5-fold CV
│   └── compare_models.py         # LR vs RF vs XGB side-by-side
├── evaluation/                   # Step 3.5 — is the model actually good?
│   └── backtest.py               # Temporal holdout validation
├── inference/                    # Step 4 — score students, explain risk
│   └── predict.py
├── serving/                      # Step 5 — expose predictions over HTTP
│   └── api.py                    # FastAPI endpoints
├── artifacts/                    # Trained model files (git-ignored)
├── data/                         # Intermediate outputs (git-ignored)
│   ├── raw/                      # Timestamped DB snapshots
│   ├── features/                 # Engineered feature matrices
│   └── predictions/              # Scored students per run
├── logs/                         # Scheduled-task logs (git-ignored)
└── mlruns/                       # MLflow experiment logs (git-ignored)
```

## Connecting to TaalMaster Admin Dashboard

The TaalMaster Express server proxies requests from the React admin panel:

```
React (Dropout Risk tab) → Express /api/ml/* → FastAPI :8000/*
```

Add `ML_API_URL=http://localhost:8000` to the TaalMaster server `.env`,
create the proxy route in Express, and add the dashboard component in React.
