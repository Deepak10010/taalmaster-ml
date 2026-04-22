# TaalMaster ML — Complete Guide

This is the full reference for the `taalmaster-ml` repo. It covers what this project
does, how every stage of the pipeline works, the ML concepts behind it, how MLflow
and DVC fit in, how to run every piece, and lessons learned along the way.

If you want the 3-minute overview, read [README.md](README.md). If you want to
understand why everything is the way it is, read this.

If you've never seen an ML project before, skim the [In Plain English](#in-plain-english)
section below first — it explains what the whole thing does in everyday language.

---

## In Plain English

Before diving into the technical sections, here is what the project actually does,
with no ML jargon, the way you'd explain it to a family member over dinner.

### The problem

TaalMaster is a Dutch-learning app. Some students use it for a while and then
quietly disappear — they stop opening the app, stop doing quizzes, and eventually
cancel. That's bad for the business and sad for the student (they came to learn
Dutch and didn't).

**This project tries to spot the students who are about to disappear, while
there's still time to do something about it.**

Like a weather forecast. We can't be certain it'll rain, but if the signs are
bad enough, you grab an umbrella.

### How it works, imagined as a friendly assistant

Picture an assistant at a desk every morning at 10am:

1. **Reads the app's diary.** Who logged in? Who did quizzes? Who submitted
   writing? Who has been quiet lately?

2. **Writes a report card** for each student — 32 lines like "hasn't logged in
   for 8 days," "quiz scores are dropping," "trial expired last week but didn't
   subscribe."

3. **Learns patterns from the past.** Shows a computer every student who quit
   in the past, plus everyone who stuck around, and asks: "what did the quitters
   look like a week before they quit?" The computer figures out the pattern.

4. **Scores today's students** against that pattern. Gives each one a number
   from 0% ("fine") to 100% ("about to leave").

5. **Tells the admin dashboard.** Shows a ranked list — the students most at
   risk, a short explanation of why the computer thinks so, and a suggested
   action ("send this student a re-engagement email").

That's the whole job.

### Why all the extra tools exist

Everything else in this repo is a **support system** wrapped around that
simple idea:

| Tool in this repo | What it actually is | Why we need it |
|---|---|---|
| **MLflow** | A filing cabinet for predictors | Keeps every version we've made so we know which is currently serving and which worked best |
| **DVC** | A safety deposit box for data + models | Lets us go back in time to exactly reproduce any past result |
| **Windows Task Scheduler** | An alarm clock | Runs the assistant every morning without anyone pushing a button |
| **Drift detection** | A smoke detector | Notices when student behavior suddenly changes in weird ways, so we know it's time to retrain |
| **FastAPI + Express proxy** | A waiter | Carries the assistant's report from the computer to the admin's browser |
| **Temporal split** | A fair-testing rule | Makes sure the assistant is genuinely *predicting* the future, not just *reading* the present |

### The one-sentence version

**This project predicts which students are about to give up on the app,
explains why, and tells admins what to do about it. The rest of the repo
is the plumbing that keeps it running reliably and reproducibly.**

### Why this was interesting to build

A naive version of this is trivial — just query the database for students
inactive for 7 days, list them. No ML needed.

The actually useful version catches people *before* they go silent, by
spotting a combination of subtle signs (quiz scores drifting down, skipping
writing, slowing engagement) that a human would probably miss. That's what
the model learns to see, and that's what makes it worth the effort.

---

## Table of Contents

1. [What This Repo Does](#1-what-this-repo-does)
2. [The Business Problem](#2-the-business-problem)
3. [Architecture Overview](#3-architecture-overview)
4. [Folder Structure](#4-folder-structure)
5. [Environment Setup](#5-environment-setup)
6. [Stage 1 — Ingestion](#6-stage-1--ingestion)
7. [Stage 2 — Feature Engineering](#7-stage-2--feature-engineering)
8. [The Temporal Split (most important concept)](#8-the-temporal-split-most-important-concept)
9. [Stage 3 — Training](#9-stage-3--training)
10. [Stage 3.5 — Evaluation & Backtest](#10-stage-35--evaluation--backtest)
11. [Stage 4 — Prediction & Explainability](#11-stage-4--prediction--explainability)
12. [Stage 5 — Serving (FastAPI)](#12-stage-5--serving-fastapi)
13. [MLflow Deep Dive](#13-mlflow-deep-dive)
14. [DVC Deep Dive](#14-dvc-deep-dive)
15. [Orchestration — pipeline.py](#15-orchestration--pipelinepy)
16. [Scheduling — Windows Task Scheduler](#16-scheduling--windows-task-scheduler)
17. [How to Run Everything (command reference)](#17-how-to-run-everything-command-reference)
18. [Operational Playbook](#18-operational-playbook)
19. [Troubleshooting](#19-troubleshooting)
20. [Lessons Learned](#20-lessons-learned)
21. [Admin Dashboard Integration (Taalmaster)](#21-admin-dashboard-integration-taalmaster)
22. [Continuous Integration (GitHub Actions)](#21b-continuous-integration-github-actions)
23. [What's Not Done Yet](#22-whats-not-done-yet)

---

## 1. What This Repo Does

`taalmaster-ml` is a standalone machine-learning service that predicts which
TaalMaster students are about to stop using the app. It connects to the main
TaalMaster PostgreSQL database (hosted on Neon), engineers behavioral features
per student, trains an XGBoost model, tracks every run in MLflow, versions all
data and pipeline stages in DVC, and serves predictions over HTTP via FastAPI.

The main TaalMaster admin dashboard (a React app served from an Express API
at `c:\Users\dlokanath\taalmaster\`) can consume these predictions to flag
at-risk students to admins before they quit.

**What makes a student a "dropout"?** A student who has had no activity for 7
consecutive days. That definition is configurable in [config.py](config.py).
The ML task is binary classification: for each student, output a probability
they'll go silent in the next 7 days.

---

## 2. The Business Problem

TaalMaster is a language-learning app. Students log in, do quizzes, master
vocabulary, submit writing, listen to audio. Some students stick around;
others stop opening the app and are eventually churn.

Admins want to know **which students are at risk of quitting** so they can
intervene — a re-engagement email, a trial extension, a nudge to try a new
feature — before they're lost.

This is a classic churn prediction task. Two things make it interesting:

1. **The signal is multi-modal.** Activity streaks, quiz scores, vocabulary
   breadth, writing submissions, subscription status — each tells part of
   the story.
2. **Timing matters.** Flagging someone after they've already been silent
   for two weeks is useless. Flagging them when they're just starting to
   fade is valuable.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Neon Postgres (production DB)                                   │
│  users, subscriptions, study_streaks, quiz_attempts,             │
│  word_progress, writing_submissions, audio_progress              │
└──────────────────────────┬───────────────────────────────────────┘
                           │ SELECT queries
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [1] INGESTION   (ingestion/data_extraction.py)                  │
│  Returns 6 DataFrames (one per source table).                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [2] TRANSFORMATION   (transformation/feature_engineering.py)    │
│  For each student, compute 32 signals AS OF ref_date.            │
│  Label: did they go silent in the 7-day window AFTER ref_date?   │
│  ↑ TEMPORAL SPLIT — features look BACKWARD, label looks FORWARD. │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [3] TRAINING   (training/train.py)                              │
│  XGBoost + stratified 5-fold CV + MLflow logging.                │
└──────────────────────────┬───────────────────────────────────────┘
                           │ run_id
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [4] REGISTRY   (pipeline.py → stage_register)                   │
│  Register as a new version of "taalmaster-dropout".              │
│  Promote to Production only if it beats the previous Production. │
└──────────────────────────┬───────────────────────────────────────┘
                           │ models:/taalmaster-dropout/Production
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [5] PREDICTION   (inference/predict.py)                         │
│  Score every student → probability + risk level + risk factors.  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  [6] SERVING   (serving/api.py)                                  │
│  FastAPI → /predictions, /summary, /predictions/{id}, /health    │
│  Loads the Production model from the MLflow registry at startup. │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```
taalmaster-ml/
│
├─ Configuration & entry points (root)
│  ├── config.py               ── shared settings
│  ├── pipeline.py             ── end-to-end orchestrator
│  ├── run_scheduled.bat       ── Windows Task Scheduler wrapper
│  ├── requirements.txt        ── Python deps
│  ├── README.md               ── quick-start
│  ├── INSTRUCTIONS.md         ── this document
│  ├── .env                    ── Neon DATABASE_URL (secret, gitignored)
│  └── .env.example            ── template
│
├─ Pipeline packages (one per pipeline stage)
│  ├── ingestion/
│  │   ├── data_extraction.py       ── SQL queries → DataFrames
│  │   └── seed_students.py         ── utility: seed fake students into DB
│  │
│  ├── transformation/
│  │   └── feature_engineering.py   ── 32 features + forward label
│  │
│  ├── training/
│  │   ├── train.py                 ── XGBoost, 5-fold CV, MLflow
│  │   └── compare_models.py        ── LR vs RF vs XGB benchmark
│  │
│  ├── evaluation/
│  │   └── backtest.py              ── temporal holdout validation
│  │
│  ├── inference/
│  │   └── predict.py               ── scoring + risk-factor explanation
│  │
│  └── serving/
│      └── api.py                   ── FastAPI endpoints
│
├─ DVC pipeline runners
│  ├── dvc.yaml                ── declarative stage DAG
│  ├── params.yaml             ── tunable parameters
│  ├── dvc.lock                ── auto-generated; hashes every stage
│  ├── metrics.json            ── latest training metrics (git-tracked)
│  └── scripts/
│      ├── run_extract.py          ── DVC stage 1 entry point
│      ├── run_features.py         ── DVC stage 2 entry point
│      └── run_train.py            ── DVC stage 3 entry point
│
├─ Outputs (git-ignored, DVC-managed where applicable)
│  ├── data/
│  │   ├── raw/                ── timestamped snapshots from pipeline.py
│  │   ├── features/           ── timestamped feature matrices
│  │   ├── predictions/        ── timestamped JSON prediction dumps
│  │   └── dvc/                ── stable-path outputs for DVC stages
│  │       ├── raw/
│  │       └── features.parquet
│  ├── artifacts/              ── latest model files (joblib)
│  ├── mlruns/                 ── MLflow tracking + registry
│  └── logs/                   ── scheduled-task stdout/stderr (one per run)
│
└─ DVC metadata
   └── .dvc/                   ── remote config, cache location
```

---

## 5. Environment Setup

### 5.1 Prerequisites

- **Python 3.12+** (`python --version` should return 3.12 or later)
- **Windows 11** (scripts assume this; cmd.exe for Task Scheduler)
- **Access to the Neon database** (DATABASE_URL from the Render dashboard)

### 5.2 First-time install

```bash
cd c:/Users/dlokanath/taalmaster-ml
python -m venv .venv                  # optional but recommended
.venv/Scripts/activate                # Windows
pip install -r requirements.txt
```

### 5.3 .env configuration

Copy the template and fill in the Neon URL:

```bash
cp .env.example .env
```

Then edit `.env`:

```
DATABASE_URL=postgresql://neondb_owner:npg_…@ep-plain-surf-am8q4jxh-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require
```

Get this URL from the Render dashboard → taalmaster-api service → Environment
tab. **Never commit this file to git** (it's in `.gitignore`).

Optional vars:
- `MLFLOW_TRACKING_URI` — defaults to `file:./mlruns`. Set to a remote URI if
  you run MLflow as a server.

### 5.4 Verify installation

```bash
python -c "from ingestion.data_extraction import extract_all; print('OK')"
```

If this runs without error, the DB connection works.

---

## 6. Stage 1 — Ingestion

**File:** [ingestion/data_extraction.py](ingestion/data_extraction.py)

### What it does

Pulls six tables from the Neon database using SQLAlchemy and returns them as
pandas DataFrames in one dict.

### The six tables and why we need each

| Table | Why we pull it |
|---|---|
| `users + subscriptions` | Who the student is, account age, trial status, paying status |
| `study_streaks` | Which days each student logged in — the primary activity signal |
| `quiz_attempts` | Practice frequency, scores, improvement trend |
| `word_progress` | How many words the student has attempted and mastered |
| `writing_submissions` | Highest-effort activity — strongest commitment signal |
| `audio_progress` | Passive listening — low-effort engagement signal |

Plus content totals (`total_vocabulary`, `total_audio`, etc.) so we can
compute coverage percentages.

### Key function

```python
from ingestion.data_extraction import extract_all
raw = extract_all()
# raw is {"students": DF, "streaks": DF, "quizzes": DF, "words": DF,
#         "writing": DF, "audio": DF, "content_counts": dict}
```

### Notes

- All queries use parameterized SQL (via `sqlalchemy.text`) — no injection risk.
- `get_students()` filters to `role = 'student'` so admin/staff accounts don't contaminate training.
- The extract is read-only; it never writes to Neon.

---

## 7. Stage 2 — Feature Engineering

**File:** [transformation/feature_engineering.py](transformation/feature_engineering.py)

### What it does

Turns raw database rows into 32 per-student features that a machine-learning
model can understand. Also computes the label (`is_dropout`).

### Why it matters

This is where 80% of ML value lives. A row saying `user_id=5, activity_date=2026-04-10`
means nothing to a model. But `user 5 has been inactive for 6 days and their
quiz scores have been declining` is a strong dropout signal.

### The 32 features, grouped by concept

#### Activity (9 features) — the most predictive group

| Feature | What it measures |
|---|---|
| `days_since_last_activity` | Days between ref_date and their last recorded activity |
| `total_active_days` | Lifetime count of distinct active days |
| `active_days_last_7` | Count of active days in the 7 days before ref_date |
| `active_days_last_14` | Same, 14-day window |
| `active_days_last_30` | Same, 30-day window |
| `longest_streak` | Their best-ever consecutive-day streak |
| `current_streak` | Consecutive days of activity up to ref_date |
| `avg_gap_between_sessions` | Average days between their distinct active days |
| `activity_trend` | `active_days_last_7 / active_days_last_30` — are they fading? |

`activity_trend` is the single most valuable engineered feature — it catches
students who are *fading* (still active, but less and less) before they fully stop.

#### Quiz (7 features) — performance + effort

| Feature | Meaning |
|---|---|
| `total_quizzes` | Lifetime count |
| `avg_quiz_score_pct` | Mean score across all quizzes |
| `quiz_score_trend` | Slope of a linear fit to score over time — are they improving? |
| `distinct_quiz_modes` | Variety (flashcard / multiple choice / fill blank / translation) |
| `distinct_sections_quizzed` | Breadth of content explored |
| `quizzes_last_7_days` | Recent practice volume |
| `quizzes_last_14_days` | Same, 2-week window |

A declining `quiz_score_trend` often predicts dropout *before* activity drops — frustration comes first.

#### Vocabulary (6 features) — breadth + depth

| Feature | Meaning |
|---|---|
| `words_attempted` | Count of distinct words they've tried |
| `words_mastered` | Count where status == "mastered" |
| `words_learning` | Count where status == "learning" |
| `mastery_rate` | `words_mastered / words_attempted` |
| `avg_accuracy` | `sum(correct_attempts) / sum(attempts)` |
| `vocab_coverage` | `words_attempted / total_vocabulary` — how much of the platform they've touched |

#### Writing (3 features) — highest-effort activity

| Feature | Meaning |
|---|---|
| `total_writing_submissions` | Lifetime count |
| `avg_writing_score` | Mean of admin/AI-assigned scores |
| `writing_score_trend` | Linear fit over time |

Students who still submit writings are usually the most committed — it's the
most effortful activity on the platform.

#### Audio (2 features) — passive engagement

| Feature | Meaning |
|---|---|
| `audio_listened_count` | Distinct audio files marked listened |
| `audio_completion_rate` | `listened / total_audio` |

Low-effort but non-zero signal.

#### Account (5 features) — subscription is a churn signal

| Feature | Meaning |
|---|---|
| `account_age_days` | Days since `users.created_at` |
| `is_trial_expired` | 1 if their trial ended AND they have no active/trialing subscription |
| `has_subscription` | 1 if subscription status is "active" or "trialing" |
| `is_subscription_cancelling` | 1 if `cancel_at_period_end = true` |
| `days_until_sub_expires` | Days until `current_period_end` (−1 if no sub) |

`is_trial_expired` is a canonical churn point — the moment the paywall hits.

### The label

```python
is_dropout = 1 if the student had ZERO activity in the window
             (ref_date, ref_date + 7 days]
             else 0
```

Not "are they currently inactive" (that would be circular). It's: *standing at
ref_date, using only what we knew then, did they actually go silent in the
next 7 days?*

See §8 for why this matters.

### Key functions

```python
from transformation.feature_engineering import build_feature_matrix, get_feature_columns

feature_cols = get_feature_columns()     # list of 32 strings
features = build_feature_matrix(
    raw,                                 # output of extract_all()
    ref_date=datetime(2026, 4, 8),       # cut-off
    label_window_days=7,                 # forward window for label
    compute_label=True,                  # False at inference time
)
# features is a DataFrame with columns: user_id + 32 features + is_dropout
```

---

## 8. The Temporal Split (most important concept)

This is the single most important design decision in the repo, and we got it
wrong on the first attempt. Read this section carefully.

### The initial bug

First version: "dropout" was defined as `days_since_last_activity >= 7`, and
`days_since_last_activity` was also one of the model's features. XGBoost
trivially learned "if the feature is ≥ 7, predict dropout" — scoring a
**perfect 1.0 F1**. The model wasn't predicting anything; it was restating
the definition.

This is called **target leakage**: when a feature can be used to algebraically
derive the label. The symptoms:
- Implausibly high metrics (0.99+ on a "real" task).
- The medium-risk bucket is empty — everyone's probability is 0 or 1.
- The model makes no actual predictions; it parrots the rule back.

### The first attempted fix (insufficient)

We tried excluding `days_since_last_activity` from the features. F1 stayed at
1.0. Why? Because other features *also* leak:

- `active_days_last_7 = 0` ⟹ inactive for 7+ days ⟹ dropout = 1 (definitional)
- `active_days_last_14 = 0` ⟹ same logic
- `current_streak > 0` ⟹ active within 1 day ⟹ dropout = 0 (definitional for negatives)
- `activity_trend` = ratio of the above → also leaks

Any recent-activity feature, combined with the snapshot-time label, lets the
model re-derive the rule.

### The actual fix — the temporal split

The real problem: **features and label were measured at the same moment**. A
model that only works "today" can still score perfectly on today's data.

The fix is to measure them at *different* moments:

```
Past (features)         |   Future (label)
|-----------------------| T |-------------------|
features use data < T   |   label uses data in [T, T+7]
```

- `ref_date = T` is a cut-off point.
- **Features** are computed using only data from before T. Everything `< T`
  is "the past the model could have seen at prediction time."
- **Label** is computed using data in the window `[T, T + 7 days]`. This is
  "the future the model is trying to predict."

The model never has access to its own answer. Features answer "what did this
student's past look like at time T?" Label answers "were they silent in the
week after T?" Those are genuinely different questions.

### Implementation

Every feature builder now takes `ref_date` and starts with:

```python
streaks = streaks[streaks["activity_date"] < ref_date]   # temporal cut
```

This single line forces every aggregation (last_activity, active_days_last_7,
etc.) to respect the cut-off.

The label is computed by a separate function:

```python
def compute_forward_label(students, streaks, ref_date, window_days=7):
    future = streaks[(streaks["activity_date"] >= ref_date)
                   & (streaks["activity_date"] < ref_date + window_days)]
    # is_dropout = 1 if the student has NO rows in `future`
```

### Why pipeline.py picks a past ref_date for training

```python
ref_date = now - timedelta(days=label_window_days + 1)   # i.e. 8 days ago
```

If we trained at `ref_date = now`, the forward window would be in the future
— no data yet — every label would be 1. Useless. By picking a past ref_date,
the forward window is entirely in the observed past, so labels are real.

At **inference** time (scoring live students), `ref_date = now`, and we pass
`compute_label=False` because the future hasn't happened yet.

### The result

After the fix:
- CV F1 dropped from a fake **1.0** to an honest **0.9757**.
- Medium-risk bucket is populated (the model now has uncertainty).
- Temporal backtest confirmed it generalizes to unseen weeks.

MLflow registry still has the leakage models as `v1` and `v2` (archived). `v3`
and `v4` are honest models. The API loads `v4` (current Production).

---

## 9. Stage 3 — Training

**File:** [training/train.py](training/train.py)

### What it does

Takes the feature matrix and the label, trains an XGBoost classifier with
5-fold cross-validation, logs everything to MLflow, and saves the final
model to disk.

### Why XGBoost

XGBoost is gradient-boosted decision trees. It wins on tabular data at this
scale because:

- Handles missing values natively.
- Handles heterogeneous feature scales without normalization.
- Builds trees sequentially — each tree corrects the previous one's errors.
- Feature importances come out interpretable.
- Fast to train (seconds) and fast to predict (milliseconds).

### Key hyperparameters

```python
n_estimators=200       # 200 trees total
max_depth=5            # each tree max 5 levels deep (prevents overfit)
learning_rate=0.05     # each tree contributes 5% (small steps, less overfit)
subsample=0.8          # each tree sees 80% of rows (bagging inside boosting)
colsample_bytree=0.8   # each tree sees 80% of columns
scale_pos_weight=n_neg/n_pos  # compensate class imbalance
min_child_weight=3     # trees can't split on tiny groups
gamma=0.1              # splits must improve loss by at least 0.1
reg_alpha=0.1          # L1 regularization on weights
reg_lambda=1.0         # L2 regularization on weights
```

### Stratified 5-fold cross-validation

The 2,352-student training matrix is split into 5 equal chunks, preserving
the dropout/active ratio in each (that's the "stratified" part). We train on
4 chunks, evaluate on the 5th, and rotate. Every student gets evaluated
exactly once, on a model that never saw them during training.

Then we average the 5 scores — that's the honest "CV F1" metric. Finally, we
retrain on ALL the data to get the model we actually deploy.

### Metrics tracked

| Metric | What it means |
|---|---|
| `cv_accuracy_mean` | Overall correctness rate |
| `cv_precision_mean` | Of students flagged as dropout, what % really are? |
| `cv_recall_mean` | Of actual dropouts, what % did we catch? |
| `cv_f1_mean` | Harmonic mean of precision & recall — our primary metric |
| `cv_roc_auc_mean` | Ranking quality (1.0 = perfect, 0.5 = random) |

Each metric also gets a `_std` suffix showing variance across the 5 folds.

### Artifacts logged per run

- The trained XGBClassifier (as both native xgboost and joblib)
- `feature_columns.joblib` — the exact 32-column order
- `feature_importances.json` — which features mattered most
- `classification_report.txt` — per-class precision/recall/support
- `confusion_matrix.json` — the 2×2 error breakdown

All browseable at http://127.0.0.1:5000 after running `mlflow ui`.

### Training standalone

```bash
python -m training.train                    # use live Neon data
python -m training.train --synthetic 500    # use 500 fake students (fast testing)
```

---

## 10. Stage 3.5 — Evaluation & Backtest

### Model comparison — [training/compare_models.py](training/compare_models.py)

Trains LR, RF, and XGBoost on the same feature matrix, under the same 5-fold
CV, and shows them side-by-side. Useful for: is XGBoost actually needed, or
would a simpler model do?

**Result on our data:**

| Model | CV F1 |
|---|---|
| Logistic Regression | 0.9757 |
| Random Forest | 0.9760 |
| XGBoost | 0.9753 |

A three-way tie within noise. **Interpretation:** the dropout patterns in our
data are mostly *linear* — a simple linear decision boundary separates active
from dropout. XGBoost's non-linear power finds no extra lift. In production
we could replace XGBoost with logistic regression for simpler deployment and
native interpretability, with no loss of accuracy.

Run with:

```bash
python -m training.compare_models
```

### Temporal backtest — [evaluation/backtest.py](evaluation/backtest.py)

Cross-validation reuses the same point in time: every fold has features AND
label from the same ref_date. A model could in principle overfit to
week-specific noise.

The backtest is stricter: train on a ref_date 21 days ago, score on a
ref_date 14 days ago — a week the model never saw. This mimics production:
you train Monday, deploy, and get scored on next week's data.

**Result on our data:**

| Metric | CV on TRAIN | HOLDOUT on TEST | Delta |
|---|---|---|---|
| F1 | 0.9649 | 0.9695 | +0.005 |
| ROC AUC | 0.9945 | 0.9942 | −0.000 |

Holdout is within noise of CV — the model generalizes cleanly. The strong
score isn't an artifact of a single-week fluke.

Run with:

```bash
python -m evaluation.backtest
# or with custom windows:
python -m evaluation.backtest --train-offset-days 28 --test-offset-days 14 --window-days 7
```

### Drift detection — [evaluation/drift_check.py](evaluation/drift_check.py)

Compares today's feature distributions against the most recent training
snapshot and flags per-feature drift. This is the **"drift" half** of the
"Scheduled / drift" retrain trigger in the architecture diagram.

Method, per feature:
- Continuous features → **Kolmogorov-Smirnov 2-sample test**
- Binary (0/1) features → **chi-square test** on a 2×2 contingency table

A feature is flagged as drifted when its p-value is below the threshold
(default 0.01 — less than 1% chance the distributions are the same).

```bash
python -m evaluation.drift_check                    # report + log to MLflow
python -m evaluation.drift_check --threshold 0.05   # looser significance
python -m evaluation.drift_check --fail-on-drift    # exit 1 if any drift
python -m evaluation.drift_check --no-mlflow        # skip MLflow logging
```

**Output** — an ASCII table of drifted features with p-value, reference
mean, current mean, and percent delta. Plus a full JSON report uploaded to
MLflow under the tag `purpose=drift_check`.

**Cron chaining** — retrain only when drift is detected:

```bash
python -m evaluation.drift_check --fail-on-drift \
    || python pipeline.py
```

**Expected behavior** — some drift is normal even in a healthy product.
`days_since_last_activity`, recent-activity windows, and `account_age_days`
will always shift if the reference snapshot is more than a few days old,
because those features encode *calendar* time. A sudden shift in
`is_trial_expired` or `has_subscription` rates is a stronger signal that
something operationally changed (campaign, price change, cohort shift).

---

## 11. Stage 4 — Prediction & Explainability

**File:** [inference/predict.py](inference/predict.py)

### What it does

Loads the current Production model from the MLflow registry, scores every
student in the DB, and for each one produces:

1. **Probability** of dropout (0.0 – 1.0)
2. **Risk level** — `high` (>0.7), `medium` (0.4–0.7), `low` (<0.4)
3. **Top risk factors** — ranked features that deviate from a "healthy" baseline
4. **Recommendation** — a specific admin action based on the top factor

### Why a probability alone isn't enough

"Student 42 has 87% dropout risk" — now what? An admin can't act on a
probability. They need to know *why* the student is at risk and *what to do*
about it. That's what the explainability layer provides.

### How risk factors are computed

For each student and each feature, we check if the value deviates from a
"healthy" threshold. If so, we compute a severity score:

```
severity = (threshold - value) / threshold × feature_importance
```

So a feature only shows up as a risk factor if:
- the student is actually unhealthy on that feature, AND
- the MODEL thinks that feature matters (not just our intuition)

This is nicer than hardcoded rules — it reflects what the model actually
learned.

### The recommendation map

Each top-factor → a specific, actionable suggestion:

| Top risk factor | Recommendation |
|---|---|
| `days_since_last_activity` | Send a re-engagement email with new content highlights |
| `active_days_last_7` | Suggest a short daily quiz challenge to rebuild habit |
| `current_streak` | Encourage starting a new streak with a small goal |
| `avg_quiz_score_pct` | Suggest reviewing vocabulary before quizzing |
| `words_mastered` | Focus on flashcard mode to build core vocabulary |
| `has_subscription` | Consider offering a discount or extended trial |
| `is_trial_expired` | Send a trial extension offer |
| `avg_gap_between_sessions` | Set up a reminder notification |
| `activity_trend` | Check if the student is stuck on a difficult section |

For "high" risk students, the recommendation is prefixed with `URGENT:` so
admins can prioritize.

### Using it

```python
from ingestion.data_extraction import extract_all
from inference.predict import predict_all_students, predict_single_student

raw = extract_all()
all_preds = predict_all_students(raw)       # sorted highest-risk first
one_pred  = predict_single_student(raw, user_id=42)
```

---

## 12. Stage 5 — Serving (FastAPI)

**File:** [serving/api.py](serving/api.py)

### What it does

Exposes predictions over HTTP so the TaalMaster admin dashboard (React + Express)
can consume them.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info, list of endpoints |
| GET | `/health` | Is the model loaded? Which version? |
| GET | `/summary` | Aggregate risk stats for the dashboard |
| GET | `/predictions` | All students, ranked by risk |
| GET | `/predictions/{user_id}` | One student's risk breakdown |
| POST | `/predict/dropout-risk` | Same payload as GET /predictions/{id}; body `{"user_id": N}` — canonical name from the architecture diagram |
| GET | `/docs` | Interactive Swagger UI |

### Example response — /health

```json
{
  "status": "healthy",
  "model_version": "taalmaster-dropout v4",
  "model_source": "mlflow_registry",
  "timestamp": "2026-04-16T20:30:03.828145"
}
```

### Example response — /summary

```json
{
  "total_students": 2503,
  "high_risk_count": 1304,
  "medium_risk_count": 25,
  "low_risk_count": 1174,
  "avg_dropout_probability": 0.5343,
  "top_risk_factors": [
    {"feature": "active_days_last_14", "count": 1329},
    {"feature": "active_days_last_7", "count": 1329},
    {"feature": "days_since_last_activity", "count": 1327},
    ...
  ]
}
```

### How the API loads the model

At first request, the API:

1. Contacts the MLflow registry.
2. Asks for the model version currently in stage "Production".
3. Downloads the joblib artifact for that run.
4. Caches the model in memory for the rest of the process lifetime.

When you promote a new version via `pipeline.py`, **restart the API** and it
picks up the new one. There's no hot-reload; this is deliberate (predictable
behavior, stable production).

If the registry is unreachable, the API falls back to the local
`artifacts/dropout_model.joblib`. You'll see this in `/health` as
`"model_source": "local_file"`.

### CORS configuration

The API allows requests from:

- `http://localhost:3000` (local React dev)
- `http://localhost:5173` (Vite dev server)
- `https://taalmaster.vercel.app` (production)

Add more origins in [serving/api.py](serving/api.py) `allow_origins` if needed.

### Running the API

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI.

---

## 13. MLflow Deep Dive

### What MLflow does for us

Two things:

1. **Experiment tracking** — every training run's params, metrics, and artifacts
   are logged to `mlruns/` on disk. Browseable via `mlflow ui`.
2. **Model registry** — named model versions with stages (None / Staging /
   Production / Archived). The API loads whichever version is Production.

### Starting the MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000.

### What you see in the UI

**Experiments tab** — the experiment `taalmaster-dropout-prediction` holds
every training run. Each row is one run. Columns show params and metrics.
Sort by any column, filter, or tick 2+ runs and click "Compare" for a
side-by-side view.

**Models tab** — the `taalmaster-dropout` registry. Shows all versions, their
metrics, and their current stage.

### How we use the registry

`pipeline.py → stage_register()` does this every time:

1. Register this run's model as a new version of `taalmaster-dropout`.
2. Fetch the current Production version's F1.
3. If the new model's F1 > old Production's F1 → promote the new version
   to Production and archive the old one.
4. Otherwise → keep old Production, leave new version in stage None.

This is the "shadow deploy" pattern in miniature: new models are always
registered, but only promoted if they actually beat the incumbent.

### Registry state history in this project

| Version | Training F1 | Status | Note |
|---|---|---|---|
| v1 | 1.0000 | Archived | Leakage bug — model memorized the rule |
| v2 | 1.0000 | Archived | Leakage bug (partial fix attempted, didn't work) |
| v3 | 0.9753 | Archived | First honest model (after full temporal split fix) |
| v4 | 0.9757 | Production | Current — same methodology, slightly better CV F1 |

### Promoting / archiving manually

If you ever need to manually intervene:

```python
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Archive a version
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME, version='5', stage='Archived'
)

# Promote a version
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME, version='6', stage='Production',
    archive_existing_versions=True
)
```

### MLflow vs MLflow tracking URI

If you leave `MLFLOW_TRACKING_URI` unset, it defaults to `file:./mlruns`
(local disk). For a team or production setup, point it at a remote MLflow
server — the code doesn't change.

---

## 14. DVC Deep Dive

### What DVC does for us

DVC is "git for big files and pipelines." It lets us:

1. Version the data snapshots (not just code).
2. Declare the pipeline as a DAG (`dvc.yaml`).
3. Rerun only stages whose inputs changed (`dvc repro`).
4. Parameter-sweep via experiments (`dvc exp run`).
5. Share data with teammates (`dvc push` / `dvc pull`).

### The DAG

```
extract → features → train
```

Defined in [dvc.yaml](dvc.yaml). View visually with `dvc dag`.

### The three stages

| Stage | Script | Deps | Outputs |
|---|---|---|---|
| extract | [scripts/run_extract.py](scripts/run_extract.py) | ingestion code, config.py | `data/dvc/raw/*.parquet` |
| features | [scripts/run_features.py](scripts/run_features.py) | transformation code, raw parquet, params | `data/dvc/features.parquet` |
| train | [scripts/run_train.py](scripts/run_train.py) | training code, features.parquet | `artifacts/*.joblib`, `metrics.json` |

The extract stage has `always_changed: true` because the Neon DB is an
external source — we want every `dvc repro` to re-pull. Downstream stages
only re-run if the extract's output hash actually changed.

### Parameters

[params.yaml](params.yaml) holds tunable pipeline parameters:

```yaml
features:
  ref_offset_days: 8    # ref_date = now - 8 days
  window_days: 7        # forward label window
```

Change any value and `dvc repro` detects it — only stages that depend on
that param (via `params:` in `dvc.yaml`) re-run.

### Running the DVC pipeline

```bash
# Run everything that's out of date
dvc repro

# Force a specific stage to re-run
dvc repro --force features

# See the DAG
dvc dag

# What's changed since the last run?
dvc status
```

### Experiments

DVC experiments let you spin off variant runs *without* git commits — each
one is tracked under `.git/refs/exps/`. We used them to sweep `window_days`:

```bash
dvc exp run --name baseline                          # window_days=7
dvc exp run --name window_14 --set-param features.window_days=14
dvc exp run --name window_3  --set-param features.window_days=3
dvc exp show                                         # table of all experiments
dvc exp apply window_3                               # restore workspace to that experiment
dvc exp remove window_3                              # delete one
```

**Our findings from this sweep:**

| window_days | dropout_rate | F1 | Takeaway |
|---|---|---|---|
| 3 | 59.6% | 0.9254 | Noisier label — more people skip 3 days occasionally |
| 7 | 53.9% | 0.9757 | Sweet spot — strong signal, earliest actionable warning |
| 14 | 53.9% | 0.9757 | Identical to 7 — extending the window adds no info |

Going past 7 days adds zero model performance. Dropping to 3 days costs 5 F1
points. So 7 days is the right business choice.

### Metrics

[metrics.json](metrics.json) is a small JSON file tracked in git (not DVC).
Every `dvc repro` overwrites it with the latest training metrics.

```bash
dvc metrics show                    # current state
dvc metrics diff                    # diff against HEAD
dvc metrics diff HEAD~1 HEAD        # diff two commits
```

### Remote storage

We configured a local-folder remote at `C:/Users/dlokanath/dvc-storage-taalmaster/`:

```bash
dvc push    # uploads data + model files to the remote
dvc pull    # downloads them (use after a fresh clone)
```

To use S3/GCS/Azure instead, add the appropriate DVC extras to requirements
and configure a remote URL:

```bash
pip install 'dvc[s3]'
dvc remote add -d s3-storage s3://my-bucket/taalmaster-dvc
dvc remote modify s3-storage region us-east-1
```

### Two pipelines coexist

| Use case | Command | Output path | Snapshots | Reproducible? |
|---|---|---|---|---|
| Ad-hoc retrain | `python pipeline.py` | `data/raw/TIMESTAMP/` | preserved | timestamp-only |
| Iterating on features | `dvc repro` | `data/dvc/...` | overwritten | fully (git + DVC) |
| Scheduled daily job | `python pipeline.py` via Task Scheduler | timestamped | preserved | timestamp-only |
| Parameter sweep | `dvc exp run --set-param X=Y` | `.git/refs/exps/` | ephemeral | fully |

Both log to MLflow. Both update the registry.

### DVC UI

The easiest UI is the **VSCode DVC extension** (install from the Extensions
marketplace, search "DVC" by Iterative). It shows:

- Experiments table (same as `dvc exp show` but interactive)
- Plots from metrics.json history
- DAG visualization
- Right-click to apply / remove / push experiments

---

## 15. Orchestration — pipeline.py

**File:** [pipeline.py](pipeline.py)

This is the end-to-end orchestrator. It chains all five pipeline stages in
one command.

### What it does, step by step

1. **Extract** — calls `extract_all()`, saves each DataFrame as parquet under
   `data/raw/YYYYMMDD_HHMMSS/`.
2. **Features** — calls `build_feature_matrix()` with `ref_date = now - 8 days`,
   saves `data/features/features_YYYYMMDD_HHMMSS.parquet`.
3. **Train** — calls `train_model()` → 5-fold CV + MLflow logging.
4. **Register** — registers as a new version of `taalmaster-dropout`,
   promotes to Production if it beats the current one.
5. **Predict** — scores every student, saves `data/predictions/predictions_YYYYMMDD_HHMMSS.json`.

### CLI options

```bash
python pipeline.py                          # full pipeline, default settings
python pipeline.py --skip-train             # refresh predictions, skip training
python pipeline.py --no-register            # train but don't touch the registry
python pipeline.py --ref-date 2026-04-09    # train on a specific ref_date
```

### Output

Each run produces three timestamped folders/files:

- `data/raw/20260416_193734/` — six parquet files + `content_counts.json`
- `data/features/features_20260416_193734.parquet` — the 32-feature matrix
- `data/predictions/predictions_20260416_193734.json` — all 2,503 students scored

Plus one MLflow run and (if promoted) one new version in the registry.

---

## 16. Scheduling — Windows Task Scheduler

### What's scheduled

A Windows Task Scheduler entry called `TaalmasterMLPipeline` that runs daily
at 10:00 AM, invoking [run_scheduled.bat](run_scheduled.bat), which in turn
runs `python pipeline.py` and pipes output to a timestamped log file under
`logs/`.

### The wrapper script

[run_scheduled.bat](run_scheduled.bat):

- Changes directory into the repo
- Creates `logs/` if it doesn't exist
- Invokes `python pipeline.py`, redirecting stdout + stderr to
  `logs/pipeline_YYYYMMDD_HHMMSS.log`
- Exits with the pipeline's return code

### Useful Task Scheduler commands (from Git Bash)

```bash
# Status and next run time
cmd.exe //c "schtasks /Query /TN TaalmasterMLPipeline /FO LIST /V"

# Run the scheduled task NOW (don't wait for 10am)
cmd.exe //c "schtasks /Run /TN TaalmasterMLPipeline"

# Change the time (e.g. to 09:30)
cmd.exe //c "schtasks /Change /TN TaalmasterMLPipeline /ST 09:30"

# Delete the task
cmd.exe //c "schtasks /Delete /TN TaalmasterMLPipeline /F"

# Re-create the task
cmd.exe //c "schtasks /Create /TN TaalmasterMLPipeline /TR \"C:\Users\dlokanath\taalmaster-ml\run_scheduled.bat\" /SC DAILY /ST 10:00 /F"
```

### Caveats

1. **Your laptop must be awake at 10am** — Task Scheduler won't wake a sleeping
   Windows machine by default. If this matters, enable "Wake the computer to
   run this task" in the Task Scheduler GUI.
2. **You must be logged in** — we didn't store a password, so the task runs
   as user `dlokanath` only when you're logged in. To run logged out, re-create
   the task with `/RU` + `/RP`.
3. **The task's working directory is set by the batch file's `cd /d "%~dp0"`**,
   which resolves to the repo root. Don't move the batch file.

### Alternatives

- **GitHub Actions cron** — runs in the cloud, survives laptop sleep. But MLflow
  writes to local disk (`file:./mlruns`), so you'd need a remote tracking server
  before this makes sense.
- **Plain Linux cron** — if/when you deploy to a server: `0 10 * * * cd /path/to/repo && python pipeline.py`
- **Drift-triggered retraining** — smarter than pure scheduled. Check feature
  distributions hourly; retrain only when they've drifted. Not built yet.

---

## 17. How to Run Everything (command reference)

### Full pipeline

```bash
python pipeline.py                          # end-to-end (extract → serve)
python pipeline.py --skip-train             # just refresh predictions
```

### DVC pipeline

```bash
dvc repro                                   # run stale stages
dvc repro --force train                     # force a stage
dvc dag                                     # show DAG
dvc status                                  # what's out of date?
```

### Individual stages (for debugging)

```bash
python -m training.train                    # train only (skips extraction)
python -m training.train --synthetic 500    # train on 500 fake students
python -m training.compare_models           # LR vs RF vs XGBoost
python -m evaluation.backtest               # temporal holdout check
python -m evaluation.drift_check            # feature drift report (logs to MLflow)
python -m evaluation.drift_check --fail-on-drift   # exit 1 if any feature drifted
python -m ingestion.seed_students           # seed 2500 fake students
python -m ingestion.seed_students --count 500
python -m scripts.run_extract               # DVC extract step alone
python -m scripts.run_features              # DVC features step alone
python -m scripts.run_train                 # DVC train step alone
```

### DVC experiments

```bash
dvc exp run --name baseline
dvc exp run --name window_14 --set-param features.window_days=14
dvc exp show
dvc exp show --csv                          # machine-readable
dvc exp apply window_14                     # restore workspace to this exp
dvc exp remove window_14                    # delete one
dvc exp push origin --all                   # push to git remote (if configured)
```

### DVC data transfer

```bash
dvc push                                    # upload tracked files to remote
dvc pull                                    # download them
dvc status --cloud                          # diff local vs remote
```

### MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
# Then open http://127.0.0.1:5000
```

### API

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
# Then:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/summary
curl "http://127.0.0.1:8000/predictions?risk_level=high&limit=5"
curl http://127.0.0.1:8000/predictions/42
# Or open http://127.0.0.1:8000/docs for Swagger
```

### Scheduler

```bash
# Trigger scheduled task now
cmd.exe //c "schtasks /Run /TN TaalmasterMLPipeline"
# Check status
cmd.exe //c "schtasks /Query /TN TaalmasterMLPipeline /FO LIST /V"
```

---

## 18. Operational Playbook

### How to retrain with fresh data

```bash
python pipeline.py
```

That's it. `pipeline.py` extracts fresh data, trains, and promotes if the
new model beats the current Production.

### How to promote a specific past run to Production

Find the `run_id` in the MLflow UI, then:

```python
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
run_id = "<your run id>"

# Register this run's model as a new version
import mlflow
v = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)

# Promote it
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=v.version,
    stage="Production",
    archive_existing_versions=True,
)
```

Then restart the API to pick up the change.

### How to roll back to a previous version

Same as above, but with the `run_id` of the older model (find it in the MLflow
UI's Models tab → taalmaster-dropout → the version you want → copy its
source run ID).

### How to change the dropout definition

Edit `DROPOUT_DAYS_THRESHOLD` in [config.py](config.py) and/or
`window_days` in [params.yaml](params.yaml). Then:

```bash
python pipeline.py          # for the ad-hoc pipeline
dvc repro                    # for the DVC pipeline
```

Both will rebuild features and retrain with the new threshold.

### How to add a new feature

1. Add the SQL query in [ingestion/data_extraction.py](ingestion/data_extraction.py).
2. Add the extraction to the `extract_all()` return dict.
3. Add a `build_X_features` function in [transformation/feature_engineering.py](transformation/feature_engineering.py)
   that takes `students`, your new DataFrame, and `ref_date`, applies the
   temporal cut (`df < ref`), and returns a per-student DataFrame.
4. Add it to the merge chain in `build_feature_matrix`.
5. Add the new column names to `get_feature_columns()`.
6. Add a "healthy threshold" in [inference/predict.py](inference/predict.py) `_get_risk_factors`
   (if you want it to appear in risk factors).
7. Add a recommendation in `_generate_recommendation` (if applicable).
8. Run `python pipeline.py`. Compare the new model's F1 against v4's in MLflow.

### How to deploy the API

Currently the API runs locally. To deploy:

1. Decide on hosting (Render, Fly, AWS Lambda container, etc.).
2. Push the repo to GitHub.
3. Configure the hosting platform:
   - Runtime: Python 3.12
   - Start command: `uvicorn serving.api:app --host 0.0.0.0 --port $PORT`
   - Env vars: `DATABASE_URL`, `MLFLOW_TRACKING_URI` (if remote)
4. For MLflow state, either:
   - Ship `mlruns/` in the container (simple but stale)
   - Use a remote MLflow server (proper)
5. Update the main Taalmaster Express server to proxy `/api/ml/*` → the deployed URL.

---

## 19. Troubleshooting

### "DATABASE_URL not set"

You haven't created `.env` yet, or it's missing the `DATABASE_URL` key.
Copy `.env.example` to `.env` and fill in the Neon URL.

### "No model found at artifacts/dropout_model.joblib"

The API is trying to fall back to the local joblib file, but neither the
registry nor local file is available. Run `python pipeline.py` to train a model.

### CV F1 is exactly 1.0

You've reintroduced target leakage — check that the temporal cut
(`streaks = streaks[streaks["activity_date"] < ref]`) is still present in
every feature builder. See §8.

### `dvc repro` re-runs extract every time

Expected. `extract` has `always_changed: true` because the Neon DB is external.
Downstream stages only re-run if the extract's output hash changed.

### `dvc exp show` fails with UnicodeEncodeError

Windows console encoding issue. Use `dvc exp show --csv` for machine-readable
output, or use the VSCode extension for a GUI.

### API returns "model_not_loaded"

Check `/health`. Either:
- The MLflow registry has no Production version (run `python pipeline.py` to
  train + promote one).
- The local joblib file is missing (see above).

### Scheduled task runs but no logs appear

Check that the task's "Start in" directory is set. Our batch script handles
this via `cd /d "%~dp0"`, but if the task was created differently, it may run
in the wrong directory and silently fail. Re-create with:

```bash
cmd.exe //c "schtasks /Create /TN TaalmasterMLPipeline /TR \"C:\Users\dlokanath\taalmaster-ml\run_scheduled.bat\" /SC DAILY /ST 10:00 /F"
```

### MLflow UI shows no runs

Make sure `mlflow ui` is pointed at the right backend:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Run it from the repo root.

---

## 20. Lessons Learned

These are the real takeaways from building this pipeline. Each one is a
general principle, not just a quirk of this project.

### 1. A "perfect" score is almost never a real signal

F1 = 1.0 on a non-trivial task means one of:
- Target leakage (the feature can derive the label)
- Test set contamination (the model saw these rows during training)
- The problem is genuinely trivial (then you don't need ML)

Always question a perfect score before celebrating it.

### 2. Temporal split is mandatory when features and label share a time dimension

If you can construct a feature X and label Y from the same point in time,
and Y is a function of X's recent state, you need the split. No exceptions.

### 3. Simpler models often tie complex ones on clean data

Logistic regression tied XGBoost on our data. That's because:
- Our engineered features already do the heavy lifting — the model just has
  to draw a boundary, not discover patterns.
- Non-linear interactions between features don't exist in this data (or are
  weak).

**Rule of thumb:** always run a linear baseline. If it ties XGBoost,
consider shipping the linear model — easier to deploy, interpret, and debug.

### 4. Cross-validation and temporal backtest measure different things

CV tells you: "on data from this time, how well does the model work?" The
temporal backtest tells you: "will this model still work next week?" Both
matter. A model that passes CV but fails backtest is fitting week-specific
noise.

### 5. Interpretability is a feature, not a bonus

A probability alone is useless to an operator. Top risk factors + a
specific recommendation is what turns "the model says X" into "we should do Y."
Build explainability into the prediction layer from day one.

### 6. MLflow and DVC solve different problems

MLflow tracks experiments and serves as a registry — good for "what's the best
model and which is serving." DVC versions data and pipelines — good for "can
we reproduce this exact training run months from now?" Use both.

### 7. Make the pipeline scriptable BEFORE automating it

The order we built things: manual train → pipeline.py → scheduled task. If
we'd gone straight to the scheduler we'd have fought debugging loops. Make
sure the pipeline works manually before letting cron touch it.

### 8. Package boundaries match pipeline stages

Organizing code by `ingestion/`, `transformation/`, `training/`, etc. makes
the pipeline legible to a stranger reading the repo. Flat structures are
faster to create and slower to understand.

---

## 21. Admin Dashboard Integration (Taalmaster)

The ML service is consumed by the main Taalmaster admin dashboard at
`c:\Users\dlokanath\taalmaster\`. This section documents exactly what was
wired up across the two repos so you can reproduce, maintain, or deploy it.

### 21.1 Architecture of the integration

```
┌───────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  React admin panel    │    │  Express server      │    │  FastAPI (this repo) │
│  (client/ — Vite)     │───▶│  (server/src)        │───▶│  (serving/api.py)    │
│  port 3000 / 5173     │    │  port 5000           │    │  port 8000           │
│  axios → /api/ml/...  │    │  authenticate +      │    │  loads Production    │
│                       │    │  requireAdmin gate   │    │  model from MLflow   │
└───────────────────────┘    │  proxies /api/ml/*   │    │  registry            │
                             │  → ML_API_URL        │    └──────────────────────┘
                             └──────────────────────┘
```

**Why a proxy and not direct browser → FastAPI?**

1. **Single origin from the browser's side.** No CORS headaches.
2. **Reuses the existing JWT auth + admin gate**, so the ML endpoints can't
   be hit anonymously. Without this, anyone who found the FastAPI URL could
   scrape per-student predictions.
3. **Keeps the ML service URL server-side.** Admins see `/api/ml/...`, not
   the actual FastAPI address.

### 21.2 Files added or changed

Three new files, two edits. All living in `c:\Users\dlokanath\taalmaster\`:

| Location | What | Purpose |
|---|---|---|
| `server/src/routes/ml.js` | **NEW** | Express proxy routes with admin gate |
| `server/src/index.js` | edit | Mount the new router at `/api/ml` |
| `server/.env` | edit | Add `ML_API_URL=http://localhost:8000` |
| `client/src/components/admin/DropoutRiskManager.jsx` | **NEW** | Summary + table + detail modal |
| `client/src/components/admin/AdminPanel.jsx` | edit | Register the "Dropout Risk" tab |

### 21.3 The Express proxy — [server/src/routes/ml.js](../taalmaster/server/src/routes/ml.js)

Four routes, all behind `authenticate + requireAdmin`:

| Method | Path | Forwards to FastAPI |
|---|---|---|
| GET | `/api/ml/summary` | `/summary` |
| GET | `/api/ml/predictions` | `/predictions` (query params passed through) |
| GET | `/api/ml/predictions/:userId` | `/predictions/{userId}` |
| GET | `/api/ml/health` | `/health` |

Implementation uses Node 18+ native `fetch` — no new dependency added to
`package.json`. Timeout is 60 seconds (the ML API extracts from Neon and
scores 2,500 students per request, which can take 20–30 seconds).

Error responses:
- **502** when the FastAPI is unreachable
- **504** when the request exceeds 60 seconds
- **4xx** passed through transparently from the FastAPI

### 21.4 Mounting the proxy — `server/src/index.js`

Two lines added (alongside the other route modules):

```js
const mlRoutes = require('./routes/ml');
// ... later:
app.use('/api/ml', mlRoutes);
```

Auth is already enforced inside `ml.js`, so no extra middleware wrapping is
needed here.

### 21.5 Environment variable

Added to `server/.env`:

```
ML_API_URL=http://localhost:8000
```

When deploying the server, set `ML_API_URL` to wherever the FastAPI lives in
production (e.g. a Render service URL). The React side doesn't need changes —
it always hits the same-origin `/api/ml/*`.

**Render / cloud deployment note**: add `ML_API_URL` to the `render.yaml`
envVars list with `sync: false` so you can set it per-environment without
committing secrets.

### 21.6 The React component — [DropoutRiskManager.jsx](../taalmaster/client/src/components/admin/DropoutRiskManager.jsx)

Visible UI elements:

1. **Header** — shows the currently serving model version (pulled from
   `/api/ml/health`, e.g. `taalmaster-dropout v4`). A Refresh button.
2. **Summary cards** (top row) — total students, high/medium/low counts.
   Data from `/api/ml/summary`.
3. **Risk-level tabs** — High / Medium / Low. Default: High.
4. **Students table** — name, email, dropout probability (as a colored
   badge), days inactive, subscription status, top risk factor message.
   Rows clickable.
5. **Detail modal** (on row click) — student name, probability, recommended
   action, context stats (streak, quizzes, words mastered, subscription),
   ranked list of risk factors with severity scores.

Follows the existing admin component patterns:
- Uses the shared `utils/api.js` axios instance (gets JWT auth + base URL for free)
- Tailwind classes matching other admin components (`bg-dutch-blue`, dark: variants)
- `react-hot-toast` for error notifications
- `react-icons/fi` for iconography

### 21.7 Registering the tab — `AdminPanel.jsx`

One import and one tab entry added:

```jsx
import DropoutRiskManager from './DropoutRiskManager';

const tabs = [
  // ... existing tabs ...
  { id: 'dropout', label: 'Dropout Risk' },
];
```

Plus one line to render the component when active:

```jsx
{activeTab === 'dropout' && <DropoutRiskManager />}
```

### 21.8 Running the full stack locally

Three terminals. The order doesn't matter but all three must be running to
see anything in the browser:

```bash
# Terminal 1 — ML service
cd c:\Users\dlokanath\taalmaster-ml
uvicorn serving.api:app --port 8000

# Terminal 2 — Express backend
cd c:\Users\dlokanath\taalmaster\server
npm run dev

# Terminal 3 — React dev server
cd c:\Users\dlokanath\taalmaster\client
npm run dev
```

Then:

1. Open the client URL (usually http://localhost:5173 for Vite or
   http://localhost:3000 depending on config).
2. Log in as an **admin** user (the middleware checks `req.user.role === 'admin'`).
3. Navigate to the Admin Panel.
4. Click the **Dropout Risk** tab.

First load takes ~25 seconds — the FastAPI is re-running `extract_all()` +
`predict_all_students()` for 2,503 students on each request. Subsequent
interactions (clicking rows, switching tabs) are instant.

### 21.9 Known caveats and their fixes

**Port 8000 already in use** — `[Errno 10048]`. A previous uvicorn didn't
shut down cleanly. On Windows, `netstat -ano | findstr :8000` shows the PID;
kill it with `taskkill /F /PID <pid>`. Or just use a different port and
update `ML_API_URL`.

**Every page load is slow** — the FastAPI re-extracts + re-predicts on every
request. For a real deployment, either:
- Cache the /summary response for N minutes server-side in FastAPI, or
- Read predictions from the latest `data/predictions/*.json` file instead
  of recomputing (whichever file was written by the most recent pipeline run).

The daily scheduled pipeline already writes a fresh predictions JSON — the
API just doesn't use it yet.

**Model version doesn't update without API restart** — by design. The
FastAPI caches the Production model at startup. If `pipeline.py` promotes
a new version, restart `uvicorn` and it picks up the new one.

**CORS errors in production** — shouldn't happen, because the browser only
talks to the Express origin. If you see them anyway, something upstream is
bypassing the proxy. Check `VITE_API_URL` in the client's `.env` — if it
points straight at the FastAPI, it's wrong.

### 21.10 Security review

The integration respects the following constraints:

- **Auth**: admin-only, enforced at the Express layer via the same JWT
  middleware that protects other admin routes.
- **Secrets**: `ML_API_URL` is server-side only. Never shipped to the
  browser.
- **No direct DB access**: the admin UI never touches Neon directly — it
  always goes through the ML API, which uses read-only queries.
- **No user-controlled inputs reach SQL**: `:userId` is passed to FastAPI
  via URL; FastAPI uses parameterized queries (via SQLAlchemy `text()`).
- **No raw model outputs in JS**: the model probabilities, risk factors,
  and recommendations are pre-computed server-side. The client just renders.

### 21.11 Deploying to production

Minimal steps:

1. **Deploy the FastAPI** (Render, Fly, a VPS). Python 3.12, `pip install -r
   requirements.txt`, start command
   `uvicorn serving.api:app --host 0.0.0.0 --port $PORT`.
   Env vars: `DATABASE_URL`, optionally `MLFLOW_TRACKING_URI`.
2. **Ship `mlruns/` with the container** (simple) or point to a remote
   MLflow server (proper). Without one of these, the registry will be empty
   and the API will fall back to the local joblib file.
3. **Set `ML_API_URL`** on the Render Taalmaster server to the deployed
   FastAPI URL.
4. **Done.** No React changes needed — the admin dashboard already hits
   `/api/ml/*` via the proxy.

---

## 21B. Continuous Integration (GitHub Actions)

Every pull request and every push to `main` runs the [CI workflow](.github/workflows/ci.yml) on GitHub Actions' free tier. The badge on the repo homepage reflects the status of the most recent run on `main`.

### What CI does

| Step | Purpose |
|---|---|
| `pip install -r requirements.txt` | Same dep set as local, with pip cache for speed |
| `pytest tests/test_feature_engineering.py -v` | Unit tests — locks in temporal-split invariants |
| `pytest tests/test_pipeline_integration.py -v` | Integration tests — full pipeline on synthetic data |
| `dvc dag` | Confirms `dvc.yaml` + `dvc.lock` + `params.yaml` parse consistently |
| `python -c "import yaml; yaml.safe_load(open('params.yaml'))"` | Catches param-file typos early |
| Smoke import of all 14 modules | Catches circular imports / missing deps / rename typos that lint-only CIs miss |

Total runtime: ~3 minutes with cache, ~5 minutes on first run.

### What CI does NOT do

- **No live DB access.** `DATABASE_URL` is set to a dummy Postgres URL in the CI job env. Unit + integration tests use synthetic data, so no real connection is attempted. To run tests against Neon in CI, add `DATABASE_URL` as a repository secret and reference it via `${{ secrets.DATABASE_URL }}`.
- **No model retraining**. The integration tests train XGBoost on 200 synthetic students, which is fast (~10s). Full-scale retraining stays on the scheduler.
- **No DVC remote operations.** `dvc push/pull` would need R2/S3 credentials; CI only validates the pipeline definition, not the artifacts.

### When CI fails

The PR will show a red X and a failing "Tests + DVC validation" check. Click the check to see which step failed. Common causes:
- A new feature column added to `get_feature_columns()` but the test still asserts `len == 32`
- A renamed function breaks a transitive import → caught by the smoke import step
- A `dvc.yaml` edit that introduces a stage with an undefined dep → caught by `dvc dag`

### Adding steps later

If you want to harden further: add `ruff` or `flake8` for lint, add `mypy` for type-checks, or add a second job that builds the Docker image and runs `/health` against it. All on the free tier.

---

## 22. What's Not Done Yet

Things that are on the roadmap but not built:

### Tests

Unit tests for:
- Feature builders with edge cases (empty streaks, new accounts)
- Label computation on synthetic data
- `compute_forward_label` boundary cases

Integration test:
- Run the full pipeline against a fixture DB, assert metrics above thresholds.

### CI

- Run tests on every PR
- Run pipeline nightly
- Alert on metric regressions

### Remote MLflow + cloud DVC

Currently local-only. For a production setup:
- MLflow server on Render or a small VPS
- DVC remote on S3 or GCS

### Feature importance visualization

Plotly/Streamlit dashboard showing feature importances over time, so you can
see which signals the model is increasingly relying on.

### Model card

A structured "model card" (YAML or markdown) documenting what the model does,
what it doesn't do, known failure modes, and governance info.

### Authentication on the API

Right now the API is open. If exposed publicly, add JWT auth matching the
Taalmaster Express server.

---

*End of guide. If something's unclear or missing, open an issue or update
this file.*
