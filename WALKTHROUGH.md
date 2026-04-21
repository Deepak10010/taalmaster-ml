# TaalMaster ML — Visual Walkthrough

A guided tour of the system, told through diagrams. Open this file in
VSCode's Markdown Preview (Ctrl+Shift+V) or push to GitHub — the Mermaid
diagrams render automatically.

---

## 1. The big picture — one glance, whole system

This is every moving part of the repo at once. Follow the arrows clockwise
from the top-left.

```mermaid
flowchart TB
    classDef source fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#1a73e8
    classDef transform fill:#e6f4ea,stroke:#34a853,stroke-width:2px,color:#137333
    classDef model fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#c5221f
    classDef registry fill:#fef7e0,stroke:#fbbc04,stroke-width:2px,color:#a56300
    classDef serving fill:#f3e8fd,stroke:#9c27b0,stroke-width:2px,color:#6a1b9a
    classDef storage fill:#e8eaed,stroke:#5f6368,stroke-width:2px,color:#3c4043
    classDef trigger fill:#fff4e5,stroke:#ff6d00,stroke-width:3px,color:#e65100

    subgraph SRC[" "]
        direction LR
        NEON[(Neon Postgres<br/>production DB)]:::source
    end

    subgraph TRAIN["TRAINING PIPELINE &nbsp;(daily at 10am)"]
        direction TB
        E[ingestion<br/>extract_all]:::transform
        F[transformation<br/>feature engineering<br/>32 features + temporal split]:::transform
        T[training<br/>XGBoost + 5-fold CV]:::model
        R[pipeline.py<br/>stage_register<br/>promote if F1 beats prod]:::registry
        E --> F --> T --> R
    end

    subgraph STATE[" "]
        direction TB
        MLFLOW[(MLflow<br/>runs + registry<br/>v4 = Production)]:::storage
        DVC[(DVC remote<br/>data snapshots)]:::storage
        ART[(artifacts/<br/>joblib files)]:::storage
    end

    subgraph SERVE["SERVING &nbsp;(always-on)"]
        direction TB
        API[FastAPI<br/>/summary /predictions<br/>/predict/dropout-risk]:::serving
        CACHE[Prediction Cache<br/>latest JSON &lt; 24h]:::serving
        LOG[Prediction Logger<br/>BackgroundTask → Neon]:::serving
        API --> CACHE
        API -.logs.-> LOG
    end

    subgraph UI["ADMIN DASHBOARD"]
        direction TB
        EXPRESS[Express proxy<br/>/api/ml/*]:::serving
        REACT[React DropoutRiskManager<br/>table + modal]:::serving
        REACT --> EXPRESS
    end

    subgraph WATCH["DRIFT MONITORING"]
        DRIFT[evaluation<br/>drift_check.py<br/>KS + chi-square tests]:::trigger
    end

    NEON -->|read tables| E
    T --> MLFLOW
    E -->|snapshot| DVC
    F -->|snapshot| DVC
    R -->|promote| MLFLOW
    MLFLOW -->|load prod model| API
    DVC -->|pull features| DRIFT
    NEON -->|fresh data| DRIFT
    NEON -->|DB state| API
    EXPRESS -->|JWT admin auth| API
    LOG -->|ml_prediction_log| NEON
    DRIFT -.alert.-> TRAIN
```

### What's happening

- **Left column** — the data enters from Neon.
- **Middle-top** — every morning at 10am, the pipeline retrains and
  potentially promotes a new Production model.
- **Middle-bottom** — MLflow, DVC, and local `artifacts/` make up the
  *state* that survives between runs.
- **Right column** — the FastAPI serves predictions to the admin
  dashboard. Every prediction is logged back into Neon so we can measure
  real-world accuracy.
- **Below-right** — the drift monitor compares today's feature
  distributions to the last training snapshot, and can trigger a retrain
  if the world has changed enough.

**Key insight**: the pipeline and the API are **decoupled**. The pipeline
produces new models; the API loads whichever one the registry says is
Production. You can update one without touching the other.

---

## 2. A day in the life of a prediction

What happens when an admin opens the "Dropout Risk" tab? This is the full
request path from their browser all the way back.

```mermaid
sequenceDiagram
    autonumber
    participant A as Admin<br/>(browser)
    participant R as React<br/>DropoutRiskManager
    participant E as Express<br/>/api/ml/*
    participant F as FastAPI<br/>taalmaster-ml
    participant C as Prediction<br/>Cache
    participant M as MLflow<br/>Registry
    participant P as data/predictions/<br/>latest JSON
    participant L as Neon<br/>ml_prediction_log

    A->>R: clicks "Dropout Risk" tab
    R->>E: GET /api/ml/summary
    E->>E: authenticate + requireAdmin
    E->>F: GET /summary
    F->>M: which model is Production?
    M-->>F: taalmaster-dropout v4
    F->>C: cached predictions?
    C->>P: read latest file
    P-->>C: 2503 rows (6h old)
    C-->>F: {predictions, source: "cache"}
    F-->>E: JSON (~50ms)
    E-->>R: JSON
    R-->>A: summary cards render

    Note over F,L: after the response is sent, logging runs in background
    F->>L: INSERT 2503 rows into ml_prediction_log
```

### What's happening

- Steps 1–3: the browser hits Express, which is same-origin, so there's
  no CORS friction. Express checks the JWT and admin role *before*
  forwarding.
- Steps 4–5: the FastAPI first asks the MLflow registry "who's
  Production?" (This only happens on the very first request; after that
  the version is cached in memory for the rest of the process.)
- Steps 6–10: the **cache layer** is the star. Instead of re-scoring
  2,500 students live (~20–30 seconds), the API reads the most recent
  JSON that `pipeline.py` wrote to disk this morning — under 50ms.
- Step 11: FastAPI returns the response, and *then* schedules the log
  writes as a background task so the user isn't waiting on them.

**The `source: "cache"` field** in the response tells the admin
dashboard that this data was served from cache (not live). If they need
fresher numbers, they can pass `?fresh=true` and pay the 30-second cost.

---

## 3. The temporal split — the single most important concept

The original version of this model scored a **perfect 1.0 F1** — which
turned out to be a bug. It was secretly just restating the rule that
defined the label. Here's how we fixed it.

```mermaid
flowchart LR
    classDef past fill:#e6f4ea,stroke:#34a853,color:#137333
    classDef future fill:#fce8e6,stroke:#ea4335,color:#c5221f
    classDef now fill:#1a73e8,color:white,stroke:#1a73e8

    subgraph BEFORE["❌ BROKEN: snapshot-time features + snapshot-time label"]
        direction LR
        NOW1((today)):::now
        PASTALL1["ALL historical data"]:::past
        PASTALL1 --> NOW1
        NOW1 -->|"label = is_inactive<br/>at this moment"| NOW1
    end

    subgraph AFTER["✅ FIXED: temporal split"]
        direction LR
        REF(("ref_date<br/>= now - 8d")):::now
        PAST2["features: data BEFORE ref_date<br/><br/>• days_since_last_activity<br/>• quiz trends<br/>• subscription status<br/>• ..."]:::past
        FUT2["label: data AFTER ref_date<br/><br/>did they go silent in<br/>the next 7 days?"]:::future
        PAST2 --> REF --> FUT2
    end
```

### What was wrong

In the broken version, the feature `days_since_last_activity` was both
in the model's input AND in the formula for the label (`is_dropout =
days_since_last_activity >= 7`). XGBoost just learned the rule. F1 = 1.0,
useless predictions.

### The fix

Split the timeline. **Features** only see data from before `ref_date`
(the past). The **label** only comes from data after `ref_date` (the
future). The model can no longer "see the answer" — it has to predict it.

- F1 dropped from a fake 1.0 to an honest 0.9757
- Medium-risk bucket finally had students in it
- Temporal holdout test confirmed generalization

For training, `ref_date = now − 8 days` so the 7-day forward window is
entirely observed. For live scoring, `ref_date = now` and we skip label
computation entirely.

---

## 4. The 32 features, grouped

Features are what turn "rows in a database" into "patterns a model can
learn from." This is where 80% of ML value lives.

```mermaid
flowchart LR
    classDef act fill:#e6f4ea,stroke:#34a853,color:#137333
    classDef quiz fill:#e8f0fe,stroke:#4285f4,color:#1a73e8
    classDef vocab fill:#fef7e0,stroke:#fbbc04,color:#a56300
    classDef writ fill:#f3e8fd,stroke:#9c27b0,color:#6a1b9a
    classDef aud fill:#fce8e6,stroke:#ea4335,color:#c5221f
    classDef acc fill:#e8eaed,stroke:#5f6368,color:#3c4043

    ACT["ACTIVITY (9)<br/>days_since_last_activity<br/>active_days_last_7 / 14 / 30<br/>longest_streak / current_streak<br/>avg_gap_between_sessions<br/>activity_trend ←★ the star"]:::act
    QUIZ["QUIZ (7)<br/>total_quizzes<br/>avg_quiz_score_pct<br/>quiz_score_trend<br/>distinct_modes / sections<br/>quizzes_last_7d / 14d"]:::quiz
    VOCAB["VOCABULARY (6)<br/>words_attempted / mastered<br/>words_learning<br/>mastery_rate<br/>avg_accuracy<br/>vocab_coverage"]:::vocab
    WRIT["WRITING (3)<br/>total_submissions<br/>avg_score<br/>score_trend"]:::writ
    AUD["AUDIO (2)<br/>audio_listened_count<br/>audio_completion_rate"]:::aud
    ACC["ACCOUNT (5)<br/>account_age_days<br/>is_trial_expired<br/>has_subscription<br/>is_subscription_cancelling<br/>days_until_sub_expires"]:::acc

    ACT --> MATRIX[("feature matrix<br/>2352 × 32")]
    QUIZ --> MATRIX
    VOCAB --> MATRIX
    WRIT --> MATRIX
    AUD --> MATRIX
    ACC --> MATRIX
    MATRIX --> MODEL[["XGBoost<br/>trained on this"]]
```

### Why these groups

- **Activity** is most predictive — a silent user is almost always a
  dropout. `activity_trend` (recent vs older ratio) catches users who
  are *fading* before they fully go silent.
- **Quiz** is second: declining scores often predict dropout before
  activity drops (frustration → quit).
- **Vocabulary & Writing** measure *investment* in the product.
- **Audio** is passive engagement — low signal but easy to produce.
- **Account** captures business-facing churn points (trial expiry,
  subscription cancellation).

---

## 5. The training pipeline (DVC DAG)

The training pipeline is declarative — `dvc.yaml` describes a DAG of
stages. Each stage has inputs, outputs, and parameters. DVC only re-runs
what's necessary.

```mermaid
flowchart TB
    classDef stage fill:#fff4e5,stroke:#ff6d00,stroke-width:2px,color:#e65100
    classDef file fill:#e8eaed,stroke:#5f6368,color:#3c4043
    classDef param fill:#e8f0fe,stroke:#4285f4,color:#1a73e8

    NEON[(Neon Postgres)]:::file
    PARAMS[params.yaml<br/>ref_offset_days<br/>window_days]:::param

    EXT[["extract<br/>(always_changed)"]]:::stage
    FEAT[["features"]]:::stage
    TRN[["train"]]:::stage

    RAW[(data/dvc/raw/<br/>6 parquet files)]:::file
    FEATS[(data/dvc/features.parquet)]:::file
    MODEL[(artifacts/<br/>dropout_model.joblib)]:::file
    METRICS[(metrics.json<br/>cv_f1_mean etc.)]:::file
    MLFLOW_LOG[(MLflow run<br/>params + metrics + artifacts)]:::file

    NEON --> EXT
    EXT --> RAW
    RAW --> FEAT
    PARAMS --> FEAT
    FEAT --> FEATS
    FEATS --> TRN
    TRN --> MODEL
    TRN --> METRICS
    TRN --> MLFLOW_LOG
```

### What's happening

- `dvc repro` walks this DAG, top to bottom.
- `extract` always re-runs (Neon is external, DVC can't hash it), but
  its *output* gets hashed. If the extracted parquet files are
  byte-identical to last time, `features` is cached — skipped.
- Changing `params.yaml` (e.g. `window_days: 7 → 14`) re-runs `features`
  and `train` but skips `extract`.
- `metrics.json` is small and tracked in git. Running `dvc metrics diff`
  shows how tonight's metrics differ from the last commit.

### The complementary ad-hoc pipeline

`pipeline.py` is the **scheduled** version of the same flow. It doesn't
use DVC — it writes timestamped snapshots to `data/raw/YYYYMMDD/` and
lets MLflow handle experiment tracking. Windows Task Scheduler runs it
daily at 10am.

Two pipelines coexist because they serve different use cases:
- `dvc repro` — iterating on features/params; compare runs via
  `dvc exp show`
- `python pipeline.py` — scheduled retraining + registry promotion

---

## 6. The model lifecycle — registry states

A new model doesn't just appear in production. It moves through states.

```mermaid
stateDiagram-v2
    [*] --> Registered: train_model + log_model
    Registered --> None_Stage: register_model

    state "No stage (new)" as None_Stage
    state "Production (live)" as Production
    state "Archived (retired)" as Archived

    None_Stage --> Production: beats current prod on cv_f1
    None_Stage --> Archived: does not beat — manual archive
    Production --> Archived: replaced by a better model

    Production --> [*]: serves predictions

    note right of Production
        Only ONE version can be
        in Production at a time.
        The FastAPI loads this.
    end note

    note left of Archived
        v1, v2 (leakage bugs)
        are here.
        v3 moved here when v4 was
        promoted.
    end note
```

### What's happening

- Every training run is **registered** as a new version of the named
  model `taalmaster-dropout`.
- `pipeline.py → stage_register` compares its CV F1 to the current
  Production version's F1. Better → promote and archive the old one.
- The API only ever loads the Production version. Promoting a new
  version + restarting the API is how deployments work.
- History stays in the registry forever. `v1` (leaky) and `v4` (current)
  both live there, so you can roll back to any past version by setting
  its stage to Production.

### Current state

| Version | CV F1 | Stage | Notes |
|---|---|---|---|
| v1 | 1.0000 | Archived | ❌ Leakage bug |
| v2 | 1.0000 | Archived | ❌ Partial fix attempt |
| v3 | 0.9753 | Archived | ✅ First honest model |
| v4 | 0.9757 | **Production** | ✅ Currently serving |

---

## 7. Serving architecture (with cache + logging)

This is what happens every time a prediction request lands. Colored
paths show the fast path (cache hit) vs the slow path (live compute).

```mermaid
flowchart TB
    classDef fast fill:#e6f4ea,stroke:#34a853,stroke-width:3px,color:#137333
    classDef slow fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#c5221f
    classDef bg fill:#fef7e0,stroke:#fbbc04,color:#a56300
    classDef data fill:#e8eaed,stroke:#5f6368,color:#3c4043

    REQ([HTTP request]):::fast
    ENDPOINT{"/summary<br/>/predictions<br/>/predict/dropout-risk"}
    FRESH{"?fresh=true<br/>in query?"}
    CACHE{"cached JSON<br/>< 24h old?"}

    READ[read<br/>data/predictions/<br/>latest_*.json]:::fast
    LIVE[live compute:<br/>extract_all<br/>+ predict_all_students]:::slow

    RESP([return<br/>source: cache &nbsp;|&nbsp; live]):::fast
    BG[["BackgroundTask<br/>(after response sent)"]]:::bg
    LOG[(Neon<br/>ml_prediction_log)]:::data

    REQ --> ENDPOINT
    ENDPOINT --> FRESH
    FRESH -->|no| CACHE
    FRESH -->|yes| LIVE
    CACHE -->|yes| READ
    CACHE -->|no| LIVE
    READ --> RESP
    LIVE --> RESP
    RESP --> BG
    BG --> LOG
```

### What's happening

- **Fast path (green)**: most requests. A daily-refreshed JSON file is
  on disk; reading and deserializing it is <50ms.
- **Slow path (red)**: either `?fresh=true` or the cache is stale. Full
  DB extract + model scoring for 2,500 students ≈ 20-30 seconds.
- **Background task (yellow)**: every response, regardless of path,
  triggers a batch INSERT into `ml_prediction_log` on Neon. This runs
  *after* the response is sent, so the user isn't waiting on it.

### Why this matters

Logging every served prediction is how you measure real-world accuracy
*later*. After 2 weeks, join the log with `study_streaks` on `user_id`
and ask: "of the students we flagged as high-risk, how many actually
went silent?" That's the only honest measure of production performance.

---

## 8. What's local vs cloud (current + recommended)

You asked about staying cheap — keep compute local, push state to cloud
storage. This is the target topology.

```mermaid
flowchart LR
    classDef laptop fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#1a73e8
    classDef cloud fill:#fef7e0,stroke:#fbbc04,stroke-width:2px,color:#a56300
    classDef free fill:#e6f4ea,stroke:#34a853,color:#137333

    subgraph LAPTOP["YOUR LAPTOP (compute)"]
        direction TB
        SCHED[Windows Task Scheduler<br/>daily 10am]:::laptop
        PIPE[python pipeline.py]:::laptop
        API[uvicorn serving.api]:::laptop
        MLUI[mlflow ui<br/>local dashboard]:::laptop
        SCHED --> PIPE
    end

    subgraph CLOUD["CLOUD (state + monitoring, all free tier)"]
        direction TB
        NEON2[(Neon Postgres<br/>source data +<br/>prediction logs)]:::free
        R2[(Cloudflare R2<br/>DVC data snapshots<br/>MLflow artifacts)]:::free
        TUN[Cloudflare Tunnel<br/>exposes laptop API]:::free
        UP[Uptime Robot<br/>free pings]:::free
        BS[Better Stack<br/>error logs]:::free
    end

    subgraph DEPLOYED["DEPLOYED TAALMASTER (Vercel / Render)"]
        direction TB
        DASH[React admin dashboard<br/>+ Express]:::cloud
    end

    PIPE -.reads.-> NEON2
    PIPE -.pushes snapshots.-> R2
    PIPE -.logs runs.-> MLUI
    MLUI -.stores artifacts.-> R2
    API -.reads latest model.-> R2
    API -.reads/writes data.-> NEON2
    API -.proxied via.-> TUN
    TUN -.HTTPS.-> DASH
    API -.health checks.-> UP
    API -.error logs.-> BS
```

### What's happening

- **Laptop**: every piece of Python that uses CPU. The scheduled task,
  the training pipeline, the FastAPI, the MLflow UI — all local, zero
  cloud compute cost.
- **Cloud (free)**:
  - Neon — already yours, already free.
  - **R2** — stores DVC data and MLflow artifacts. 10GB free forever,
    no egress fees. If your laptop dies, all your data and models are
    safe on R2.
  - **Cloudflare Tunnel** — creates a secure HTTPS URL for your laptop
    API so a deployed Taalmaster admin dashboard can reach it without
    port forwarding.
  - **Uptime Robot** + **Better Stack** — free-tier monitoring/alerting.

**Total monthly cost for this topology: $0.** Even at 10× growth.

### The one non-obvious piece: Cloudflare Tunnel

Your deployed admin dashboard runs in the cloud. Your API runs on your
laptop. Without a tunnel, the cloud can't reach your laptop. Cloudflare
Tunnel solves this: it creates an encrypted connection FROM your laptop
TO Cloudflare, and Cloudflare exposes a public URL that forwards to it.

```
Admin browser → taalmaster.vercel.app (Vercel)
  → /api/ml/* → ml.yourdomain.com (Cloudflare)
    → (encrypted tunnel) → localhost:8000 (your laptop)
```

No firewall changes. No public port. Still free.

---

## 9. The retrain trigger — scheduled + drift

The architecture diagram you showed me at the start had two retrain
triggers: **Scheduled** and **Drift**. Both are built.

```mermaid
flowchart TB
    classDef sched fill:#e8f0fe,stroke:#4285f4,color:#1a73e8
    classDef drift fill:#fce8e6,stroke:#ea4335,color:#c5221f
    classDef action fill:#fff4e5,stroke:#ff6d00,stroke-width:2px,color:#e65100

    CLOCK(("⏰ 10am daily")):::sched
    SCHED_RUN[run_scheduled.bat<br/>python pipeline.py]:::sched

    ALARM(("🚨 drift check")):::drift
    DRIFT_RUN[python -m evaluation.drift_check<br/>KS + chi-square per feature]:::drift
    DRIFT_DECIDE{any feature<br/>p < 0.01?}
    DRIFT_TRIGGER[automatic retrain]:::action

    TRAIN[pipeline.py<br/>full retrain + promote]:::action

    CLOCK --> SCHED_RUN --> TRAIN

    ALARM --> DRIFT_RUN --> DRIFT_DECIDE
    DRIFT_DECIDE -->|yes| DRIFT_TRIGGER
    DRIFT_DECIDE -->|no| NOOP([no action])
    DRIFT_TRIGGER --> TRAIN

    TRAIN --> PROMOTE[/new version → Production<br/>if CV F1 improves/]:::action
```

### What's happening

- **Scheduled path (blue)**: the simple one. Windows Task Scheduler
  fires the batch file every morning at 10am, regardless of anything.
- **Drift path (red)**: compares today's feature distributions against
  the most recent training snapshot. If enough features have shifted
  (p < 0.01 on the per-feature statistical test), trigger a retrain.
  Useful on top of scheduled runs when conditions change suddenly —
  e.g., right after a marketing campaign or a pricing change.

### How to chain them

```bash
python -m evaluation.drift_check --fail-on-drift \
    || python pipeline.py
```

Translation: "run the drift check. If drift is found (non-zero exit
code), run the full pipeline. If no drift, do nothing." This is the
smart version of scheduled retraining — retrain only when something
actually changed.

---

## 10. Where to go from here

Each diagram above maps to one or more files in the repo. To dive
deeper:

| Topic | Start here |
|---|---|
| Pipeline code | [pipeline.py](pipeline.py), [INSTRUCTIONS.md §15](INSTRUCTIONS.md) |
| Temporal split | [transformation/feature_engineering.py](transformation/feature_engineering.py), [INSTRUCTIONS.md §8](INSTRUCTIONS.md) |
| DVC stages | [dvc.yaml](dvc.yaml), [scripts/](scripts/), [INSTRUCTIONS.md §14](INSTRUCTIONS.md) |
| Training internals | [training/train.py](training/train.py) |
| Registry management | [pipeline.py](pipeline.py) → stage_register |
| API endpoints | [serving/api.py](serving/api.py) |
| Caching strategy | [serving/prediction_cache.py](serving/prediction_cache.py) |
| Prediction logging | [inference/prediction_logger.py](inference/prediction_logger.py) |
| Drift detection | [evaluation/drift_check.py](evaluation/drift_check.py) |
| Admin dashboard wiring | [server/src/routes/ml.js](../taalmaster/server/src/routes/ml.js), [client/.../DropoutRiskManager.jsx](../taalmaster/client/src/components/admin/DropoutRiskManager.jsx) |

Or just read [INSTRUCTIONS.md](INSTRUCTIONS.md) end-to-end — it's the
written version of this walkthrough, section by section.
