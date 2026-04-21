"""
STEP 5 — FastAPI Serving Layer
================================
Expose predictions via HTTP so the TaalMaster admin dashboard can consume them.

ARCHITECTURE:
  React Admin Dashboard → Express API (proxy) → THIS FastAPI → Neon DB
                                                     ↓
                                              Trained XGBoost Model

WHY A SEPARATE SERVICE?
  - Different tech stack (Python ML vs. Node.js web app)
  - Independent scaling (ML can be CPU-heavy)
  - Independent deployment (retrain model without redeploying the web app)
  - Clean separation of concerns

Usage:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

MODEL SOURCE:
    At startup (first request) the API loads whatever version is tagged
    "Production" in the MLflow registry. Re-registering a new Production
    version is picked up on the next API restart.
"""
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from inference.predict import current_model_version, current_model_source
from inference.prediction_logger import (
    ensure_log_table_exists,
    log_prediction,
    log_predictions_batch,
)
from serving.prediction_cache import get_predictions_for_serving, cache_info

app = FastAPI(
    title="TaalMaster ML API",
    description="Student dropout prediction service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://taalmaster.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _create_log_table_on_boot():
    """Ensure ml_prediction_log exists before the first request lands."""
    ensure_log_table_exists()


# ---------------------------------------------------------------------------
# Response models (Pydantic validates the shape of every response)
# ---------------------------------------------------------------------------
class RiskFactor(BaseModel):
    feature: str
    value: float
    threshold: float
    severity: float
    message: str


class StudentPrediction(BaseModel):
    user_id: int
    name: str
    email: str
    dropout_probability: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    days_since_last_activity: int
    total_quizzes: int
    words_mastered: int
    current_streak: int
    has_subscription: bool
    recommendation: str


class PredictionResponse(BaseModel):
    predictions: list[StudentPrediction]
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    model_version: str
    generated_at: str
    # "cache" = served from the most recent predictions JSON (daily pipeline).
    # "live" = full recompute on this request.
    source: str = "live"


class DropoutRiskRequest(BaseModel):
    """Body for POST /predict/dropout-risk (the diagram's canonical name)."""
    user_id: int


class SummaryResponse(BaseModel):
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_dropout_probability: float
    top_risk_factors: list[dict]
    generated_at: str
    source: str = "live"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "TaalMaster ML API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Service health check",
            "/summary": "Dropout risk summary across all students",
            "/predictions": "All student predictions (query: ?risk_level=high&limit=10)",
            "/predictions/{user_id}": "Single student prediction",
            "/docs": "Interactive Swagger UI documentation",
        },
    }


@app.get("/health")
def health_check():
    try:
        version = current_model_version()
        source = current_model_source()
        return {
            "status": "healthy",
            "model_version": version,
            "model_source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        return {
            "status": "model_not_loaded",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/predictions", response_model=PredictionResponse)
def get_predictions(
    background_tasks: BackgroundTasks,
    risk_level: str | None = Query(None, description="Filter: high, medium, low"),
    limit: int = Query(100, ge=1, le=500),
    fresh: bool = Query(False, description="Skip cache and force a live recompute"),
):
    """Get dropout risk predictions for all students.

    By default served from the latest daily snapshot (fast). Pass
    `?fresh=true` to force a live recompute (slow, ~30s) when you need
    up-to-the-minute numbers.
    """
    try:
        predictions, source = get_predictions_for_serving(force_fresh=fresh)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    if risk_level:
        predictions = [p for p in predictions if p["risk_level"] == risk_level]

    predictions = predictions[:limit]

    high = sum(1 for p in predictions if p["risk_level"] == "high")
    medium = sum(1 for p in predictions if p["risk_level"] == "medium")
    low = sum(1 for p in predictions if p["risk_level"] == "low")

    # Log AFTER response is sent — doesn't block the request.
    background_tasks.add_task(
        log_predictions_batch, predictions, current_model_version(), source, "/predictions",
    )

    return PredictionResponse(
        predictions=[StudentPrediction(**p) for p in predictions],
        total_students=len(predictions),
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        model_version=current_model_version(),
        generated_at=datetime.utcnow().isoformat(),
        source=source,
    )


@app.get("/predictions/{user_id}", response_model=StudentPrediction)
def get_student_prediction(
    user_id: int,
    background_tasks: BackgroundTasks,
    fresh: bool = Query(False),
):
    """Get dropout risk prediction for a single student.

    Also cache-first. Single-student lookups walk the cached list, so they
    inherit the same freshness as /predictions.
    """
    try:
        predictions, source = get_predictions_for_serving(force_fresh=fresh)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    match = next((p for p in predictions if p.get("user_id") == user_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Student {user_id} not found")

    background_tasks.add_task(
        log_prediction, match, current_model_version(), source, "/predictions/{user_id}",
    )

    return StudentPrediction(**match)


# Canonical inference endpoint as drawn in the architecture diagram.
# Same handler as GET /predictions/{user_id}; the POST form is the one a
# third-party system (non-browser) would call. Kept as an alias for naming
# parity with the diagram — not used by the React admin dashboard, which
# uses the GET form via the Express proxy.
@app.post("/predict/dropout-risk", response_model=StudentPrediction)
def predict_dropout_risk(request: DropoutRiskRequest, background_tasks: BackgroundTasks):
    return get_student_prediction(request.user_id, background_tasks)


@app.get("/summary", response_model=SummaryResponse)
def get_summary(fresh: bool = Query(False, description="Skip cache and recompute")):
    """Aggregate risk overview for the admin dashboard."""
    try:
        predictions, source = get_predictions_for_serving(force_fresh=fresh)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    factor_counts: dict[str, int] = {}
    for p in predictions:
        if p["risk_level"] in ("high", "medium"):
            for f in p["top_risk_factors"]:
                factor_counts[f["feature"]] = factor_counts.get(f["feature"], 0) + 1

    top_factors = [
        {"feature": k, "count": v}
        for k, v in sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    ][:10]

    return SummaryResponse(
        total_students=len(predictions),
        high_risk_count=sum(1 for p in predictions if p["risk_level"] == "high"),
        medium_risk_count=sum(1 for p in predictions if p["risk_level"] == "medium"),
        low_risk_count=sum(1 for p in predictions if p["risk_level"] == "low"),
        avg_dropout_probability=round(
            sum(p["dropout_probability"] for p in predictions) / max(len(predictions), 1), 4
        ),
        top_risk_factors=top_factors,
        generated_at=datetime.utcnow().isoformat(),
        source=source,
    )
