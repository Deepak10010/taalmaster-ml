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
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

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


class SummaryResponse(BaseModel):
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_dropout_probability: float
    top_risk_factors: list[dict]
    generated_at: str


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
    from config import MODEL_PATH
    model_exists = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_exists else "model_not_found",
        "model_loaded": model_exists,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/predictions", response_model=PredictionResponse)
def get_predictions(
    risk_level: str | None = Query(None, description="Filter: high, medium, low"),
    limit: int = Query(100, ge=1, le=500),
):
    """Get dropout risk predictions for all students."""
    from config import MODEL_PATH
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python train.py")

    from data_extraction import extract_all
    from predict import predict_all_students

    raw_data = extract_all()
    predictions = predict_all_students(raw_data)

    if risk_level:
        predictions = [p for p in predictions if p["risk_level"] == risk_level]

    predictions = predictions[:limit]

    high = sum(1 for p in predictions if p["risk_level"] == "high")
    medium = sum(1 for p in predictions if p["risk_level"] == "medium")
    low = sum(1 for p in predictions if p["risk_level"] == "low")

    return PredictionResponse(
        predictions=[StudentPrediction(**p) for p in predictions],
        total_students=len(predictions),
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        model_version="xgboost-v1",
        generated_at=datetime.utcnow().isoformat(),
    )


@app.get("/predictions/{user_id}", response_model=StudentPrediction)
def get_student_prediction(user_id: int):
    """Get dropout risk prediction for a single student."""
    from config import MODEL_PATH
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python train.py")

    from data_extraction import extract_all
    from predict import predict_single_student

    raw_data = extract_all()
    prediction = predict_single_student(raw_data, user_id)

    if not prediction:
        raise HTTPException(status_code=404, detail=f"Student {user_id} not found")

    return StudentPrediction(**prediction)


@app.get("/summary", response_model=SummaryResponse)
def get_summary():
    """Aggregate risk overview for the admin dashboard."""
    from config import MODEL_PATH
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python train.py")

    from data_extraction import extract_all
    from predict import predict_all_students

    raw_data = extract_all()
    predictions = predict_all_students(raw_data)

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
    )
