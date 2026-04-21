# -----------------------------------------------------------------------------
# Dockerfile for the taalmaster-ml FastAPI service.
#
# TARGET: the serving layer (serving/api.py). NOT the training pipeline —
# that's scheduled separately and usually runs on a different machine.
#
# WHY A MULTI-STAGE BUILD?
#   Stage 1 (builder) installs deps — it's allowed to pull compilers, build
#   wheels, and end up large. Stage 2 (runtime) copies only the built
#   site-packages + the app code, so the final image is lean.
#
# WHAT'S IN THE RUNTIME IMAGE:
#   - Python 3.12 slim
#   - Installed requirements.txt (no dev tools like pytest/dvc — the API
#     doesn't need them at runtime)
#   - Your code (serving/, inference/, transformation/, config.py, etc.)
#   - The mlruns/ and artifacts/ directories are INTENTIONALLY NOT baked in.
#     Mount them via a volume, or have the container hit a remote MLflow
#     server (see INSTRUCTIONS.md §21.11).
#
# RUN LOCALLY:
#   docker build -t taalmaster-ml .
#   docker run --rm -p 8000:8000 \
#       -e DATABASE_URL=postgresql://... \
#       -v "$(pwd)/mlruns:/app/mlruns" \
#       -v "$(pwd)/artifacts:/app/artifacts" \
#       -v "$(pwd)/data:/app/data" \
#       taalmaster-ml
#
# Or use docker-compose for a repeatable local setup (see docker-compose.yml).
# -----------------------------------------------------------------------------

# ---- Stage 1: build dependencies -------------------------------------------
FROM python:3.12-slim AS builder

# Keep apt quiet and skip man pages / docs — shrinks the image.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# gcc + libpq are needed to build a couple of wheels from source (psycopg2,
# scipy fallback). Once wheels are built they stay in the venv; runtime
# stage gets them pre-built and doesn't need gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy just requirements first for Docker layer caching — if requirements
# don't change, pip install is cached across code edits.
COPY requirements.txt .

# Install into an isolated venv so we can copy it cleanly into stage 2.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Serve cached predictions; default is fine (24h) but make it visible
    # so ops can override per environment.
    PREDICTION_CACHE_MAX_AGE_MINUTES=1440

# libpq5 is the runtime counterpart to libpq-dev (psycopg2 links against it).
# We don't need gcc at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — containers should never run as root in production.
# UID/GID 1000 matches most host user setups, which makes bind-mounts sane.
RUN groupadd --system --gid 1000 app && \
    useradd --system --uid 1000 --gid app --create-home app

# Copy the pre-built venv from stage 1.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy the application. NOT copying: tests/, mlruns/, artifacts/, data/,
# logs/, scripts/, dvc config — those are either dev-only or mounted.
# The .dockerignore file enforces this (belt and braces).
COPY --chown=app:app config.py ./
COPY --chown=app:app serving/        ./serving/
COPY --chown=app:app inference/      ./inference/
COPY --chown=app:app transformation/ ./transformation/
COPY --chown=app:app ingestion/      ./ingestion/

# Create the mount points the container expects at runtime. The volumes
# that back these live OUTSIDE the image — nothing production-sensitive
# bakes into the image.
RUN mkdir -p /app/mlruns /app/artifacts /app/data/predictions && \
    chown -R app:app /app

USER app

EXPOSE 8000

# A HEALTHCHECK lets orchestrators (Render, Kubernetes) detect a broken
# container before it serves bad traffic. Probes /health — which returns
# the registry-loaded model version or a diagnostic error.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)"

# Single-worker is fine for a prediction API that hits a local model +
# cached JSON. If you add sustained load, bump --workers or scale
# horizontally behind a load balancer.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
