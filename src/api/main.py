"""FastAPI service: heart-disease risk prediction with request logging and metrics.

Run locally (from the project root):
    python -m uvicorn src.api.main:app --port 8000
Then open http://127.0.0.1:8000/docs for the interactive documentation.

Endpoints
    GET  /          service information
    GET  /health    200 when the model is loaded, 503 otherwise (used by Kubernetes probes)
    GET  /model-info metadata about the loaded model
    POST /predict   JSON patient record -> prediction + confidence
    GET  /metrics   Prometheus metrics
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.logging_config import configure_logging
from src.api.schemas import PatientRecord, PredictionResponse
from src.predict import MODEL_PATH, load_model, predict_records

logger = configure_logging()

# ---- Prometheus metrics ------------------------------------------------------------------
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
CONFIDENCE = Histogram(
    "prediction_confidence", "Confidence of served predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, else 0")

KNOWN_PATHS = {"/", "/health", "/model-info", "/predict", "/metrics", "/docs", "/openapi.json"}
LOG_INPUTS = os.environ.get("LOG_INPUTS", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (path can be overridden with MODEL_PATH)."""
    model_path = Path(os.environ.get("MODEL_PATH", str(MODEL_PATH)))
    app.state.model = None
    app.state.metadata = {}
    try:
        app.state.model = load_model(model_path)
        metadata_file = model_path.with_name("model_metadata.json")
        if metadata_file.exists():
            app.state.metadata = json.loads(metadata_file.read_text())
        MODEL_LOADED.set(1)
        logger.info("model_loaded", extra={"fields": {"path": str(model_path)}})
    except Exception as exc:  # keep the service up so /health can report the problem
        MODEL_LOADED.set(0)
        logger.error("model_load_failed", extra={"fields": {"path": str(model_path),
                                                            "error": str(exc)}})
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the presence of heart disease from patient health data "
                "(UCI Heart Disease, Cleveland). For educational use only.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Give every request an id, time it, count it and write one log line."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        latency = time.perf_counter() - start
        path = request.url.path
        endpoint = path if path in KNOWN_PATHS else "other"   # keeps metric labels bounded
        REQUESTS.labels(request.method, endpoint, str(status_code)).inc()
        LATENCY.labels(endpoint).observe(latency)
        logger.info("request", extra={"fields": {
            "request_id": request_id, "method": request.method, "path": path,
            "status": status_code, "latency_ms": round(latency * 1000, 2),
            "client": request.client.host if request.client else None,
        }})


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(request: Request):
    loaded = getattr(request.app.state, "model", None) is not None
    body = {"status": "ok" if loaded else "unavailable", "model_loaded": loaded}
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.get("/model-info")
def model_info(request: Request):
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return request.app.state.metadata or {"model": "unknown"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRecord, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    record = patient.model_dump()
    result = predict_records(model, [record])[0]

    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    fields = {"request_id": request.state.request_id, "prediction": result["prediction"],
              "confidence": result["confidence"]}
    if LOG_INPUTS:                      # off by default: inputs are patient health data
        fields["input"] = record
    logger.info("prediction", extra={"fields": fields})

    model_name = request.app.state.metadata.get("model", "unknown")
    return PredictionResponse(**result, model=model_name, request_id=request.state.request_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
