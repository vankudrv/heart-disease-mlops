# Heart Disease Risk API - serving image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_model.joblib \
    LOG_FILE=""

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only code changes.
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Application code and the trained model (model_metadata.json is served by /model-info).
COPY src ./src
COPY models ./models

# Run as a non-root user with a fixed numeric id (Kubernetes can only verify "non-root" for numeric ids).
RUN useradd --uid 10001 --create-home appuser && chown -R 10001 /app
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
