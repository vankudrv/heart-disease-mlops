# Heart Disease Risk Prediction - MLOps Assignment 01

![CI](https://github.com/vankudrv/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

End-to-end ML project (BITS Pilani, AIMLCZG523): predict the presence of heart disease from
patient health data using the UCI Heart Disease (Cleveland) dataset, then track, test, containerise,
deploy and monitor the model.

> Status: Phases 1-4 done (data, EDA, training, MLflow tracking, API, unit tests, CI pipeline, Docker). Later phases add Kubernetes and monitoring.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

```powershell
python scripts/download_data.py      # downloads data/raw/processed.cleveland.data from UCI
python src/data_processing.py        # writes data/processed/heart_clean.csv
```

If the download fails, get `processed.cleveland.data` from
<https://archive.ics.uci.edu/dataset/45/heart+disease> and put it in `data/raw/`.

## Run the EDA

Open `notebooks/01_eda.ipynb` in VS Code, choose the `.venv` kernel, and click **Run All**.
Figures are saved to `reports/figures/`.

## Train and track experiments

```powershell
python -m src.train                       # trains 3 models, logs everything to MLflow, saves models/best_model.joblib
python -m src.train --no-fe --no-save     # ablation: same experiment without the engineered feature
mlflow ui                                 # then open http://127.0.0.1:5000
python -m src.predict                     # sample prediction with the saved model
```

Models: Logistic Regression, Random Forest, Gradient Boosting, each tuned with 5-fold stratified
cross-validated grid search on an 80% training split; a 20% stratified hold-out set is used only for
the final test metrics. Selection rule: highest CV ROC-AUC, ties (within 0.01) broken by CV recall.

## Run the API locally

```powershell
python -m uvicorn src.api.main:app --port 8000
```
Interactive docs: <http://127.0.0.1:8000/docs>. In a second terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body (Get-Content samples/sample_patient.json -Raw)
Invoke-RestMethod http://127.0.0.1:8000/health
```
Metrics for Prometheus: <http://127.0.0.1:8000/metrics>. Every request is logged as one JSON line
to the console and to `logs/api.log` (patient inputs are only logged if `LOG_INPUTS=true`).

## Docker

```powershell
python scripts/pin_api_requirements.py        # pin the image's dependencies to your training environment
docker build -t heart-disease-api:1.0 .
docker run --rm -p 8000:8000 --name heart-api heart-disease-api:1.0
```
Then call `http://127.0.0.1:8000/predict` exactly as above. The image contains only the API code, the
trained model (`models/`) and the slim dependencies in `requirements-api.txt`; it runs as a non-root
user and has a health check.

## CI pipeline (GitHub Actions)

`.github/workflows/ci.yml` runs on every push and pull request:
1. **lint-test-train**: flake8, pytest (JUnit report), model training with MLflow tracking, model
   comparison table in the job summary, and the models/reports/mlruns uploaded as an artifact.
2. **docker-build-and-test**: builds the image, starts the container and smoke-tests `/health` and `/predict`.

A failing lint, test, training or container step fails the pipeline and shows the log of that step.

## Tests and lint

```powershell
pip install -r requirements-dev.txt
python -m pytest
python -m flake8 src tests
```

## Project layout

```
data/raw, data/processed   raw UCI file and cleaned CSV
scripts/download_data.py   dataset download script
src/data_processing.py     loading + cleaning (no leakage-prone steps)
src/preprocessing.py       feature engineering + imputation/scaling/encoding pipeline
src/train.py               tuning, evaluation, MLflow tracking, model saving
src/predict.py             load model + predict (reused by the API)
src/api/                   FastAPI app (main.py), request schemas, JSON logging
samples/                   example patient JSON files for testing the API
tests/                     pytest unit tests (data, preprocessing, model, API)
Dockerfile, .dockerignore  container image for the API
requirements-api.txt       slim pinned dependencies for the image
.github/workflows/ci.yml   CI pipeline
models/                    best_model.joblib + model_metadata.json (created by training)
notebooks/01_eda.ipynb     exploratory data analysis
reports/figures/           saved plots
tests/, k8s/, monitoring/, .github/workflows/   added in later phases
screenshots/               evidence for the report
```

## Data attribution
Hungarian Institute of Cardiology (A. Janosi), University Hospital Zurich (W. Steinbrunn),
University Hospital Basel (M. Pfisterer), V.A. Medical Center Long Beach and Cleveland Clinic
Foundation (R. Detrano). Donor: David W. Aha (1988).
