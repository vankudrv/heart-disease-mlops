# Heart Disease Risk Prediction - MLOps Assignment 01

End-to-end ML project (BITS Pilani, AIMLCZG523): predict the presence of heart disease from
patient health data using the UCI Heart Disease (Cleveland) dataset, then track, test, containerise,
deploy and monitor the model.

> Status: Phase 1 done (data + EDA). Later phases add training/MLflow, API, tests/CI, Docker, Kubernetes, monitoring.

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

## Project layout

```
data/raw, data/processed   raw UCI file and cleaned CSV
scripts/download_data.py   dataset download script
src/data_processing.py     loading + cleaning (no leakage-prone steps)
notebooks/01_eda.ipynb     exploratory data analysis
reports/figures/           saved plots
tests/, k8s/, monitoring/, .github/workflows/   added in later phases
screenshots/               evidence for the report
```

## Data attribution
Hungarian Institute of Cardiology (A. Janosi), University Hospital Zurich (W. Steinbrunn),
University Hospital Basel (M. Pfisterer), V.A. Medical Center Long Beach and Cleveland Clinic
Foundation (R. Detrano). Donor: David W. Aha (1988).
