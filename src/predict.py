"""Inference helpers shared by the command line and the API (next phase).

Try it:   python -m src.predict
"""
from pathlib import Path

import joblib
import pandas as pd

from src.data_processing import PROJECT_ROOT
from src.preprocessing import RAW_FEATURES  # also makes the saved pipeline's custom function importable

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.train")
    return joblib.load(path)


def predict_records(model, records: list) -> list:
    """Predict for a list of patient dicts. 'confidence' is the probability of the predicted class."""
    df = pd.DataFrame(records)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    X = df[RAW_FEATURES].astype(float)
    p_disease = model.predict_proba(X)[:, 1]
    out = []
    for p in p_disease:
        label = int(p >= 0.5)
        out.append({"prediction": label,
                    "label": "heart disease" if label else "no heart disease",
                    "confidence": round(float(p if label else 1 - p), 4),
                    "probability_disease": round(float(p), 4)})
    return out


if __name__ == "__main__":
    print(predict_records(load_model(), [SAMPLE_PATIENT])[0])
