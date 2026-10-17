import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.preprocessing import build_pipeline
from src.predict import SAMPLE_PATIENT, load_model, predict_records


def test_prediction_has_expected_fields_and_ranges(trained_pipeline):
    result = predict_records(trained_pipeline, [SAMPLE_PATIENT])[0]
    assert set(result) == {"prediction", "label", "confidence", "probability_disease"}
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0


def test_label_and_confidence_agree_with_probability(trained_pipeline):
    result = predict_records(trained_pipeline, [SAMPLE_PATIENT])[0]
    p = result["probability_disease"]
    assert result["prediction"] == int(p >= 0.5)
    assert result["confidence"] == pytest.approx(max(p, 1 - p), abs=1e-4)


def test_batch_prediction_returns_one_result_per_patient(trained_pipeline):
    results = predict_records(trained_pipeline, [SAMPLE_PATIENT] * 3)
    assert len(results) == 3


def test_missing_feature_raises_clear_error(trained_pipeline):
    incomplete = {k: v for k, v in SAMPLE_PATIENT.items() if k != "chol"}
    with pytest.raises(ValueError, match="chol"):
        predict_records(trained_pipeline, [incomplete])


def test_model_accepts_missing_ca_and_thal(trained_pipeline):
    patient = dict(SAMPLE_PATIENT, ca=None, thal=None)
    assert predict_records(trained_pipeline, [patient])[0]["prediction"] in (0, 1)


def test_high_risk_patient_scores_higher_than_low_risk(trained_pipeline):
    low = dict(SAMPLE_PATIENT, age=41, cp=2, thalach=172, exang=0, oldpeak=0.2, ca=0, thal=3)
    high = dict(SAMPLE_PATIENT, age=67, cp=4, thalach=108, exang=1, oldpeak=2.5, ca=3, thal=7)
    p_low, p_high = (r["probability_disease"] for r in predict_records(trained_pipeline, [low, high]))
    assert p_high > p_low


def test_model_clearly_beats_chance(xy):
    X, y = xy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=42))
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    assert scores.mean() > 0.80


def test_saved_model_gives_identical_predictions(trained_pipeline, model_path, xy):
    X, _ = xy
    reloaded = load_model(model_path)
    assert np.allclose(trained_pipeline.predict_proba(X), reloaded.predict_proba(X))


def test_load_model_gives_helpful_error_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model(tmp_path / "missing.joblib")


def test_joblib_roundtrip_keeps_pipeline_steps(trained_pipeline, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(trained_pipeline, path)
    assert list(joblib.load(path).named_steps) == ["preprocess", "model"]
