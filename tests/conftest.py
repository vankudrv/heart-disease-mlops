"""Shared fixtures. The model used in tests is trained on the fly (fast Logistic Regression),
so the tests never depend on mlruns/ or on a previously saved models/best_model.joblib."""
import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_processing import load_clean
from src.preprocessing import build_pipeline, prepare_xy


@pytest.fixture(scope="session")
def clean_df():
    return load_clean()


@pytest.fixture(scope="session")
def xy(clean_df):
    return prepare_xy(clean_df)


@pytest.fixture(scope="session")
def trained_pipeline(xy):
    X, y = xy
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=42))
    return pipeline.fit(X, y)


@pytest.fixture(scope="session")
def model_path(trained_pipeline, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "best_model.joblib"
    joblib.dump(trained_pipeline, path)
    return path
