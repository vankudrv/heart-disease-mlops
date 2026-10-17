import numpy as np
import pandas as pd
import pytest

from src.data_processing import NUMERIC_FEATURES
from src.preprocessing import (
    RAW_FEATURES, add_engineered_features, build_preprocessor, feature_names, prepare_xy,
)


def test_prepare_xy_returns_float_features_in_fixed_order(clean_df):
    X, y = prepare_xy(clean_df)
    assert list(X.columns) == RAW_FEATURES
    assert all(dtype == float for dtype in X.dtypes)
    assert set(y.unique()) == {0, 1}


def test_engineered_feature_value():
    df = pd.DataFrame({"age": [60.0], "thalach": [80.0]})
    assert add_engineered_features(df)["hr_pct_max"].iloc[0] == pytest.approx(0.5)


def test_engineered_feature_does_not_divide_by_zero_for_extreme_age():
    df = pd.DataFrame({"age": [220.0, 300.0], "thalach": [100.0, 100.0]})
    assert np.isfinite(add_engineered_features(df)["hr_pct_max"]).all()


def test_engineered_feature_does_not_modify_input():
    df = pd.DataFrame({"age": [60.0], "thalach": [80.0]})
    add_engineered_features(df)
    assert "hr_pct_max" not in df.columns


def test_preprocessor_removes_all_missing_values(xy):
    X, _ = xy
    assert X.isna().sum().sum() == 6          # the 6 raw gaps are still there ...
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()            # ... and are all filled after preprocessing


def test_numeric_features_are_standardised(xy):
    X, _ = xy
    out = build_preprocessor().fit_transform(X)
    n_numeric = len(NUMERIC_FEATURES) + 1     # + engineered hr_pct_max
    assert out[:, :n_numeric].mean(axis=0) == pytest.approx(np.zeros(n_numeric), abs=1e-6)
    assert out[:, :n_numeric].std(axis=0) == pytest.approx(np.ones(n_numeric), abs=1e-6)


def test_unseen_category_does_not_crash(xy):
    X, _ = xy
    preprocessor = build_preprocessor().fit(X)
    odd = X.head(1).copy()
    odd["thal"] = 99.0
    assert preprocessor.transform(odd).shape[0] == 1


def test_pipeline_without_engineered_feature_has_one_column_less(xy):
    X, _ = xy
    with_fe = build_preprocessor(True).fit_transform(X)
    without_fe = build_preprocessor(False).fit_transform(X)
    assert with_fe.shape[1] == without_fe.shape[1] + 1


def test_feature_names_match_transformed_width(trained_pipeline, xy):
    X, _ = xy
    width = trained_pipeline.named_steps["preprocess"].transform(X).shape[1]
    assert len(feature_names(trained_pipeline)) == width
