import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    COLUMN_NAMES, TARGET, clean, load_raw, missing_value_report,
)


def test_raw_file_has_expected_shape_and_columns():
    raw = load_raw()
    assert list(raw.columns) == COLUMN_NAMES
    assert len(raw) == 303


def test_question_marks_become_missing_values():
    raw = load_raw()
    assert raw["ca"].isna().sum() == 4
    assert raw["thal"].isna().sum() == 2


def test_clean_creates_binary_target_and_drops_original_label(clean_df):
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert "num" not in clean_df.columns


def test_target_logic_on_known_severity_levels():
    raw = load_raw().head(3).copy()
    raw["num"] = [0, 1, 4]
    assert clean(raw)[TARGET].tolist() == [0, 1, 1]


def test_clean_removes_duplicate_rows():
    raw = load_raw().head(5)
    doubled = pd.concat([raw, raw], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw))


def test_only_ca_and_thal_have_missing_values(clean_df):
    report = missing_value_report(clean_df)
    assert set(report.index) == {"ca", "thal"}
    assert report["missing"].sum() == 6


def test_numeric_columns_are_complete(clean_df):
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        assert not clean_df[col].isna().any()


def test_load_raw_gives_helpful_error_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nope.data")


def test_class_balance_is_not_extreme(clean_df):
    share = clean_df[TARGET].mean()
    assert 0.35 < share < 0.65
    assert not np.isnan(share)
