"""Feature engineering + preprocessing, packaged as one scikit-learn Pipeline.

Everything that touches the data statistically (imputation values, scaling
means, one-hot categories) lives inside the pipeline. Therefore:
  * during cross-validation it is fitted on the training folds only (no leakage), and
  * the saved model can take a raw patient record and do all the steps itself,
    which is what makes the API reproducible.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

# The 13 raw input columns, in a fixed order (the API expects exactly these).
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENGINEERED_FEATURES = ["hr_pct_max"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain features.

    hr_pct_max = maximum heart rate achieved / age-predicted maximum (220 - age).
    A patient who cannot get close to their expected maximum heart rate during
    the exercise test is a classic warning sign, and the ratio combines two raw
    columns (thalach, age) into one clinically meaningful number.
    """
    X = X.copy()
    predicted_max = (220 - X["age"]).clip(lower=1)
    X["hr_pct_max"] = X["thalach"] / predicted_max
    return X


def build_preprocessor(use_engineered: bool = True) -> Pipeline:
    numeric_cols = NUMERIC_FEATURES + (ENGINEERED_FEATURES if use_engineered else [])

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    encode = ColumnTransformer([
        ("num", numeric_steps, numeric_cols),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])

    steps = []
    if use_engineered:
        steps.append((
            "features",
            FunctionTransformer(add_engineered_features).set_output(transform="pandas"),
        ))
    steps.append(("encode", encode))
    return Pipeline(steps)


def build_pipeline(model, use_engineered: bool = True) -> Pipeline:
    """Preprocessing + classifier as a single estimator."""
    return Pipeline([
        ("preprocess", build_preprocessor(use_engineered)),
        ("model", model),
    ])


def prepare_xy(df: pd.DataFrame):
    """Split a cleaned data frame into float features X and integer target y."""
    X = df[RAW_FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return X, y


def feature_names(pipeline: Pipeline) -> list:
    """Column names after preprocessing (used for importance plots)."""
    names = pipeline.named_steps["preprocess"].named_steps["encode"].get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]
