"""Loading and cleaning the Cleveland Heart Disease data.

Design choice: this module only does *safe, row-independent* cleaning
(column names, '?' -> NaN, binary target, duplicates). Statistical steps such
as imputing missing values and scaling are done inside the scikit-learn
Pipeline (see src/preprocessing.py in the next phase) so that they are fitted
on training folds only. That avoids data leakage and lets the exact same
steps run again at prediction time.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

# Human-readable meaning of each column (used in plots and the report).
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
    "target": "1 = heart disease present, 0 = absent",
}


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks a missing value."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data frame."""
    df = df.copy()
    # The original label 'num' is 0 (no disease) or 1-4 (disease severity).
    # The assignment asks for a binary target, so anything > 0 becomes 1.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    df = df.drop_duplicates().reset_index(drop=True)
    # Categorical codes arrive as floats (1.0, 2.0...); make them whole numbers
    # where present, keeping <NA> for the missing ones.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("Int64")
    return df


def build_clean_dataset(raw_path: Path = RAW_FILE, out_path: Path = CLEAN_FILE) -> pd.DataFrame:
    """Raw file -> cleaned CSV on disk. Returns the cleaned data frame."""
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_clean(path: Path = CLEAN_FILE) -> pd.DataFrame:
    """Load the cleaned CSV, creating it from the raw file if needed."""
    if not Path(path).exists():
        return build_clean_dataset(out_path=path)
    df = pd.read_csv(path)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("Int64")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column (only those > 0)."""
    counts = df.isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": (counts / len(df) * 100).round(2)})
    return report[report["missing"] > 0]


if __name__ == "__main__":
    data = build_clean_dataset()
    print(f"Cleaned dataset saved to {CLEAN_FILE}")
    print(f"Shape: {data.shape}")
    print(data[TARGET].value_counts().rename({0: "no disease", 1: "disease"}))
    print("\nMissing values:\n", missing_value_report(data))
