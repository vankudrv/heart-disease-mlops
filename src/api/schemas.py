"""Request and response models for the prediction API (validated by pydantic)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


class PatientRecord(BaseModel):
    """One patient. Field meanings follow the UCI Heart Disease documentation.

    `ca` and `thal` are optional because the original data has missing values there;
    the model pipeline imputes them (median / most frequent), exactly as in training.
    """
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Optional[int] = Field(None, description="Thalassemia: 3 normal, 6 fixed, 7 reversible")

    @field_validator("thal")
    @classmethod
    def thal_must_be_known_code(cls, value):
        if value is not None and value not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., description="Model probability of the predicted class")
    probability_disease: float
    model: str
    request_id: str
