"""
app.py
------
Minimal FastAPI inference service for the deployed churn model.

It loads the best model produced by train.py (models/best_model.joblib,
saved alongside models/feature_columns.json describing the exact column
order the model expects) and exposes:

    GET  /health   -> simple liveness/readiness check
    POST /predict  -> churn prediction for one customer record

The model itself is also registered in MLflow ("telco_churn_best_model");
loblib is used here for the runtime container so the image doesn't need a
live MLflow tracking server to serve predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/best_model.joblib")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Serves the best model produced by the churn-prediction MLOps pipeline.",
    version="1.0.0",
)

_model = None
_feature_columns: list[str] | None = None


def _load_artifacts() -> None:
    global _model, _feature_columns
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. Run the training pipeline first "
            "(python run_pipeline.py) so models/best_model.joblib is produced."
        )
    _model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        _feature_columns = json.load(f)


@app.on_event("startup")
def startup_event() -> None:
    _load_artifacts()


class CustomerFeatures(BaseModel):
    """
    Generic feature payload: a flat dict of {feature_name: value} matching
    the columns the model was trained on (see models/feature_columns.json).
    Using a permissive dict here (instead of one rigid field per column)
    keeps the API in sync automatically whenever feature engineering changes.
    """

    features: dict[str, Any] = Field(
        ..., description="Mapping of feature name -> value, matching feature_columns.json"
    )


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    label: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "n_features_expected": len(_feature_columns) if _feature_columns else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    if _model is None or _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    missing = [c for c in _feature_columns if c not in payload.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}",
        )

    row = pd.DataFrame([{c: payload.features[c] for c in _feature_columns}])

    prediction = int(_model.predict(row)[0])
    probability = float(_model.predict_proba(row)[0][1]) if hasattr(_model, "predict_proba") else float(prediction)

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        label="Churn" if prediction == 1 else "No Churn",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
