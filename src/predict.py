"""Prediction helpers for single-customer and batch scoring."""

from pathlib import Path

import joblib
import pandas as pd

from src.config import DEFAULT_ARTIFACT_PATH, RAW_FEATURE_COLUMNS
from src.data_cleaning import clean_credit_data, validate_raw_columns
from src.feature_engineering import add_features


def risk_from_probability(default_probability: float) -> dict[str, float | int | str]:
    """Convert default probability to 0-100 score and risk segment."""
    risk_score = int(round(float(default_probability) * 100))
    risk_score = max(0, min(100, risk_score))
    if risk_score <= 30:
        segment = "Low Risk"
    elif risk_score <= 60:
        segment = "Medium Risk"
    else:
        segment = "High Risk"
    return {
        "default_probability": float(default_probability),
        "risk_score": risk_score,
        "risk_segment": segment,
    }


def load_artifacts(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict:
    """Load saved model artifacts."""
    return joblib.load(path)


def prepare_features(raw_dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Clean and engineer features using the same logic as training."""
    missing = validate_raw_columns(raw_dataframe, include_target=False)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = clean_credit_data(raw_dataframe[RAW_FEATURE_COLUMNS])
    engineered = add_features(cleaned)
    return engineered[feature_columns]


def predict_dataframe(raw_dataframe: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Predict probability, score, segment, and binary class for a dataframe."""
    model = artifacts["model"]
    threshold = artifacts.get("selected_threshold", 0.30)
    feature_columns = artifacts.get("feature_columns") or artifacts.get("raw_model_columns")

    if feature_columns is None:
        feature_columns = list(model.named_steps["preprocessor"].feature_names_in_)

    features = prepare_features(raw_dataframe, feature_columns)
    probabilities = model.predict_proba(features)[:, 1]

    result = raw_dataframe.copy()
    result["default_probability"] = probabilities
    result["risk_score"] = [risk_from_probability(p)["risk_score"] for p in probabilities]
    result["risk_segment"] = [risk_from_probability(p)["risk_segment"] for p in probabilities]
    result["predicted_default"] = (result["default_probability"] >= threshold).astype(int)
    return result
