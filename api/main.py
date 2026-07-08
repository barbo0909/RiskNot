"""FastAPI backend for RiskNot predictions."""

from pathlib import Path
from typing import Literal
import os

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.config import DEFAULT_ARTIFACT_PATH
from src.explain import explain_single_prediction
from src.predict import load_artifacts, predict_dataframe

load_dotenv()

app = FastAPI(title="RiskNot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_PATH = Path(DEFAULT_ARTIFACT_PATH)
artifacts_cache = None
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


class CustomerInput(BaseModel):
    LIMIT_BAL: float = Field(..., ge=0)
    SEX: Literal[1, 2]
    EDUCATION: int = Field(..., ge=1, le=6)
    MARRIAGE: int = Field(..., ge=1, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int = Field(..., ge=-2, le=8)
    PAY_2: int = Field(..., ge=-2, le=8)
    PAY_3: int = Field(..., ge=-2, le=8)
    PAY_4: int = Field(..., ge=-2, le=8)
    PAY_5: int = Field(..., ge=-2, le=8)
    PAY_6: int = Field(..., ge=-2, le=8)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


def get_artifacts():
    """Load artifacts once per API process."""
    global artifacts_cache
    if artifacts_cache is None:
        if not ARTIFACT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {ARTIFACT_PATH}. Train the model first.",
            )
        artifacts_cache = load_artifacts(ARTIFACT_PATH)
    return artifacts_cache


def supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def supabase_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def save_assessment(input_payload: dict, model_response: dict) -> dict | None:
    """Persist prediction result to Supabase when credentials are configured."""
    if not supabase_enabled():
        return None

    payload = {
        "default_probability": model_response["default_probability"],
        "risk_score": model_response["risk_score"],
        "risk_segment": model_response["risk_segment"],
        "predicted_default": model_response["predicted_default"],
        "selected_threshold": model_response.get("selected_threshold"),
        "final_model_name": model_response.get("final_model_name"),
        "input_payload": input_payload,
        "model_response": model_response,
    }
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/risk_assessments",
        headers=supabase_headers(),
        json=payload,
        timeout=4,
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


def fetch_assessments(limit: int = 25) -> list[dict]:
    """Fetch latest saved assessments from Supabase."""
    if not supabase_enabled():
        return []

    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/risk_assessments",
        headers=supabase_headers(),
        params={
            "select": "id,created_at,default_probability,risk_score,risk_segment,predicted_default,final_model_name",
            "order": "created_at.desc",
            "limit": limit,
        },
        timeout=4,
    )
    response.raise_for_status()
    return response.json()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifact_exists": ARTIFACT_PATH.exists(),
        "supabase_enabled": supabase_enabled(),
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    artifacts = get_artifacts()
    raw = pd.DataFrame([customer.model_dump()])
    try:
        prediction = predict_dataframe(raw, artifacts).iloc[0].to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_response = {
        "model_probability": float(prediction.get("model_probability", prediction["default_probability"])),
        "default_probability": float(prediction["default_probability"]),
        "policy_adjusted": bool(prediction.get("policy_adjusted", False)),
        "risk_score": int(prediction["risk_score"]),
        "risk_segment": prediction["risk_segment"],
        "predicted_default": int(prediction["predicted_default"]),
        "selected_threshold": float(artifacts.get("selected_threshold", 0.30)),
        "final_model_name": artifacts.get("final_model_name", "model"),
    }
    try:
        model_response["explanation"] = explain_single_prediction(raw, artifacts)
    except Exception:
        model_response["explanation"] = {"risk_increasing": [], "risk_decreasing": []}

    try:
        saved_row = save_assessment(customer.model_dump(), model_response)
    except Exception:
        saved_row = None
    model_response["saved"] = saved_row is not None
    model_response["assessment_id"] = saved_row.get("id") if saved_row else None
    return model_response


@app.get("/assessments")
def assessments(limit: int = 25):
    try:
        return {"items": fetch_assessments(limit=limit)}
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Supabase request failed: {exc}") from exc
