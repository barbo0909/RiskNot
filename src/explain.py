"""SHAP explainability utilities."""

import numpy as np
import pandas as pd
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
import shap

from src.predict import prepare_features


def get_feature_names(preprocessor) -> list[str]:
    """Return readable feature names from a fitted ColumnTransformer."""
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            onehot = transformer.named_steps["onehot"]
            feature_names.extend(onehot.get_feature_names_out(columns).tolist())
        elif hasattr(transformer, "get_feature_names_out"):
            try:
                feature_names.extend(transformer.get_feature_names_out(columns).tolist())
            except Exception:
                feature_names.extend(list(columns))
        else:
            feature_names.extend(list(columns))
    return feature_names


def local_shap_table(shap_values, transformed_row, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """Create a readable local SHAP contribution table."""
    row_values = np.asarray(transformed_row).reshape(-1)
    shap_row = np.asarray(shap_values).reshape(-1)
    top_idx = np.argsort(np.abs(shap_row))[-top_n:][::-1]
    explanation = pd.DataFrame(
        {
            "feature": np.asarray(feature_names)[top_idx],
            "value": row_values[top_idx],
            "shap_value": shap_row[top_idx],
        }
    )
    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0, "increases risk", "decreases risk"
    )
    return explanation


FEATURE_LABELS = {
    "LIMIT_BAL": "Higher approved credit limit",
    "AGE": "Applicant age",
    "PAY_0": "Current month repayment delay",
    "PAY_2": "Previous month repayment delay",
    "PAY_3": "Repayment delay two months ago",
    "PAY_4": "Repayment delay three months ago",
    "PAY_5": "Repayment delay four months ago",
    "PAY_6": "Repayment delay five months ago",
    "max_delay": "Worst recent repayment delay",
    "avg_delay": "Average repayment delay",
    "recent_delay": "Most recent repayment behavior",
    "num_delayed_months": "Number of delayed months",
    "severe_delay_flag": "Severe repayment delay",
    "repeated_delay_flag": "Repeated repayment delays",
    "delay_trend": "Repayment delay trend",
    "credit_utilization_ratio": "Credit utilization ratio",
    "recent_utilization_ratio": "Current statement utilization",
    "avg_utilization_last_3": "Average utilization over last three months",
    "high_utilization_flag": "High utilization",
    "payment_to_bill_ratio": "Payment-to-bill ratio",
    "recent_payment_to_bill_ratio": "Current payment-to-bill ratio",
    "payment_coverage_last_3": "Payment coverage over last three months",
    "avg_delay_last_3": "Average delay over last three months",
    "max_delay_last_3": "Worst delay over last three months",
    "num_delayed_last_3": "Delayed months in last three months",
    "zero_payment_months": "Months with no payment",
    "zero_payment_last_3": "No-payment months in last three months",
    "avg_bill_last_3": "Average statement balance over last three months",
    "avg_payment_last_3": "Average payment over last three months",
    "payment_to_limit_ratio": "Payment-to-limit ratio",
    "recent_payment_to_limit_ratio": "Current payment-to-limit ratio",
    "low_payment_flag": "Low payment behavior",
    "recent_nonpayment_flag": "Current non-payment",
    "avg_payment_amount": "Average payment amount",
    "min_payment_amount": "Lowest payment amount",
    "recent_payment_amount": "Current payment amount",
    "PAY_AMT1": "Current payment amount",
    "PAY_AMT2": "Previous payment amount",
    "PAY_AMT3": "Payment amount two months ago",
    "PAY_AMT4": "Payment amount three months ago",
    "PAY_AMT5": "Payment amount four months ago",
    "PAY_AMT6": "Payment amount five months ago",
    "BILL_AMT1": "Current statement balance",
    "BILL_AMT2": "Previous statement balance",
    "BILL_AMT3": "Statement balance two months ago",
    "BILL_AMT4": "Statement balance three months ago",
    "BILL_AMT5": "Statement balance four months ago",
    "BILL_AMT6": "Statement balance five months ago",
    "avg_bill_amount": "Average statement balance",
    "recent_bill_amount": "Current statement balance",
    "bill_amount_std": "Statement balance volatility",
    "bill_trend": "Statement balance trend",
}


def readable_feature_name(feature_name: str) -> str:
    """Convert transformed feature names into short labels for the UI."""
    clean_name = feature_name.replace("num__", "").replace("cat__", "")
    if clean_name in FEATURE_LABELS:
        return FEATURE_LABELS[clean_name]
    if clean_name.startswith("SEX_"):
        return "Gender"
    if clean_name.startswith("EDUCATION_"):
        return "Education level"
    if clean_name.startswith("MARRIAGE_"):
        return "Marital status"
    return clean_name.replace("_", " ").title()


def explain_single_prediction(raw_dataframe: pd.DataFrame, artifacts: dict, top_n: int = 4) -> dict:
    """Return local SHAP factors increasing and decreasing default risk."""
    model_pipeline = artifacts["model"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_columns = artifacts.get("feature_columns") or artifacts.get("raw_model_columns")

    if feature_columns is None:
        feature_columns = list(preprocessor.feature_names_in_)

    features = prepare_features(raw_dataframe, feature_columns)
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = get_feature_names(preprocessor)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[1][0])
    else:
        shap_array = np.asarray(shap_values)
        shap_row = shap_array[0, :, 1] if shap_array.ndim == 3 else shap_array[0]

    rows = []
    for feature_name, value, contribution in zip(feature_names, transformed[0], shap_row):
        rows.append(
            {
                "feature": readable_feature_name(feature_name),
                "raw_feature": feature_name,
                "value": float(value),
                "contribution": float(contribution),
            }
        )

    increasing = sorted(
        [row for row in rows if row["contribution"] > 0],
        key=lambda row: row["contribution"],
        reverse=True,
    )[:top_n]
    decreasing = sorted(
        [row for row in rows if row["contribution"] < 0],
        key=lambda row: row["contribution"],
    )[:top_n]

    return {
        "risk_increasing": increasing,
        "risk_decreasing": decreasing,
    }
