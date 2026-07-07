"""Streamlit dashboard for RiskNot."""

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_ARTIFACT_PATH, RAW_FEATURE_COLUMNS, REPORTS_DIR
from src.predict import load_artifacts, predict_dataframe, risk_from_probability

st.set_page_config(page_title="RiskNot", page_icon="Risk", layout="wide")


@st.cache_resource
def cached_artifacts(path: str):
    return load_artifacts(path)


def numeric_input(column: str, default: float):
    return st.number_input(column, value=float(default), step=1000.0)


def single_customer_form() -> pd.DataFrame:
    defaults = {
        "LIMIT_BAL": 200000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 50000,
        "BILL_AMT2": 48000,
        "BILL_AMT3": 46000,
        "BILL_AMT4": 44000,
        "BILL_AMT5": 42000,
        "BILL_AMT6": 40000,
        "PAY_AMT1": 3000,
        "PAY_AMT2": 3000,
        "PAY_AMT3": 3000,
        "PAY_AMT4": 3000,
        "PAY_AMT5": 3000,
        "PAY_AMT6": 3000,
    }

    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(RAW_FEATURE_COLUMNS):
        with columns[index % 3]:
            if feature in {"SEX", "EDUCATION", "MARRIAGE"} or feature.startswith("PAY_"):
                values[feature] = st.number_input(
                    feature, value=int(defaults[feature]), step=1
                )
            else:
                values[feature] = numeric_input(feature, defaults[feature])
    return pd.DataFrame([values])


st.sidebar.title("RiskNot")
page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Model Performance",
        "Single Customer Risk Prediction",
        "Batch CSV Prediction",
        "Fairness Diagnostics",
    ],
)

artifact_path = st.sidebar.text_input("Artifact path", str(DEFAULT_ARTIFACT_PATH))

if page == "Project Overview":
    st.title("RiskNot: Explainable Credit Default Risk")
    st.write(
        "RiskNot predicts next-month credit-card default probability and converts "
        "the result into a 0-100 risk score with Low, Medium, and High segments."
    )
    report_path = REPORTS_DIR / "final_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))

elif page == "Model Performance":
    st.title("Model Performance")
    try:
        artifacts = cached_artifacts(artifact_path)
        st.subheader("Validation Results")
        st.dataframe(artifacts.get("validation_results"))
        st.subheader("Threshold Comparison")
        st.dataframe(artifacts.get("threshold_table"))
        st.metric("Selected Threshold", artifacts.get("selected_threshold", 0.30))
    except Exception as exc:
        st.warning(f"Model artifacts are not available yet: {exc}")

elif page == "Single Customer Risk Prediction":
    st.title("Single Customer Risk Prediction")
    try:
        artifacts = cached_artifacts(artifact_path)
        raw_input = single_customer_form()
        if st.button("Predict Risk"):
            prediction = predict_dataframe(raw_input, artifacts)
            probability = float(prediction.loc[0, "default_probability"])
            risk = risk_from_probability(probability)
            col1, col2, col3 = st.columns(3)
            col1.metric("Default Probability", f"{probability:.2%}")
            col2.metric("Risk Score", f"{risk['risk_score']}/100")
            col3.metric("Risk Segment", risk["risk_segment"])
            st.dataframe(prediction)
    except Exception as exc:
        st.warning(f"Model artifacts are not available yet: {exc}")

elif page == "Batch CSV Prediction":
    st.title("Batch CSV Prediction")
    st.write("Upload a CSV with the same raw columns as the original dataset.")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    try:
        artifacts = cached_artifacts(artifact_path)
        if uploaded_file is not None:
            batch = pd.read_csv(uploaded_file)
            prediction = predict_dataframe(batch, artifacts)
            st.dataframe(prediction.head(50))
            csv_buffer = StringIO()
            prediction.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download predictions",
                csv_buffer.getvalue(),
                file_name="risknot_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

elif page == "Fairness Diagnostics":
    st.title("Fairness Diagnostics")
    st.info(
        "This page is a placeholder for group diagnostics by SEX, EDUCATION, "
        "MARRIAGE, and AGE_GROUP. This diagnostic check does not prove fairness."
    )
    st.write(
        "Real deployment would require deeper fairness, legal, compliance, and "
        "model governance review."
    )
