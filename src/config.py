"""Project configuration for RiskNot."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "default"
ORIGINAL_TARGET_COLUMN = "default.payment.next.month"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_CANDIDATES = [
    DATA_DIR / "UCI_Credit_Card.csv",
    DATA_DIR / "UCI_Credit_Card.csv.zip",
    RAW_DATA_DIR / "UCI_Credit_Card.csv",
    RAW_DATA_DIR / "UCI_Credit_Card.csv.zip",
    PROJECT_ROOT / "UCI_Credit_Card.csv",
    PROJECT_ROOT / "UCI_Credit_Card.csv.zip",
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAYMENT_AMOUNT_COLUMNS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

RAW_FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS_COLUMNS,
    *BILL_AMOUNT_COLUMNS,
    *PAYMENT_AMOUNT_COLUMNS,
]

DEFAULT_ARTIFACT_PATH = MODELS_DIR / "risknot_artifacts.joblib"
