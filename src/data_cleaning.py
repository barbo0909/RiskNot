"""Data loading and cleaning utilities."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    DATA_CANDIDATES,
    ORIGINAL_TARGET_COLUMN,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def resolve_data_path(candidates: Iterable[Path] = DATA_CANDIDATES) -> Path:
    """Return the first existing dataset path from known project locations."""
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset not found. Checked:\n{checked}")


def load_credit_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the UCI credit-card CSV or zipped CSV."""
    data_path = Path(path) if path is not None else resolve_data_path()
    compression = "zip" if data_path.suffix == ".zip" else "infer"
    return pd.read_csv(data_path, compression=compression)


def clean_credit_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply project cleaning rules without dropping valid negative bill amounts."""
    cleaned = dataframe.copy()

    if ORIGINAL_TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned.rename(columns={ORIGINAL_TARGET_COLUMN: TARGET_COLUMN})

    if "ID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["ID"])

    if "EDUCATION" in cleaned.columns:
        cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    if "MARRIAGE" in cleaned.columns:
        cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace({0: 3})

    return cleaned


def validate_raw_columns(dataframe: pd.DataFrame, include_target: bool = False) -> list[str]:
    """Return missing raw columns required for prediction or training."""
    required = list(RAW_FEATURE_COLUMNS)
    if include_target:
        required.append(TARGET_COLUMN)
    return [column for column in required if column not in dataframe.columns]
