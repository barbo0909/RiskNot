"""Preprocessing pipelines for RiskNot models."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES


def get_feature_groups(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split columns into categorical and numeric groups."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in columns]
    numeric = [column for column in columns if column not in categorical]
    return categorical, numeric


def build_preprocessor(columns: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """Build a preprocessing pipeline fitted only on training data."""
    categorical, numeric = get_feature_groups(columns)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", categorical_preprocessor, categorical),
        ]
    )
