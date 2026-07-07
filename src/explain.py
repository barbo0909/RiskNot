"""SHAP explainability utilities."""

import numpy as np
import pandas as pd


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
