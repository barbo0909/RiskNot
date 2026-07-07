"""Evaluation helpers for classification, thresholds, calibration, and fairness."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Return key binary classification metrics."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def threshold_table(y_true, y_proba, thresholds: list[float]) -> pd.DataFrame:
    """Compare precision, recall, F1, false positives, and false negatives."""
    rows = []
    for threshold in thresholds:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def fairness_diagnostics(
    features: pd.DataFrame,
    y_true,
    y_proba,
    group_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Compute basic group diagnostics. This does not prove model fairness."""
    data = features[[group_column]].copy()
    data["actual"] = np.asarray(y_true)
    data["proba"] = np.asarray(y_proba)
    data["predicted"] = (data["proba"] >= threshold).astype(int)

    rows = []
    for group_value, group_df in data.groupby(group_column):
        tn, fp, fn, tp = confusion_matrix(
            group_df["actual"], group_df["predicted"], labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "group_column": group_column,
                "group": group_value,
                "n": len(group_df),
                "default_rate": group_df["actual"].mean(),
                "predicted_high_risk_rate": group_df["predicted"].mean(),
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
            }
        )
    return pd.DataFrame(rows)
