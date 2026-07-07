"""Reusable feature engineering for training and prediction."""

import numpy as np
import pandas as pd

from src.config import BILL_AMOUNT_COLUMNS, PAYMENT_AMOUNT_COLUMNS, PAY_STATUS_COLUMNS


def safe_divide(numerator, denominator, clip_value: float = 1000.0) -> np.ndarray:
    """Divide safely, replacing invalid values with 0 and clipping extremes."""
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    result = np.zeros_like(numerator, dtype="float64")
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    result[~np.isfinite(result)] = 0.0
    return np.clip(result, -clip_value, clip_value)


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create repayment, bill, payment, and behavior features."""
    df_feat = dataframe.copy()

    delays = df_feat[PAY_STATUS_COLUMNS]
    bills = df_feat[BILL_AMOUNT_COLUMNS]
    payments = df_feat[PAYMENT_AMOUNT_COLUMNS]

    df_feat["max_delay"] = delays.max(axis=1)
    df_feat["avg_delay"] = delays.mean(axis=1)
    df_feat["recent_delay"] = df_feat["PAY_0"]
    df_feat["num_delayed_months"] = (delays > 0).sum(axis=1)
    df_feat["severe_delay_flag"] = (delays.max(axis=1) >= 2).astype(int)
    df_feat["repeated_delay_flag"] = (delays.gt(0).sum(axis=1) >= 3).astype(int)
    df_feat["delay_trend"] = df_feat["PAY_0"] - df_feat["PAY_6"]

    df_feat["avg_bill_amount"] = bills.mean(axis=1)
    df_feat["max_bill_amount"] = bills.max(axis=1)
    df_feat["min_bill_amount"] = bills.min(axis=1)
    df_feat["bill_amount_std"] = bills.std(axis=1).fillna(0)
    df_feat["recent_bill_amount"] = df_feat["BILL_AMT1"]
    df_feat["bill_trend"] = df_feat["BILL_AMT1"] - df_feat["BILL_AMT6"]

    df_feat["avg_payment_amount"] = payments.mean(axis=1)
    df_feat["max_payment_amount"] = payments.max(axis=1)
    df_feat["min_payment_amount"] = payments.min(axis=1)
    df_feat["payment_amount_std"] = payments.std(axis=1).fillna(0)
    df_feat["recent_payment_amount"] = df_feat["PAY_AMT1"]
    df_feat["payment_trend"] = df_feat["PAY_AMT1"] - df_feat["PAY_AMT6"]

    recent_delays = df_feat[["PAY_0", "PAY_2", "PAY_3"]]
    recent_bills = df_feat[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]]
    recent_payments = df_feat[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]]
    df_feat["avg_delay_last_3"] = recent_delays.mean(axis=1)
    df_feat["max_delay_last_3"] = recent_delays.max(axis=1)
    df_feat["num_delayed_last_3"] = (recent_delays > 0).sum(axis=1)
    df_feat["zero_payment_months"] = (payments == 0).sum(axis=1)
    df_feat["zero_payment_last_3"] = (recent_payments == 0).sum(axis=1)
    df_feat["avg_bill_last_3"] = recent_bills.mean(axis=1)
    df_feat["avg_payment_last_3"] = recent_payments.mean(axis=1)
    df_feat["bill_volatility_ratio"] = safe_divide(
        df_feat["bill_amount_std"], df_feat["avg_bill_amount"].abs()
    )
    df_feat["payment_volatility_ratio"] = safe_divide(
        df_feat["payment_amount_std"], df_feat["avg_payment_amount"].abs()
    )
    df_feat["payment_to_limit_ratio"] = safe_divide(
        df_feat["avg_payment_amount"], df_feat["LIMIT_BAL"]
    )
    df_feat["recent_payment_to_limit_ratio"] = safe_divide(
        df_feat["PAY_AMT1"], df_feat["LIMIT_BAL"]
    )
    df_feat["avg_utilization_last_3"] = safe_divide(
        df_feat["avg_bill_last_3"], df_feat["LIMIT_BAL"]
    )
    df_feat["utilization_change"] = safe_divide(
        df_feat["BILL_AMT1"] - df_feat["BILL_AMT6"], df_feat["LIMIT_BAL"]
    )
    df_feat["payment_coverage_last_3"] = safe_divide(
        df_feat["avg_payment_last_3"], df_feat["avg_bill_last_3"]
    )
    df_feat["months_with_negative_bill"] = (bills < 0).sum(axis=1)

    df_feat["credit_utilization_ratio"] = safe_divide(
        df_feat["avg_bill_amount"], df_feat["LIMIT_BAL"]
    )
    df_feat["recent_utilization_ratio"] = safe_divide(
        df_feat["BILL_AMT1"], df_feat["LIMIT_BAL"]
    )
    df_feat["payment_to_bill_ratio"] = safe_divide(
        df_feat["avg_payment_amount"], df_feat["avg_bill_amount"]
    )
    df_feat["recent_payment_to_bill_ratio"] = safe_divide(
        df_feat["PAY_AMT1"], df_feat["BILL_AMT1"]
    )
    df_feat["low_payment_flag"] = (df_feat["payment_to_bill_ratio"] < 0.2).astype(int)
    df_feat["high_utilization_flag"] = (
        df_feat["credit_utilization_ratio"] > 0.7
    ).astype(int)
    df_feat["recent_nonpayment_flag"] = (df_feat["PAY_AMT1"] == 0).astype(int)

    return df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
