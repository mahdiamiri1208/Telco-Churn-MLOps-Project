"""
features.py
------------
Dataset version 3 (v3) = feature-engineered data.

Steps performed:
    1. Build new, business-meaningful features on top of the cleaned (v2)
       data (tenure buckets, service-adoption counts, spend ratios, ...).
    2. Scale numeric columns with StandardScaler so distance/gradient based
       models (Logistic Regression) are not dominated by raw-scale columns
       such as total_charges.

The fitted scaler is saved next to the data so evaluate/serving code can
apply the exact same transform to new data at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import save_versioned_copy

CLEAN_DATA_PATH = Path("data/v2/telco_churn_clean.csv")
SCALER_PATH = Path("data/v3/scaler.joblib")
TARGET_COLUMN = "churn_value"

# Internet/security/backup/etc. columns end up one-hot encoded upstream as
# "<col>_Yes" after preprocessing; we use them to build an adoption count.
SERVICE_ONE_HOT_PREFIXES = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

NUMERIC_COLUMNS_TO_SCALE = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_spend",
    "num_active_services",
]


def _tenure_group(tenure_months: pd.Series) -> pd.DataFrame:
    bins = [-1, 12, 24, 48, 60, 1000]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
    grouped = pd.cut(tenure_months, bins=bins, labels=labels)
    return pd.get_dummies(grouped, prefix="tenure_group").astype(int)


def _num_active_services(df: pd.DataFrame) -> pd.Series:
    count = pd.Series(0, index=df.index, dtype=int)
    for prefix in SERVICE_ONE_HOT_PREFIXES:
        yes_col = f"{prefix}_Yes"
        if yes_col in df.columns:
            count = count + df[yes_col].astype(int)
    if "phone_service" in df.columns:
        count = count + df["phone_service"].fillna(0).astype(int)
    return count


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure_months" in df.columns:
        df = pd.concat([df, _tenure_group(df["tenure_months"])], axis=1)

    df["num_active_services"] = _num_active_services(df)

    if {"total_charges", "tenure_months"}.issubset(df.columns):
        # +1 avoids divide-by-zero for brand-new customers (tenure == 0).
        df["avg_monthly_spend"] = df["total_charges"] / (df["tenure_months"] + 1)

    return df


def scale_numeric(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler | None = None):
    df = df.copy()
    cols = [c for c in NUMERIC_COLUMNS_TO_SCALE if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        df[cols] = scaler.transform(df[cols])

    return df, scaler


def run() -> Path:
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    featured_df = build_features(clean_df)
    featured_df, scaler = scale_numeric(featured_df, fit=True)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    feature_columns = [c for c in featured_df.columns if c != TARGET_COLUMN]
    with open(Path("data/v3/feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    out_path = save_versioned_copy(featured_df, "v3", "telco_churn_featured.csv")
    print(f"Saved feature-engineered dataset (v3): {featured_df.shape} -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()
