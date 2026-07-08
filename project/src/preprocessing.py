"""
preprocessing.py
-----------------
Dataset version 2 (v2) = cleaned data.

Steps performed:
    1. Drop columns that carry no predictive value or leak the target
       (identifiers, geo columns, churn reason/score/cltv which are
       derived *after* churn happens, etc.).
    2. Fix `total_charges` (arrives as string in the raw export and is
       blank for customers with 0 tenure) and other missing values.
    3. Encode categorical columns to numeric values.

The result is written to data/v2/telco_churn_clean.csv so that every model
trained downstream can be pointed at "dataset version 2" explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_data, save_versioned_copy

# Columns that either identify a customer or leak information that would
# not be available at prediction time (they are only known *after* churn
# already happened), plus pure geo columns that add noise for this task.
COLUMNS_TO_DROP = [
    "customer_id",
    "count",
    "country",
    "state",
    "city",
    "zip_code",
    "lat_long",
    "latitude",
    "longitude",
    "churn_score",
    "cltv",
    "churn_reason",
    "churn_label",  # redundant with churn_value (target), keep only the numeric target
]

BINARY_YES_NO_COLUMNS = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
]

MULTI_CATEGORY_COLUMNS = [
    "gender",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "payment_method",
]

TARGET_COLUMN = "churn_value"


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """total_charges is read as text and blank for brand-new customers."""
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    missing_mask = df["total_charges"].isna()
    # A missing total_charges only happens when tenure_months == 0 (brand new
    # customer, so total billed so far really is 0).
    df.loc[missing_mask, "total_charges"] = 0.0
    return df


def _encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype("Int64")
    if "senior_citizen" in df.columns:
        # Sometimes already 0/1, sometimes Yes/No depending on export.
        if df["senior_citizen"].dtype == object:
            df["senior_citizen"] = df["senior_citizen"].map({"Yes": 1, "No": 0})
        df["senior_citizen"] = df["senior_citizen"].astype("Int64")
    return df


def _encode_multi_category(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the remaining nominal categorical columns."""
    present = [c for c in MULTI_CATEGORY_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=False)
    # get_dummies produces bool columns in recent pandas -> cast to int for
    # a stable, MLflow/serialization-friendly numeric schema.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    if "total_charges" in df.columns:
        df = _coerce_total_charges(df)

    df = _encode_binary_yes_no(df)
    df = _encode_multi_category(df)

    # Drop any row still missing the target - can't train/evaluate on it.
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    # Any leftover numeric NaNs (shouldn't be many) -> median impute.
    numeric_cols = df.select_dtypes(include=["number", "Int64"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.reset_index(drop=True)
    return df


def run() -> Path:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    out_path = save_versioned_copy(clean_df, "v2", "telco_churn_clean.csv")
    print(f"Saved cleaned dataset (v2): {clean_df.shape} -> {out_path}")
    return out_path


if __name__ == "__main__":
    run()
