"""
data_loader.py
---------------
Responsible for locating the raw Telco Customer Churn (IBM Dataset) file,
loading it into a pandas DataFrame, doing minimal column-name normalization
(so downstream code does not care whether the source file was the IBM
"Telco_customer_churn.xlsx" export or the plain Kaggle CSV export), and
persisting the untouched raw copy as data/v1 (dataset version 1).

Dataset version 1 (v1) = raw data, no modification of values.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/v1")

# Columns that show up across the different public exports of this dataset.
# We normalize everything to snake_case so every later module can rely on a
# single, stable naming scheme regardless of which file was dropped in v1.
COLUMN_RENAME_MAP = {
    "customerID": "customer_id",
    "CustomerID": "customer_id",
    "gender": "gender",
    "Gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "Tenure Months": "tenure_months",
    "PhoneService": "phone_service",
    "Phone Service": "phone_service",
    "MultipleLines": "multiple_lines",
    "Multiple Lines": "multiple_lines",
    "InternetService": "internet_service",
    "Internet Service": "internet_service",
    "OnlineSecurity": "online_security",
    "Online Security": "online_security",
    "OnlineBackup": "online_backup",
    "Online Backup": "online_backup",
    "DeviceProtection": "device_protection",
    "Device Protection": "device_protection",
    "TechSupport": "tech_support",
    "Tech Support": "tech_support",
    "StreamingTV": "streaming_tv",
    "Streaming TV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "Paperless Billing": "paperless_billing",
    "PaymentMethod": "payment_method",
    "Payment Method": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "Monthly Charges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Total Charges": "total_charges",
    "Churn": "churn_label",
    "Churn Label": "churn_label",
    "Churn Value": "churn_value",
    "Churn Score": "churn_score",
    "CLTV": "cltv",
    "Churn Reason": "churn_reason",
    "Count": "count",
    "Country": "country",
    "State": "state",
    "City": "city",
    "Zip Code": "zip_code",
    "Lat Long": "lat_long",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def _find_raw_file(data_dir: Path = RAW_DATA_DIR) -> Path:
    """Locate a single raw dataset file (csv or xlsx) inside data/v1."""
    candidates = sorted(glob.glob(str(data_dir / "*.csv"))) + sorted(
        glob.glob(str(data_dir / "*.xlsx"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No raw dataset file (.csv/.xlsx) found in '{data_dir}'. "
            "Download the IBM Telco Customer Churn dataset and place it there "
            "(e.g. data/v1/Telco_customer_churn.xlsx)."
        )
    return Path(candidates[0])


def load_raw_data(data_dir: str | Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load the raw dataset exactly as provided (v1 - no modification)."""
    data_dir = Path(data_dir)
    raw_path = _find_raw_file(data_dir)

    if raw_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path)

    # Only rename columns we recognize; leave anything unexpected untouched
    # so we never silently drop information from the raw version.
    df = df.rename(columns={c: COLUMN_RENAME_MAP.get(c, c) for c in df.columns})
    return df


def save_versioned_copy(df: pd.DataFrame, version: str, filename: str) -> Path:
    """Persist a DataFrame under data/<version>/<filename>."""
    out_dir = Path("data") / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    raw_df = load_raw_data()
    print(f"Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(raw_df.head())
