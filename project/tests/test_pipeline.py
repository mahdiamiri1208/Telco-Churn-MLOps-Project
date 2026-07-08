"""
test_pipeline.py
----------------
Lightweight unit tests that don't require the real dataset to be present,
so they can run in CI. They validate the pure-logic pieces of the
pipeline: cleaning transformations, feature engineering, and the metrics
helper.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics, plot_confusion_matrix
from src.features import build_features, scale_numeric
from src.preprocessing import clean_data


@pytest.fixture
def raw_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["0001-AAA", "0002-BBB", "0003-CCC"],
            "gender": ["Female", "Male", "Female"],
            "senior_citizen": [0, 1, 0],
            "partner": ["Yes", "No", "Yes"],
            "dependents": ["No", "No", "Yes"],
            "tenure_months": [1, 34, 0],
            "phone_service": ["Yes", "Yes", "No"],
            "multiple_lines": ["No", "Yes", "No phone service"],
            "internet_service": ["DSL", "Fiber optic", "DSL"],
            "online_security": ["No", "Yes", "No"],
            "online_backup": ["Yes", "No", "No"],
            "device_protection": ["No", "Yes", "No"],
            "tech_support": ["No", "No", "No"],
            "streaming_tv": ["No", "Yes", "No"],
            "streaming_movies": ["No", "Yes", "No"],
            "contract": ["Month-to-month", "Two year", "Month-to-month"],
            "paperless_billing": ["Yes", "No", "Yes"],
            "payment_method": ["Electronic check", "Mailed check", "Electronic check"],
            "monthly_charges": [29.85, 56.95, 53.85],
            "total_charges": ["29.85", "1889.5", " "],  # blank -> brand new customer
            "churn_label": ["No", "No", "Yes"],
            "churn_value": [0, 0, 1],
        }
    )


def test_clean_data_drops_id_and_encodes(raw_sample_df):
    cleaned = clean_data(raw_sample_df)

    assert "customer_id" not in cleaned.columns
    assert "churn_label" not in cleaned.columns
    assert "churn_value" in cleaned.columns
    # total_charges blank row should become 0.0, not NaN
    assert cleaned["total_charges"].isna().sum() == 0
    assert cleaned.loc[cleaned["tenure_months"] == 0, "total_charges"].iloc[0] == 0.0


def test_build_features_adds_expected_columns(raw_sample_df):
    cleaned = clean_data(raw_sample_df)
    featured = build_features(cleaned)

    assert "num_active_services" in featured.columns
    assert "avg_monthly_spend" in featured.columns
    assert any(c.startswith("tenure_group_") for c in featured.columns)


def test_scale_numeric_is_fit_transform_consistent(raw_sample_df):
    cleaned = clean_data(raw_sample_df)
    featured = build_features(cleaned)
    scaled, scaler = scale_numeric(featured, fit=True)

    # scaled numeric columns should now have ~0 mean
    assert abs(scaled["monthly_charges"].mean()) < 1e-6

    scaled_again, _ = scale_numeric(featured, fit=False, scaler=scaler)
    pd.testing.assert_frame_equal(scaled, scaled_again)


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred, prefix="test")

    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_plot_confusion_matrix_creates_file(tmp_path):
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    out_path = tmp_path / "cm.png"

    result_path = plot_confusion_matrix(y_true, y_pred, out_path)

    assert result_path.exists()
