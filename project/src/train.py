"""
train.py
--------
Trains and cross-validates several classifiers on a given dataset version,
logging every run (model, hyperparameters, dataset version, seed, metrics,
confusion matrix) to MLflow, and finally selects + persists the best model
based on test F1-score.

Models used: Logistic Regression, Random Forest, XGBoost, CatBoost.

Process per dataset version:
    1. train / validation / test split (stratified on the target).
    2. K-Fold cross validation on the train split to pick/validate
       hyperparameters and get a stable estimate of performance.
    3. Validation split used to pick the final hyperparameters / model.
    4. A single evaluation on the untouched test split is reported as the
       real-world performance for that dataset version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import mlflow_utils
from src.evaluate import compute_metrics, plot_confusion_matrix

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    HAS_CATBOOST = False


SEED = 42
CV_FOLDS = 5
TARGET_COLUMN = "churn_value"
MODELS_DIR = Path("models")


def load_dataset(version: str) -> pd.DataFrame:
    path_map = {
        "v2": Path("data/v2/telco_churn_clean.csv"),
        "v3": Path("data/v3/telco_churn_featured.csv"),
    }
    if version not in path_map:
        raise ValueError(f"Unknown dataset version '{version}'. Use one of {list(path_map)}.")
    return pd.read_csv(path_map[version])


def split_data(df: pd.DataFrame, seed: int = SEED):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=seed
        # 0.176 * 0.85 ~= 0.15 -> final split is roughly 70/15/15
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _model_registry(seed: int) -> dict:
    registry = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=seed),
            "param_grid": {"C": [0.1, 1.0, 3.0]},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=seed),
            "param_grid": {"n_estimators": [200, 400], "max_depth": [6, 12, None]},
        },
    }
    if HAS_XGBOOST:
        registry["xgboost"] = {
            "estimator": XGBClassifier(random_state=seed, eval_metric="logloss"),
            "param_grid": {"n_estimators": [200, 400], "max_depth": [3, 6]},
        }
    if HAS_CATBOOST:
        registry["catboost"] = {
            "estimator": CatBoostClassifier(random_state=seed, verbose=False),
            "param_grid": {"depth": [4, 6], "iterations": [200, 400]},
        }
    return registry


def train_one_model(
    name: str,
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dataset_version: str,
    seed: int = SEED,
    cv_folds: int = CV_FOLDS,
):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    val_metrics = compute_metrics(y_val, best_model.predict(X_val), best_model, X_val, prefix="val")
    test_metrics = compute_metrics(y_test, best_model.predict(X_test), best_model, X_test, prefix="test")

    cm_path = Path("models") / f"{name}_confusion_matrix.png"
    plot_confusion_matrix(y_test, best_model.predict(X_test), cm_path, title=f"{name} - Test Confusion Matrix")

    tags = {"dataset_version": dataset_version, "model_family": name}
    with mlflow_utils.start_run(run_name=f"{name}_{dataset_version}", tags=tags):
        mlflow_utils.log_params(
            {
                "model": name,
                "dataset_version": dataset_version,
                "cv": cv_folds,
                "seed": seed,
                "random_state": seed,
                "train_test_split": "70/15/15",
                **{f"best_{k}": v for k, v in grid.best_params_.items()},
            }
        )
        mlflow_utils.log_metrics({**val_metrics, **test_metrics})
        mlflow_utils.log_artifact(str(cm_path))
        mlflow_utils.log_sklearn_model(best_model, artifact_path="model")

    return {
        "name": name,
        "model": best_model,
        "best_params": grid.best_params_,
        **val_metrics,
        **test_metrics,
    }


def run(dataset_version: str = "v3", seed: int = SEED) -> pd.DataFrame:
    df = load_dataset(dataset_version)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=seed)

    registry = _model_registry(seed)
    results = []
    for name, cfg in registry.items():
        print(f"Training {name} on dataset {dataset_version} ...")
        result = train_one_model(
            name,
            cfg["estimator"],
            cfg["param_grid"],
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            dataset_version=dataset_version,
            seed=seed,
        )
        results.append(result)

    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "model"} for r in results]
    ).sort_values("test_f1", ascending=False)

    MODELS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    best_row = results_df.iloc[0]
    best_result = next(r for r in results if r["name"] == best_row["name"])
    best_model = best_result["model"]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    feature_columns = list(X_train.columns)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    with mlflow_utils.start_run(
        run_name=f"best_model_{dataset_version}",
        tags={"dataset_version": dataset_version, "best_model": best_row["name"]},
    ):
        mlflow_utils.log_metrics({"best_test_f1": float(best_row["test_f1"])})
        mlflow_utils.log_artifact(str(MODELS_DIR / "model_comparison.csv"))
        mlflow_utils.log_sklearn_model(
            best_model, artifact_path="model", registered_model_name="telco_churn_best_model"
        )

    print("\n=== Model comparison (sorted by test F1) ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_row['name']} (test_f1={best_row['test_f1']:.4f})")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version", default="v3", choices=["v2", "v3"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(dataset_version=args.dataset_version, seed=args.seed)
