"""
mlflow_utils.py
---------------
Small wrapper around the MLflow tracking API so every module logs
experiments the same way: same experiment name, same tagging convention,
same artifact layout. Keeping this in one place means train.py / evaluate.py
never have to repeat MLflow boilerplate.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import mlflow

EXPERIMENT_NAME = "telco-churn-mlops"
# A SQLite-backed tracking store is used by default (instead of the legacy
# plain-folder "file:./mlruns" store) since recent MLflow versions put the
# file store in maintenance mode. This still lives locally in the project
# folder (mlflow.db) and needs no external service - `mlflow ui` reads it
# directly. Override with the MLFLOW_TRACKING_URI env var if you point this
# at a real tracking server instead.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def init_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str, tags: dict[str, Any] | None = None):
    init_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def log_sklearn_model(model, artifact_path: str = "model", registered_model_name: str | None = None):
    import mlflow.sklearn

    # Recent MLflow versions default sklearn-flavor serialization to "skops",
    # which only allows a small allow-list of trusted types and rejects
    # boosting-library estimators (XGBClassifier, CatBoostClassifier, ...)
    # with an UntrustedTypesFoundException. Since this project logs several
    # non-pure-sklearn model families through this same helper, pickle is
    # used instead so every model family (LogisticRegression, RandomForest,
    # XGBoost, CatBoost) can be logged/loaded the same way.
    mlflow.sklearn.log_model(
        model,
        artifact_path,
        serialization_format="pickle",
        registered_model_name=registered_model_name,
    )


def get_best_run(experiment_name: str = EXPERIMENT_NAME, metric: str = "test_f1"):
    """Return the run with the highest value of `metric` for this experiment."""
    init_mlflow(experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found for this experiment yet.")
    return runs[0]
