"""
evaluate.py
-----------
Shared evaluation helpers: computing the standard classification metrics
(Accuracy, Precision, Recall, F1-score, ROC-AUC) and plotting/saving a
confusion matrix image, both used by train.py for every model/version and
reusable stand-alone for ad-hoc evaluation of a saved model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers/containers/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, model=None, X=None, prefix: str = "test") -> dict:
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if model is not None and X is not None and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        metrics[f"{prefix}_roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def plot_confusion_matrix(y_true, y_pred, out_path: str | Path, title: str = "Confusion Matrix") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Quick smoke test with synthetic data.
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    print(compute_metrics(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, "models/_smoke_test_confusion_matrix.png")
