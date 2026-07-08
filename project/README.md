# Telco Customer Churn — Standard MLOps Pipeline

Implementation of a standard MLOps pipeline for customer churn prediction using the **IBM Telco Customer Churn Dataset**.

**Course:** Machine Learning
**Instructor:** Dr. Bahaghighat

---

# 1. Project Overview

The objective of this project is to predict whether a telecommunications customer will churn (`churn_value = 1`) or remain with the company (`churn_value = 0`).

The complete pipeline includes:

* Data versioning (raw → cleaned → feature-engineered datasets)
* Training and evaluation of multiple machine learning models using cross-validation
* Experiment tracking, parameter logging, and metric recording with MLflow
* Deployment of the best-performing model as a Dockerized REST API

---

# 2. Project Structure

```text
project/
├── data/
│   ├── v1/                     # Raw dataset (unchanged)
│   │   ├── README.md           # How to obtain/place the raw file
│   │   └── Telco_customer_churn.xlsx
│   ├── v2/                     # Cleaned dataset
│   │   └── telco_churn_clean.csv
│   └── v3/                     # Feature-engineered dataset
│       ├── telco_churn_featured.csv
│       ├── feature_columns.json
│       └── scaler.joblib
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_utils.py
├── docker/
│   ├── app.py                  # FastAPI inference service
│   └── Dockerfile
├── tests/
│   └── test_pipeline.py
├── models/                     # best_model.joblib, feature_columns.json, model_comparison.csv, confusion matrices
├── reports/
│   └── final_report.md         # Full project write-up (see section 10)
├── screenshots/                # MLflow UI + GitHub evidence for the report
├── mlruns/                     # MLflow model-registry artifacts (auto-generated)
├── mlflow.db                   # MLflow SQLite tracking store (auto-generated)
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# 3. Installation

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Download the original Telco Customer Churn dataset and place it inside the `data/v1/` directory as `data/v1/Telco_customer_churn.xlsx` (or `.csv`) — see `data/v1/README.md` for the exact source link and expected columns.

---

# 4. Data Versioning

| Version | Description                                                                                        | Output                             |
| ------- | --------------------------------------------------------------------------------------------------- | ----------------------------------- |
| **v1**  | Raw dataset, no modifications                                                                       | `data/v1/*.xlsx` or `*.csv`         |
| **v2**  | Data cleaning, missing-value handling, removal of unnecessary/leaky columns, categorical encoding   | `data/v2/telco_churn_clean.csv`     |
| **v3**  | Feature engineering and numerical feature scaling                                                   | `data/v3/telco_churn_featured.csv`  |

Each stage is implemented in a dedicated module (`preprocessing.py` and `features.py`), and every version's output file is committed to Git so the transformation from one version to the next is fully traceable and reproducible.

---

# 5. Running the Pipeline

```bash
# Execute the complete pipeline:
# preprocessing → feature engineering → training on v3
python run_pipeline.py

# Train using the cleaned dataset (v2) without feature engineering
python run_pipeline.py --dataset-version v2

# Skip preprocessing if v2/v3 datasets already exist
python run_pipeline.py --skip-prep
```

Launch the MLflow tracking UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```
http://localhost:5000
```

> **Note:** Starting from MLflow 3.x, the legacy file-based backend (`file:./mlruns`) is maintained in maintenance mode only. This project therefore uses `sqlite:///mlflow.db` as the default tracking backend. Override it with the `MLFLOW_TRACKING_URI` environment variable if you point this at a real tracking server instead.

---

# 6. Model Training and Evaluation

The following machine learning models are trained and evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

For each model, the pipeline performs:

* Stratified train/validation/test split (≈70% / 15% / 15%)
* Hyperparameter optimization using `GridSearchCV`
* Five-fold `StratifiedKFold` cross-validation on the training split
* Model selection using the validation split
* A single, final evaluation on the untouched test split, reporting:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
  * Confusion Matrix (saved as an image and logged as an MLflow artifact)
* Fixed random seed (`SEED = 42`) for reproducibility

Every run — model name, hyperparameters, dataset version, seed, and all metrics above — is logged to MLflow. This whole train → validate → test cycle is repeated independently for each dataset version (`v2` and `v3`), so results stay comparable per-version.

The comparison of all trained models for the current run is saved to:

```
models/model_comparison.csv
```

and is also logged automatically as an MLflow artifact.

---

# 7. Model Deployment (MLflow + Docker)

The best model (by test F1-score) is:

1. Registered in the MLflow Model Registry under `telco_churn_best_model` during `train.py`, together with the dataset version and metrics that produced it — giving full lineage from raw data to deployed model.
2. Persisted locally as `models/best_model.joblib` (+ `models/feature_columns.json`, describing the exact column order the model expects).

The Docker image packages a lightweight FastAPI service (`docker/app.py`) that loads these two files directly. This design was chosen deliberately over `mlflow models build-docker` so the serving container is small and does **not** need a live MLflow tracking server to answer predictions — the model is still fully traceable back to its MLflow run and registry entry via the logged `dataset_version` / `model_family` tags.

Build the Docker image (run from the project root, since the Dockerfile copies `models/` and `requirements.txt` from there):

```bash
docker build -t telco-churn-api -f docker/Dockerfile .
```

Run the inference service:

```bash
docker run -p 8000:8000 telco-churn-api
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request — **the payload must include every column listed in `models/feature_columns.json`** (the exact set depends on the dataset version and the one-hot-encoded columns produced by the pipeline, so it differs per environment). Generate the full payload from that file, e.g.:

```bash
python -c "
import json
cols = json.load(open('models/feature_columns.json'))
print(json.dumps({'features': {c: 0 for c in cols}}, indent=2))
"
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "features": {
          "tenure_months": 12,
          "monthly_charges": 70.5,
          "...": "... every other column from feature_columns.json ..."
        }
      }'
```

A `422` response listing `Missing required features` means the payload is incomplete — check `models/feature_columns.json` for the full list.

---

# 8. Running Tests

```bash
pytest tests/ -v
```

The tests in `tests/test_pipeline.py` are self-contained (synthetic in-memory data), so they run in CI without requiring the real dataset or a trained model.

---

# 9. Git Version Control

Initialize the repository:

```bash
git init
git add .
git commit -m "Initial commit: MLOps pipeline structure"

git remote add origin mahdiamiri1208
git push -u origin main
```

Use separate, descriptive commits for each major development stage, e.g.:

* `feat: add data versioning (v1 raw loader)`
* `feat: add preprocessing pipeline (v2)`
* `feat: add feature engineering (v3)`
* `feat: add model training + cross-validation`
* `feat: add MLflow experiment tracking`
* `feat: add Docker deployment`
* `docs: add final report and screenshots`

A clear, incremental commit history is part of the grading criteria — avoid a single "final version" commit.

---

# 10. Project Report

The full write-up required by the assignment (preprocessing & feature-engineering steps, all dataset versions, models/hyperparameters/results and comparison table, MLflow screenshots, GitHub commit-history screenshots, and conclusions) lives in [`reports/final_report.md`](reports/final_report.md).
