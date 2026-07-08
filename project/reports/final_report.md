# Final Project Report — Telco Customer Churn Prediction (MLOps Pipeline)

**Course:** Machine Learning
**Instructor:** Dr. Bahaghighat
**University:** Imam Khomeini International University

> This report follows the structure required by the assignment. Sections marked
> **TODO** need to be filled in with real numbers/screenshots from your own run —
> everything else is already filled in from the project's actual results.

---

## 1. Project Objective

The goal of this project was to design and implement a standard MLOps pipeline for predicting customer churn using the IBM Telco Customer Churn dataset. Data is managed through a structured, three-stage versioning scheme (raw → cleaned → feature-engineered). Several machine learning models are trained and evaluated, with every experiment, parameter, and metric tracked in MLflow. The best-performing model is then packaged as a deployable REST service using Docker, so training, tracking, and deployment all follow a standardized, repeatable, operational workflow.

---

## 2. Data Source & Versioning

- **File:** `data/v1/Telco_customer_churn.xlsx`
- **Rows:** 7,043
- **Target column:** `Churn Value` (normalized to `churn_value`: `1` = churned, `0` = retained)

| Version | Description | Output |
|---|---|---|
| **v1 — Raw** | Unmodified export | `data/v1/Telco_customer_churn.xlsx` — 7,043 rows, **TODO** columns |
| **v2 — Cleaned** | Dropped leaky/identifier/geo columns, fixed `total_charges`, encoded categoricals | `data/v2/telco_churn_clean.csv` — **TODO** rows × **TODO** columns |
| **v3 — Feature-engineered** | Added derived features, scaled numeric columns | `data/v3/telco_churn_featured.csv` — **TODO** rows × **TODO** columns |

**Columns dropped in v2** (identifiers, geo, or target-leakage columns — only known *after* churn already happened): `customer_id`, `count`, `country`, `state`, `city`, `zip_code`, `lat_long`, `latitude`, `longitude`, `churn_score`, `cltv`, `churn_reason`, `churn_label`.

**Missing values handled:** `total_charges` arrives as text and is blank for customers with 0-month tenure; these are coerced to `0.0`. **TODO:** number of rows affected. Any remaining numeric NaNs are median-imputed.

**New features added in v3:**
- `tenure_group_*` — tenure bucketed into `0-1yr`, `1-2yr`, `2-4yr`, `4-5yr`, `5yr+` (one-hot)
- `num_active_services` — count of active add-on services (security, backup, protection, tech support, streaming TV/movies, phone)
- `avg_monthly_spend` — `total_charges / (tenure_months + 1)`
- Numeric columns (`tenure_months`, `monthly_charges`, `total_charges`, `avg_monthly_spend`, `num_active_services`) scaled with `StandardScaler`

---

## 3. Preprocessing & Feature Engineering — Implementation Workflow

1. Load raw data from the Excel export (`src/data_loader.py`), normalizing column names to a consistent snake_case schema.
2. Clean data (`src/preprocessing.py`): drop leaky/identifier/geo columns, fix `total_charges`, binary-encode Yes/No columns, one-hot encode remaining categoricals.
3. Engineer features (`src/features.py`): tenure buckets, service-adoption count, spend ratio, then scale numeric columns with a fitted `StandardScaler` (saved to `data/v3/scaler.joblib` for reuse at inference time).
4. Train and cross-validate multiple models (`src/train.py`).
5. Track every experiment (params, metrics, confusion matrix, model artifact) with MLflow (`src/mlflow_utils.py`).
6. Persist the best model for deployment (`models/best_model.joblib`, `models/feature_columns.json`).
7. Serve predictions through a FastAPI inference service (`docker/app.py`).
8. Containerize the service with Docker (`docker/Dockerfile`).

---

## 4. Models & Evaluation Method

- **Models compared:** Logistic Regression, Random Forest, XGBoost, CatBoost
- **Cross-validation:** 5-fold `StratifiedKFold`, hyperparameters selected via `GridSearchCV` (scoring = F1)
- **Data split:** 70% train / 15% validation / 15% test (stratified on target)
- **Fixed seed:** 42 (for reproducibility across all stages)
- **Metrics reported per model:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

---

## 5. Results — Model Comparison

**Best model: Logistic Regression**
- **Test F1-score:** 0.6107
- **Test ROC-AUC:** 0.8600
- **Reason for selection:** highest test F1-score among all four evaluated models (`train.py` selects the best row from `models/model_comparison.csv` by `test_f1`, descending).

> **Note:** an earlier draft of this report listed Random Forest as the best model with F1 = 0.8635 / ROC-AUC = 0.9686. Those numbers do not match the actual `models/model_comparison.csv` output below and were most likely copied from a different run, a different dataset version, or a placeholder — they have been corrected here. Always regenerate this table directly from your own `models/model_comparison.csv` before submitting.

Comparison table (from `models/model_comparison.csv`):

| Model | best_params | val_accuracy | val_precision | val_recall | val_f1 | val_roc_auc | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | `{'C': 1.0}` | 0.8121 | 0.6798 | 0.5536 | 0.6102 | 0.8508 | 0.8070 | 0.6557 | 0.5714 | **0.6107** | **0.8600** |
| Random Forest | `{'max_depth': 12, 'n_estimators': 400}` | 0.8074 | 0.6742 | 0.5321 | 0.5948 | 0.8420 | 0.7975 | 0.6375 | 0.5464 | 0.5885 | 0.8516 |
| CatBoost | `{'depth': 4, 'iterations': 200}` | 0.8178 | 0.6913 | 0.5679 | 0.6235 | 0.8517 | 0.8004 | 0.6520 | 0.5286 | 0.5838 | 0.8577 |
| XGBoost | `{'max_depth': 3, 'n_estimators': 200}` | 0.8055 | 0.6518 | 0.5750 | 0.6110 | 0.8411 | 0.7947 | 0.6352 | 0.5286 | 0.5770 | 0.8435 |

All four models land in a fairly tight band (test F1 between 0.577 and 0.611; test ROC-AUC between 0.844 and 0.860), suggesting the dataset's remaining signal after cleaning/feature engineering is largely linearly separable — the extra capacity of the tree-based models (Random Forest, XGBoost, CatBoost) does not translate into a meaningfully better F1 here, and simple Logistic Regression edges ahead.

### Confusion Matrix Images
**TODO:** insert `models/logistic_regression_confusion_matrix.png`, `models/random_forest_confusion_matrix.png`, `models/xgboost_confusion_matrix.png`, `models/catboost_confusion_matrix.png`

### MLflow UI Screenshots
**TODO:** insert screenshots of:
- Experiments/runs list page
- Run-comparison view across models
- Model Registry page showing `telco_churn_best_model`

---

## 6. Model Deployment (MLflow + Docker)

The best model is registered in the MLflow Model Registry as `telco_churn_best_model` and persisted locally as `models/best_model.joblib` + `models/feature_columns.json`. A lightweight FastAPI service (`docker/app.py`) loads these two files directly, so the serving container does not require a live MLflow tracking server to answer predictions — the model remains fully traceable back to its MLflow run via logged `dataset_version` / `model_family` tags.

```bash
docker build -t telco-churn-api -f docker/Dockerfile .
docker run -p 8000:8000 telco-churn-api
curl http://localhost:8000/health
```

**TODO:** insert screenshots of:
- `docker build` output
- `docker run` startup log
- `/health` response
- sample `/predict` response

---

## 7. Deliverables

- Cleaned dataset: `data/v2/telco_churn_clean.csv`
- Feature-engineered dataset: `data/v3/telco_churn_featured.csv`
- Trained model: `models/best_model.joblib`
- Feature schema: `models/feature_columns.json`
- Model comparison: `models/model_comparison.csv`
- Prediction API: `docker/app.py`
- Container definition: `docker/Dockerfile`

---

## 8. Git / GitHub

**TODO:** insert screenshots of:
- `git log --oneline`
- GitHub repository page showing commit history
- Repository link: `mahdiamiri1208`

---

## 9. Conclusion

Logistic Regression achieved the best test-set performance (F1 = 0.6107, ROC-AUC = 0.8600) among the four models evaluated, narrowly ahead of CatBoost, Random Forest, and XGBoost, all of which scored within a few points of each other. This suggests the churn signal in this dataset is largely captured by linear relationships between features and the target, so the added complexity of tree-based ensembles does not yield a clear advantage here. **TODO:** if you ran the pipeline on both dataset versions, add a sentence on whether feature engineering (v3) improved results over the cleaned-only data (v2).

**Limitations:** no automated retraining trigger, no CI/CD for tests/Docker builds, no live data-drift or prediction monitoring once deployed.

**Future improvements:** add GitHub Actions to run `pytest` and build the Docker image automatically on push; add a monitoring/alerting layer for data drift and model performance decay in production; consider scheduled retraining as new labeled data arrives.
