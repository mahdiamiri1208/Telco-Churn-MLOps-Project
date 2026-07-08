# data/v1 — Raw Dataset (unchanged)

This folder holds **dataset version 1 (v1)**: the raw IBM Telco Customer Churn export, completely unmodified.

## Source

IBM Telco Customer Churn dataset (also mirrored on Kaggle as "Telco Customer Churn"). Download either export:

* IBM export: `Telco_customer_churn.xlsx`
* Kaggle export: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## What to do

1. Download the file from the source above.
2. Place it directly inside this folder, e.g.:

   ```
   data/v1/Telco_customer_churn.xlsx
   ```

   or

   ```
   data/v1/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

3. Do **not** rename, edit, or clean the file — `src/data_loader.py` reads whichever single `.csv`/`.xlsx` file it finds in this folder and only normalizes column *names* (not values) to a consistent snake_case schema before handing it to `preprocessing.py`.

## Expected columns (either export)

`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn` (plus `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`, and geo columns in the IBM export — these are dropped later during preprocessing, in `v2`).

~7,000 customer rows, target column `Churn` / `Churn Value` (`churn_value` after normalization: `1` = churned, `0` = retained).

## Note on Git

This raw file is tracked in Git so the exact input to the pipeline is versioned and reproducible from a clean checkout. If the file is large, consider Git LFS instead of committing it directly.
