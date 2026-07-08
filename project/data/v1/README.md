# data/v1 — Raw dataset (version 1)

This folder holds the **untouched raw** IBM Telco Customer Churn dataset.

1. Download the dataset (e.g. "Telco Customer Churn" — IBM sample dataset,
   available on Kaggle / IBM Community as `Telco_customer_churn.xlsx` or
   `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Place the file directly inside this folder, e.g.:
   - `data/v1/Telco_customer_churn.xlsx`, or
   - `data/v1/WA_Fn-UseC_-Telco-Customer-Churn.csv`

`src/data_loader.py` will automatically pick up whichever `.csv`/`.xlsx`
file it finds here — no code changes needed.

No values are modified at this stage; this is dataset **version 1 (v1)** as
required by the project spec ("raw data, no changes").
