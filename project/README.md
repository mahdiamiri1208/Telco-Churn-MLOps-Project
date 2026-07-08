# Telco Customer Churn — استاندارد MLOps Pipeline

پیاده‌سازی یک Pipeline استاندارد MLOps برای پیش‌بینی ریزش مشتری (Churn) با استفاده از
دیتاست **Telco Customer Churn (IBM Dataset)**.

درس: یادگیری ماشین — استاد: دکتر باحقیقت

---

## 1. معرفی پروژه

هدف این پروژه پیش‌بینی اینکه آیا یک مشتری شرکت مخابراتی ریزش می‌کند (`churn_value = 1`)
یا مشتری باقی می‌ماند (`churn_value = 0`) است. کل فرآیند شامل:

- مدیریت نسخه‌بندی داده (v1 خام → v2 پاک‌سازی‌شده → v3 مهندسی‌شده)
- آموزش و ارزیابی چند مدل یادگیری ماشین با Cross Validation
- ثبت تمام آزمایش‌ها، پارامترها و متریک‌ها در MLflow
- استقرار بهترین مدل به‌صورت یک سرویس REST با Docker

## 2. ساختار پروژه

```
project/
├── data/
│   ├── v1/        # داده خام (بدون تغییر)
│   ├── v2/        # داده پاک‌سازی‌شده
│   └── v3/        # داده مهندسی‌شده (feature engineering)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_utils.py
├── docker/
│   ├── app.py         # FastAPI inference service
│   └── Dockerfile
├── tests/
│   └── test_pipeline.py
├── models/            # best_model.joblib, feature_columns.json, model_comparison.csv
├── reports/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## 3. نصب

```bash
python -m venv venv
source venv/bin/activate        # ویندوز: venv\Scripts\activate
pip install -r requirements.txt
```

دیتاست خام را دانلود کرده و در `data/v1/` قرار دهید (جزئیات در `data/v1/README.md`).

## 4. مدیریت نسخه داده (Data Versioning)

| نسخه | توضیح | فایل |
|------|--------|------|
| v1 | داده خام، بدون هیچ تغییری | `data/v1/*.xlsx` یا `*.csv` |
| v2 | حذف ستون‌های غیرضروری، رفع missing values، Encoding | `data/v2/telco_churn_clean.csv` |
| v3 | ساخت ویژگی جدید + نرمال‌سازی داده‌های عددی | `data/v3/telco_churn_featured.csv` |

هر مرحله در یک ماژول جدا (`preprocessing.py`, `features.py`) پیاده‌سازی شده و تغییرات
از طریق Git قابل ردیابی است.

## 5. اجرای Pipeline

```bash
# اجرای کامل: preprocessing + feature engineering + آموزش روی v3
python run_pipeline.py

# آموزش روی داده پاک‌سازی‌شده (v2) بدون feature engineering
python run_pipeline.py --dataset-version v2

# اگر v2/v3 از قبل تولید شده‌اند و فقط می‌خواهید دوباره آموزش دهید
python run_pipeline.py --skip-prep
```

مشاهده نتایج در MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# سپس مرورگر: http://localhost:5000
```

> توجه: از MLflow نسخه ۳ به بعد، backend فایلی ساده (`file:./mlruns`) در حالت
> maintenance-only قرار گرفته است؛ به همین دلیل این پروژه به‌صورت پیش‌فرض از
> `sqlite:///mlflow.db` به‌عنوان tracking store استفاده می‌کند (همچنان کاملاً
> محلی و بدون نیاز به سرور جداگانه). در صورت نیاز می‌توانید با متغیر محیطی
> `MLFLOW_TRACKING_URI` این مقدار را تغییر دهید.

## 6. مدل‌ها و ارزیابی

مدل‌های آموزش‌دیده: **Logistic Regression, Random Forest, XGBoost, CatBoost**

برای هر مدل:
- تقسیم داده به train (70%) / validation (15%) / test (15%) با stratify روی هدف
- `GridSearchCV` + `StratifiedKFold` (5-fold) روی train برای تنظیم hyperparameters
- انتخاب مدل نهایی بر اساس عملکرد روی validation
- یک ارزیابی نهایی روی test (Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix)
- تمام seedها ثابت (`SEED = 42`) برای تکرارپذیری نتایج

نتایج مقایسه‌ای مدل‌ها در `models/model_comparison.csv` و در MLflow ذخیره می‌شود.

## 7. استقرار مدل (Docker + MLflow)

```bash
docker build -t telco-churn-api -f docker/Dockerfile .
docker run -p 8000:8000 telco-churn-api
```

تست سرویس:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure_months": 12, "monthly_charges": 70.5, ...}}'
```

لیست دقیق فیچرهای موردنیاز در `models/feature_columns.json` پس از اجرای pipeline
تولید می‌شود.

## 8. تست

```bash
pytest tests/ -v
```

## 9. Git

```bash
git init
git add .
git commit -m "Initial commit: MLOps pipeline structure"
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

پیشنهاد می‌شود هر مرحله (data versioning، preprocessing، feature engineering،
training، MLflow logging، Docker deployment) در یک یا چند commit جداگانه و با پیام
مناسب ثبت شود تا تاریخچه توسعه پروژه در گیت‌هاب قابل بررسی باشد.
