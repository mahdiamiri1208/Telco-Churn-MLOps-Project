# گزارش نهایی پروژه — پیش‌بینی ریزش مشتری (Telco Customer Churn)

> این فایل یک **قالب** است. بعد از اجرای واقعی pipeline روی سیستم خودتان،
> بخش‌های مشخص‌شده با «TODO» را با نتایج، اعداد و اسکرین‌شات‌های واقعی خودتان
> پر کنید.

## 1. معرفی و هدف پروژه
(خلاصه‌ای از هدف پروژه، دیتاست، و رویکرد کلی — بر اساس بخش 1 README)

## 2. دیتاست و نسخه‌بندی داده
- تعداد رکورد و ستون در نسخه خام (v1): TODO
- ستون‌های حذف‌شده در مرحله پاک‌سازی (v2) و دلیل حذف هرکدام
- تعداد missing value رفع‌شده
- ویژگی‌های جدید ساخته‌شده در v3 (`num_active_services`, `avg_monthly_spend`, `tenure_group_*`)

## 3. پیش‌پردازش و Feature Engineering
(توضیح روش‌های encoding، مدیریت total_charges، نرمال‌سازی و ...)

## 4. مدل‌ها و روش ارزیابی
- مدل‌های استفاده‌شده: Logistic Regression, Random Forest, XGBoost, CatBoost
- روش CV: StratifiedKFold(5) + GridSearchCV
- تقسیم داده: 70% train / 15% validation / 15% test
- seed ثابت: 42

## 5. نتایج مقایسه مدل‌ها
TODO: جدول از `models/model_comparison.csv` را اینجا paste کنید.

| model | val_f1 | test_f1 | test_accuracy | test_precision | test_recall | test_roc_auc |
|-------|--------|---------|----------------|-----------------|--------------|---------------|
| ...   | ...    | ...     | ...            | ...             | ...          | ...           |

### تصاویر Confusion Matrix
TODO: اسکرین‌شات‌های `models/*_confusion_matrix.png` را اینجا اضافه کنید.

### تصاویر MLflow UI
TODO: اسکرین‌شات از صفحه experiments و مقایسه run ها در MLflow.

## 6. استقرار مدل (Docker)
TODO: اسکرین‌شات از:
- خروجی `docker build`
- خروجی `docker run`
- پاسخ endpoint های `/health` و `/predict`

## 7. Git / GitHub
TODO: اسکرین‌شات از:
- `git log --oneline`
- صفحه Repository در گیت‌هاب با تاریخچه Commit ها

## 8. نتیجه‌گیری
(جمع‌بندی: کدام مدل بهترین عملکرد را داشت، چرا، محدودیت‌ها و پیشنهادهای بهبود آینده)
