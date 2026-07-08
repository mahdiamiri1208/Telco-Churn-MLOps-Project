FROM python:3.11-slim

WORKDIR /app

# System deps needed by some ML libraries (xgboost/catboost wheels usually
# ship prebuilt, but libgomp is required at runtime for OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the serving container actually needs at runtime.
COPY docker/app.py .
COPY models/best_model.joblib models/best_model.joblib
COPY models/feature_columns.json models/feature_columns.json

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
