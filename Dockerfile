FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN addgroup --system creditrisk \
    && adduser --system --ingroup creditrisk --home /app creditrisk

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-runtime.txt

COPY --chown=creditrisk:creditrisk api ./api
COPY --chown=creditrisk:creditrisk config ./config
COPY --chown=creditrisk:creditrisk data/processed ./data/processed
COPY --chown=creditrisk:creditrisk docs ./docs
COPY --chown=creditrisk:creditrisk models ./models
COPY --chown=creditrisk:creditrisk reports ./reports
COPY --chown=creditrisk:creditrisk sql ./sql
COPY --chown=creditrisk:creditrisk src ./src

RUN mkdir -p /app/mlartifacts /app/mlflow-state /app/reports/airflow \
    && chown -R creditrisk:creditrisk /app

USER creditrisk

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
