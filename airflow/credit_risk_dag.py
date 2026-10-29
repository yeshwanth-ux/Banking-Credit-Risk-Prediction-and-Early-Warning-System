"""Airflow DAG for the daily credit-risk ingestion and scoring workflow."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

from src.pipeline import (
    DEFAULT_REPORT_ROOT,
    DEFAULT_SOURCE,
    batch_predict_task,
    clean_data_task,
    generate_features_task,
    ingest_source_task,
    persist_predictions_task,
    pipeline_run_key,
    refresh_reports_task,
    run_directory,
    validate_source_task,
)

DAG_ID = "credit_risk_daily_pipeline"
SOURCE_PATH = Path(os.getenv("CREDIT_RISK_SOURCE_PATH", str(DEFAULT_SOURCE)))
BATCH_LIMIT_VALUE = os.getenv("CREDIT_RISK_BATCH_LIMIT")
BATCH_LIMIT = int(BATCH_LIMIT_VALUE) if BATCH_LIMIT_VALUE else None


@dag(
    dag_id=DAG_ID,
    description="Validate, ingest, clean, engineer, score, persist, and report credit risk",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "credit-risk",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
        "execution_timeout": timedelta(minutes=45),
    },
    tags=["credit-risk", "ingestion", "mlflow", "reporting"],
)
def credit_risk_daily_pipeline():  # noqa: ANN202
    @task(task_id="validate_source")
    def validate_source():  # noqa: ANN202
        context = get_current_context()
        summary = validate_source_task(SOURCE_PATH)
        logical_date = context["logical_date"].isoformat()
        summary["logical_date"] = logical_date
        summary["pipeline_run_key"] = pipeline_run_key(
            logical_date,
            summary["source_sha256"],
        )
        return summary

    @task(task_id="ingest_source")
    def ingest_source(validation_summary: dict):  # noqa: ANN202
        return ingest_source_task(validation_summary)

    @task(task_id="clean_and_validate")
    def clean_and_validate(
        ingestion_summary: dict,
        validation_summary: dict,
    ):  # noqa: ANN202
        directory = run_directory(validation_summary["pipeline_run_key"])
        return clean_data_task(
            ingestion_summary,
            output_path=directory / "data_quality.md",
        )

    @task(task_id="generate_features")
    def generate_features(
        cleaning_summary: dict,
        validation_summary: dict,
    ):  # noqa: ANN202
        directory = run_directory(validation_summary["pipeline_run_key"])
        return generate_features_task(
            cleaning_summary,
            output_dir=directory / "features",
        )

    @task(task_id="batch_risk_prediction")
    def batch_risk_prediction(
        feature_summary: dict,
        validation_summary: dict,
    ):  # noqa: ANN202
        directory = run_directory(validation_summary["pipeline_run_key"])
        return batch_predict_task(
            feature_summary,
            output_path=directory / "predictions.json",
            max_records=BATCH_LIMIT,
        )

    @task(task_id="persist_predictions")
    def persist_predictions(
        prediction_summary: dict,
        validation_summary: dict,
    ):  # noqa: ANN202
        return persist_predictions_task(
            prediction_summary,
            idempotency_key=validation_summary["pipeline_run_key"],
        )

    @task(task_id="refresh_portfolio_reports")
    def refresh_portfolio_reports(
        persistence_summary: dict,
        validation_summary: dict,
    ):  # noqa: ANN202
        suffix = validation_summary["pipeline_run_key"].rsplit(":", maxsplit=1)[-1]
        return refresh_reports_task(
            persistence_summary,
            output_directory=DEFAULT_REPORT_ROOT / suffix,
        )

    validation = validate_source()
    ingestion = ingest_source(validation)
    cleaning = clean_and_validate(ingestion, validation)
    features = generate_features(cleaning, validation)
    predictions = batch_risk_prediction(features, validation)
    persistence = persist_predictions(predictions, validation)
    refresh_portfolio_reports(persistence, validation)


credit_risk_pipeline_dag = credit_risk_daily_pipeline()
