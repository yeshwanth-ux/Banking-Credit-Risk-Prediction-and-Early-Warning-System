"""Small-data task and topology tests for the Airflow credit-risk DAG."""

from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from src.model_registry import RegisteredModelBundle
from src.pipeline import (
    batch_predict_task,
    persist_predictions_task,
    pipeline_run_key,
    refresh_reports_task,
    validate_source_task,
)
from src.risk_reporting import StoredScoringRun


class FixedPipeline:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([0.4, 0.6], (len(features), 1))


def prototype_features() -> dict[str, Any]:
    return {
        "duration_months": 24,
        "credit_amount_dm": 4000.0,
        "age_years": 35,
        "credit_amount_per_month": 4000 / 24,
        "checking_account_status": 2,
        "credit_history": 2,
        "purpose": 3,
        "savings": 2,
        "employment_duration": 3,
        "installment_rate": 3,
        "personal_status_sex": 3,
        "other_debtors": 1,
        "present_residence": 3,
        "property": 2,
        "other_installment_plans": 3,
        "housing": 2,
        "number_credits": 1,
        "job": 3,
        "people_liable": 2,
        "telephone": 2,
        "foreign_worker": 1,
        "young_borrower_flag": 0,
        "long_duration_flag": 0,
    }


def model_bundle() -> RegisteredModelBundle:
    return RegisteredModelBundle(
        pipeline=FixedPipeline(),
        classification_config={
            "configuration_version": "pipeline-test-config",
            "model": {"version": "pipeline-test-model"},
            "target_definition": "prototype test target",
            "probability_label": "prototype test probability",
            "risk_bands": {"medium_threshold": 0.4, "high_threshold": 0.8},
            "warning_rules": {
                "severe_delinquency_days": 90,
                "worsening_delay_days": 15,
                "high_debt_to_income_ratio": 0.4,
                "debt_to_income_increase": 0.1,
            },
            "explanation_reference_values": prototype_features(),
        },
        model_uri="models:/PipelineTest@champion",
        registered_model_name="PipelineTest",
        registered_model_version="4",
        alias="champion",
        run_id="pipeline-test-run",
    )


class IdempotentRepository:
    def __init__(self) -> None:
        self.runs: dict[str, StoredScoringRun] = {}
        self.insert_calls = 0

    def store_results(
        self,
        results,  # noqa: ANN001
        metadata,  # noqa: ANN001
        *,
        request_type: str,
        requested_by: str,
        idempotency_key: str | None = None,
    ) -> StoredScoringRun:
        assert request_type == "batch"
        assert requested_by == "airflow-credit-risk-dag"
        assert metadata.model_version == "4"
        assert idempotency_key is not None
        existing = self.runs.get(idempotency_key)
        if existing:
            return existing
        self.insert_calls += 1
        result_list = list(results)
        stored = StoredScoringRun(
            scoring_run_id=21,
            run_reference="pipeline-run-21",
            prediction_ids=tuple(range(201, 201 + len(result_list))),
            prediction_timestamp=datetime(2026, 10, 28, 2, 0, tzinfo=UTC),
        )
        self.runs[idempotency_key] = stored
        return stored

    def portfolio_summary(self, filters):  # noqa: ANN001, ANN201
        return {
            "total_accounts": 2,
            "risk_counts": {"high": 0, "medium": 2, "low": 0},
            "risk_percentages": {"high": 0.0, "medium": 1.0, "low": 0.0},
            "earliest_prediction_timestamp": datetime(2026, 10, 28, tzinfo=UTC),
            "latest_prediction_timestamp": datetime(2026, 10, 28, tzinfo=UTC),
        }

    def fetch_scoring_history(self, filters):  # noqa: ANN001, ANN201
        return [
            {
                "prediction_id": prediction_id,
                "scoring_run_reference": "pipeline-run-21",
                "reference_identifier": reference,
                "loan_reference": None,
                "loan_status": None,
                "prediction_timestamp": datetime(2026, 10, 28, 2, 0, tzinfo=UTC),
                "model_name": "PipelineTest",
                "model_version": "4",
                "model_uri": "models:/PipelineTest@champion",
                "configuration_version": "pipeline-test-config",
                "default_probability": 0.6,
                "risk_band": "medium",
                "warning_reasons": [],
            }
            for prediction_id, reference in ((201, "record-a"), (202, "record-b"))
        ]


def test_small_dataset_prediction_persistence_and_report_are_repeatable(
    tmp_path: Path,
) -> None:
    features = pd.DataFrame(
        [
            {"source_record_id": "record-a", **prototype_features()},
            {"source_record_id": "record-b", **prototype_features()},
        ]
    )
    feature_path = tmp_path / "prototype_features.csv"
    features.to_csv(feature_path, index=False)
    prediction_path = tmp_path / "predictions.json"
    prediction_summary = batch_predict_task(
        {"prototype_features": str(feature_path)},
        output_path=prediction_path,
        bundle=model_bundle(),
    )
    key = pipeline_run_key("2026-10-28T02:00:00+00:00", "a" * 64)
    repository = IdempotentRepository()

    first = persist_predictions_task(
        prediction_summary,
        idempotency_key=key,
        repository=repository,
    )
    second = persist_predictions_task(
        prediction_summary,
        idempotency_key=key,
        repository=repository,
    )
    report_directory = tmp_path / "reports"
    report = refresh_reports_task(
        second,
        output_directory=report_directory,
        repository=repository,
    )

    assert prediction_summary["record_count"] == 2
    assert first == second
    assert repository.insert_calls == 1
    assert report["latest_account_count"] == 2
    assert report["high_risk_account_count"] == 0
    assert Path(report["summary_csv"]).is_file()
    assert Path(report["html_report"]).is_file()


def test_source_validation_is_deterministic_for_small_valid_file(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        feature = prototype_features()
        feature.pop("credit_amount_per_month")
        feature.pop("young_borrower_flag")
        feature.pop("long_duration_flag")
        feature["credit_risk"] = 1 - index
        feature["bad_credit"] = index
        rows.append(feature)
    source = tmp_path / "small.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    first = validate_source_task(source)
    second = validate_source_task(source)

    assert first == second
    assert first["rows_read"] == first["rows_accepted"] == 2
    assert first["rows_rejected"] == 0


@dataclass
class FakeTaskDefinition:
    task_id: str
    upstream_task_ids: set[str]


class FakeTaskArgument:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class FakeDag:
    def __init__(self, configuration: dict[str, Any], tasks: dict[str, Any]) -> None:
        self.dag_id = configuration["dag_id"]
        self.schedule = configuration["schedule"]
        self.catchup = configuration["catchup"]
        self.max_active_runs = configuration["max_active_runs"]
        self.default_args = configuration["default_args"]
        self.task_dict = tasks


def _collect_task_arguments(value: Any) -> set[str]:
    if isinstance(value, FakeTaskArgument):
        return {value.task_id}
    if isinstance(value, (list, tuple)):
        return set().union(*(_collect_task_arguments(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(_collect_task_arguments(item) for item in value.values()))
    return set()


def _fake_airflow_modules() -> tuple[ModuleType, ModuleType]:
    airflow_module = ModuleType("airflow")
    sdk_module = ModuleType("airflow.sdk")
    task_registry: dict[str, FakeTaskDefinition] = {}

    def fake_task(*, task_id: str):  # noqa: ANN202
        def decorate(function):  # noqa: ANN001, ANN202
            def call(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                dependencies = set()
                for value in (*args, *kwargs.values()):
                    dependencies.update(_collect_task_arguments(value))
                task_registry[task_id] = FakeTaskDefinition(task_id, dependencies)
                return FakeTaskArgument(task_id)

            return call

        return decorate

    def fake_dag(**configuration):  # noqa: ANN003, ANN202
        def decorate(function):  # noqa: ANN001, ANN202
            def factory():  # noqa: ANN202
                task_registry.clear()
                function()
                return FakeDag(configuration, dict(task_registry))

            return factory

        return decorate

    sdk_module.task = fake_task
    sdk_module.dag = fake_dag
    sdk_module.get_current_context = lambda: {}
    airflow_module.sdk = sdk_module
    return airflow_module, sdk_module


def test_dag_topology_retries_and_schedule(monkeypatch) -> None:  # noqa: ANN001
    airflow_module, sdk_module = _fake_airflow_modules()
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", sdk_module)
    dag_path = Path(__file__).resolve().parents[1] / "airflow" / "credit_risk_dag.py"

    module = runpy.run_path(str(dag_path))
    dag = module["credit_risk_pipeline_dag"]

    assert dag.dag_id == "credit_risk_daily_pipeline"
    assert dag.schedule == "0 2 * * *"
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.default_args["retries"] == 2
    assert dag.default_args["retry_exponential_backoff"] is True
    assert list(dag.task_dict) == [
        "validate_source",
        "ingest_source",
        "clean_and_validate",
        "generate_features",
        "batch_risk_prediction",
        "persist_predictions",
        "refresh_portfolio_reports",
    ]
    expected_chain = [
        ("validate_source", "ingest_source"),
        ("ingest_source", "clean_and_validate"),
        ("clean_and_validate", "generate_features"),
        ("generate_features", "batch_risk_prediction"),
        ("batch_risk_prediction", "persist_predictions"),
        ("persist_predictions", "refresh_portfolio_reports"),
    ]
    for upstream, downstream in expected_chain:
        assert upstream in dag.task_dict[downstream].upstream_task_ids
