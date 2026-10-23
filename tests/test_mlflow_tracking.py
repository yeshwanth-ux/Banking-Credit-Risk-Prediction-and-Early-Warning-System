"""Tests for MLflow tracking metadata and prediction-service loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import model_registry
from src.mlflow_tracking import feature_definitions, sqlite_tracking_uri


class ProbabilityPipeline:
    def predict_proba(self, rows):  # noqa: ANN001, ANN201
        return [[0.7, 0.3] for _ in range(len(rows))]


class RegistryClient:
    def __init__(self, version):  # noqa: ANN001
        self.version = version

    def get_model_version_by_alias(self, name: str, alias: str):  # noqa: ANN201
        assert name == "TestCreditRisk"
        assert alias == "champion"
        return self.version


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "configuration_version": "risk-config-test",
                "model": {"sha256": "abc123"},
            }
        ),
        encoding="utf-8",
    )


def test_sqlite_tracking_uri_uses_absolute_forward_slash_path(tmp_path: Path) -> None:
    uri = sqlite_tracking_uri(tmp_path / "tracking.db")

    assert uri.startswith("sqlite:///")
    assert uri.endswith("tracking.db")
    assert "\\" not in uri


def test_feature_definitions_cover_every_raw_feature() -> None:
    definitions = feature_definitions()
    names = {
        item["name"]
        for group in ("numerical_features", "categorical_features")
        for item in definitions[group]
    }

    assert definitions["feature_count"] == 23
    assert len(names) == definitions["feature_count"]
    assert definitions["target"]["name"] == "bad_credit"


def test_load_registered_model_verifies_and_returns_aliased_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "risk.json"
    _write_config(config_path)
    version = SimpleNamespace(
        version="7",
        run_id="run-123",
        tags={
            "configuration_version": "risk-config-test",
            "frozen_model_sha256": "abc123",
            "test_split_used": "false",
        },
    )
    pipeline = ProbabilityPipeline()
    monkeypatch.setattr(
        model_registry,
        "MlflowClient",
        lambda tracking_uri: RegistryClient(version),
    )
    monkeypatch.setattr(model_registry.mlflow.sklearn, "load_model", lambda uri: pipeline)

    bundle = model_registry.load_registered_model(
        tracking_uri="sqlite:///test.db",
        registered_model_name="TestCreditRisk",
        config_path=config_path,
    )

    assert bundle.pipeline is pipeline
    assert bundle.model_uri == "models:/TestCreditRisk@champion"
    assert bundle.registered_model_version == "7"
    assert bundle.run_id == "run-123"


def test_load_registered_model_rejects_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "risk.json"
    _write_config(config_path)
    version = SimpleNamespace(
        version="7",
        run_id="run-123",
        tags={
            "configuration_version": "wrong",
            "frozen_model_sha256": "abc123",
            "test_split_used": "false",
        },
    )
    monkeypatch.setattr(
        model_registry,
        "MlflowClient",
        lambda tracking_uri: RegistryClient(version),
    )

    with pytest.raises(ValueError, match="does not match frozen config"):
        model_registry.load_registered_model(
            tracking_uri="sqlite:///test.db",
            registered_model_name="TestCreditRisk",
            config_path=config_path,
        )
