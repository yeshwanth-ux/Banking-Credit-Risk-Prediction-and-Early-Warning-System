"""Load the approved MLflow model bundle for prediction-service integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.mlflow_tracking import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_REGISTERED_MODEL,
    local_tracking_uri,
)
from src.predict import DEFAULT_CONFIG, RiskClassifier


@dataclass(frozen=True)
class RegisteredModelBundle:
    """Loaded model plus the immutable classification and registry metadata."""

    pipeline: Any
    classification_config: dict[str, Any]
    model_uri: str
    registered_model_name: str
    registered_model_version: str
    alias: str
    run_id: str

    def build_classifier(self) -> RiskClassifier:
        """Construct the Day 10 warning classifier with the registered pipeline."""

        return RiskClassifier(self.pipeline, self.classification_config)


def load_registered_model(
    *,
    tracking_uri: str | None = None,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL,
    alias: str = DEFAULT_MODEL_ALIAS,
    config_path: Path = DEFAULT_CONFIG,
) -> RegisteredModelBundle:
    """Load an aliased model version and verify its frozen configuration metadata."""

    uri = tracking_uri or local_tracking_uri()
    mlflow.set_tracking_uri(uri)
    if not config_path.is_file():
        raise FileNotFoundError(f"Risk-classification config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version_by_alias(registered_model_name, alias)
    expected_tags = {
        "configuration_version": config["configuration_version"],
        "frozen_model_sha256": config["model"]["sha256"],
        "test_split_used": "false",
    }
    mismatches = {
        key: {"expected": expected, "actual": version.tags.get(key)}
        for key, expected in expected_tags.items()
        if version.tags.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Registered model metadata does not match frozen config: {mismatches}")
    model_uri = f"models:/{registered_model_name}@{alias}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Registered model does not expose predict_proba")
    return RegisteredModelBundle(
        pipeline=pipeline,
        classification_config=config,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_model_version=str(version.version),
        alias=alias,
        run_id=version.run_id,
    )
