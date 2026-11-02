"""Static contracts for the Docker Compose deployment."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def compose_config() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_api_image_runs_as_non_root_and_exposes_service() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "USER creditrisk" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'CMD ["uvicorn", "api.main:app"' in dockerfile


def test_compose_defines_services_health_checks_and_persistent_volumes() -> None:
    configuration = compose_config()
    services = configuration["services"]

    assert {"postgres", "mlflow", "model-init", "api", "airflow"} == set(services)
    for service_name in ("postgres", "mlflow", "api", "airflow"):
        assert "healthcheck" in services[service_name]
    assert "mlflow,localhost,127.0.0.1" in services["mlflow"]["command"]
    assert set(configuration["volumes"]) == {
        "postgres-data",
        "mlflow-state",
        "mlflow-artifacts",
        "airflow-state",
        "airflow-logs",
        "report-artifacts",
    }


def test_api_and_airflow_wait_for_healthy_dependencies_and_model_registration() -> None:
    services = compose_config()["services"]

    for service_name in ("api", "airflow"):
        dependencies = services[service_name]["depends_on"]
        assert dependencies["postgres"]["condition"] == "service_healthy"
        assert dependencies["mlflow"]["condition"] == "service_healthy"
        assert dependencies["model-init"]["condition"] == (
            "service_completed_successfully"
        )


def test_compose_requires_secrets_instead_of_embedding_them() -> None:
    source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "change-me" not in source
    assert "POSTGRES_PASSWORD:?" in source
    assert "RISK_REPORT_API_KEY:?" in source
    assert "MLFLOW_TRACKING_URI: http://mlflow:5000" in source


def test_dockerignore_excludes_local_secrets_and_runtime_state() -> None:
    entries = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".env.*", "mlartifacts", "mlflow.db", "airflow/logs"} <= entries
    assert "!.env.example" in entries
