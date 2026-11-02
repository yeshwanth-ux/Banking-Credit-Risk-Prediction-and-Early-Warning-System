"""Verify the containerized API health endpoint and one prediction request."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 10,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def prediction_payload() -> dict[str, Any]:
    return {
        "loan_reference": "DOCKER-SMOKE-001",
        "features": {
            "duration_months": 24,
            "credit_amount_dm": 4000.0,
            "age_years": 35,
            "credit_amount_per_month": 166.6667,
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
        },
        "warning_inputs": {},
    }


def wait_for_healthy_api(base_url: str, attempts: int, delay: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            status, response = request_json(f"{base_url}/health")
            if status == 200 and response.get("status") == "healthy":
                return response
            last_error = RuntimeError(f"API health response was {status}: {response}")
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            last_error = error
        if attempt < attempts - 1:
            time.sleep(delay)
    raise RuntimeError("Containerized API did not become healthy") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health = wait_for_healthy_api(base_url, args.attempts, args.delay)
    status, prediction = request_json(
        f"{base_url}/predict",
        payload=prediction_payload(),
    )
    if status != 200:
        raise RuntimeError(f"Prediction request returned HTTP {status}")
    required = {"predicted_default_probability", "risk_band", "model_version"}
    missing = sorted(required - prediction.keys())
    if missing:
        raise RuntimeError("Prediction response is missing: " + ", ".join(missing))
    if prediction["risk_band"] not in {"low", "medium", "high"}:
        raise RuntimeError("Prediction response contains an invalid risk band")

    print(
        "Docker Compose smoke test passed: "
        f"model={health['registered_model_version']}, "
        f"probability={prediction['predicted_default_probability']}, "
        f"risk_band={prediction['risk_band']}"
    )


if __name__ == "__main__":
    main()
