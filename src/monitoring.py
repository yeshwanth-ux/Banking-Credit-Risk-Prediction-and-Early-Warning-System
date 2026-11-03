"""Privacy-safe operational and model monitoring for the prediction API."""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_preparation import (
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_PROFILE = PROJECT_ROOT / "config" / "training_distributions.json"
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "data" / "processed" / "train.csv"
MONITORED_FEATURES = PROTOTYPE_NUMERIC_FEATURES + PROTOTYPE_CATEGORICAL_FEATURES


def build_training_profile(training: pd.DataFrame) -> dict[str, Any]:
    """Build non-identifying reference distributions from the training split."""

    missing = sorted(set(MONITORED_FEATURES) - set(training.columns))
    if missing:
        raise ValueError("Training profile is missing features: " + ", ".join(missing))
    features: dict[str, Any] = {}
    for name in PROTOTYPE_NUMERIC_FEATURES:
        values = pd.to_numeric(training[name], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Training feature has no numeric values: {name}")
        quantiles = np.quantile(values, np.linspace(0, 1, 11))
        edges = np.unique(quantiles).astype(float)
        if len(edges) < 2:
            value = float(edges[0])
            edges = np.array([value - 1.0, value + 1.0])
        edges[0], edges[-1] = -math.inf, math.inf
        counts, _ = np.histogram(values, bins=edges)
        features[name] = {
            "type": "numeric",
            "bin_edges": [
                "-inf" if math.isinf(edge) and edge < 0 else
                "inf" if math.isinf(edge) else float(edge)
                for edge in edges
            ],
            "proportions": (counts / counts.sum()).round(8).tolist(),
        }
    for name in PROTOTYPE_CATEGORICAL_FEATURES:
        proportions = training[name].astype(str).value_counts(normalize=True, dropna=False)
        features[name] = {
            "type": "categorical",
            "proportions": {
                str(category): round(float(proportion), 8)
                for category, proportion in sorted(proportions.items())
            },
        }
    return {
        "profile_version": "1.0.0",
        "source": "training split only",
        "row_count": len(training),
        "features": features,
    }


def save_training_profile(training_path: Path, output_path: Path) -> dict[str, Any]:
    profile = build_training_profile(pd.read_csv(training_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def load_training_profile(path: Path = DEFAULT_TRAINING_PROFILE) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training distribution profile does not exist: {path}")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("source") != "training split only" or not profile.get("features"):
        raise ValueError("Training distribution profile is invalid")
    return profile


def _numeric_edges(values: list[float | str]) -> np.ndarray:
    return np.array(
        [
            -math.inf if value == "-inf" else math.inf if value == "inf" else value
            for value in values
        ],
        dtype="float64",
    )


def population_stability_index(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    expected_safe = np.clip(expected.astype(float), epsilon, None)
    observed_safe = np.clip(observed.astype(float), epsilon, None)
    expected_safe /= expected_safe.sum()
    observed_safe /= observed_safe.sum()
    return float(((observed_safe - expected_safe) * np.log(observed_safe / expected_safe)).sum())


def feature_drift(
    observations: pd.DataFrame,
    profile: dict[str, Any],
    *,
    minimum_samples: int = 30,
    alert_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    """Compare a scoring window with saved training distributions."""

    results: list[dict[str, Any]] = []
    for name, reference in profile["features"].items():
        values = observations[name].dropna() if name in observations else pd.Series(dtype=float)
        if len(values) < minimum_samples:
            results.append(
                {
                    "feature": name,
                    "method": "psi" if reference["type"] == "numeric" else "tvd",
                    "score": None,
                    "sample_count": len(values),
                    "status": "insufficient_data",
                }
            )
            continue
        if reference["type"] == "numeric":
            edges = _numeric_edges(reference["bin_edges"])
            numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
            counts, _ = np.histogram(numeric, bins=edges)
            observed = counts / max(counts.sum(), 1)
            score = population_stability_index(
                np.asarray(reference["proportions"]),
                observed,
            )
            method = "psi"
        else:
            expected_map = reference["proportions"]
            observed_map = values.astype(str).value_counts(normalize=True).to_dict()
            categories = set(expected_map) | set(observed_map)
            score = 0.5 * sum(
                abs(float(observed_map.get(item, 0)) - float(expected_map.get(item, 0)))
                for item in categories
            )
            method = "tvd"
        results.append(
            {
                "feature": name,
                "method": method,
                "score": round(float(score), 6),
                "sample_count": len(values),
                "status": "alert" if score >= alert_threshold else "stable",
            }
        )
    return results


class ApplicationMonitor:
    """Maintain bounded operational and model-monitoring aggregates."""

    def __init__(
        self,
        training_profile: dict[str, Any],
        *,
        window_size: int = 1000,
        minimum_drift_samples: int = 30,
    ) -> None:
        self.training_profile = training_profile
        self.minimum_drift_samples = minimum_drift_samples
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._latency_ms_total = 0.0
        self._requests_by_path: Counter[str] = Counter()
        self._risk_counts: Counter[str] = Counter()
        self._observations: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._last_scoring_timestamp: datetime | None = None

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._latency_ms_total += latency_ms
            self._requests_by_path[path] += 1
            if status_code >= 400:
                self._error_count += 1

    def record_prediction(
        self,
        features: dict[str, Any],
        risk_band: str,
        scoring_timestamp: datetime,
    ) -> None:
        safe_features = {name: features.get(name) for name in MONITORED_FEATURES}
        with self._lock:
            self._observations.append(safe_features)
            self._risk_counts[risk_band] += 1
            self._last_scoring_timestamp = scoring_timestamp

    def summary(self, model_version: str, now: datetime | None = None) -> dict[str, Any]:
        current_time = now or datetime.now(tz=UTC)
        with self._lock:
            request_count = self._request_count
            error_count = self._error_count
            latency_total = self._latency_ms_total
            requests_by_path = dict(self._requests_by_path)
            risk_counts = {
                band: self._risk_counts.get(band, 0)
                for band in ("high", "medium", "low")
            }
            observations = list(self._observations)
            last_scoring = self._last_scoring_timestamp
        prediction_count = sum(risk_counts.values())
        drift = feature_drift(
            pd.DataFrame(observations),
            self.training_profile,
            minimum_samples=self.minimum_drift_samples,
        )
        return {
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": error_count / request_count if request_count else 0.0,
            "average_latency_ms": latency_total / request_count if request_count else 0.0,
            "requests_by_path": requests_by_path,
            "prediction_count": prediction_count,
            "risk_counts": risk_counts,
            "risk_proportions": {
                band: count / prediction_count if prediction_count else 0.0
                for band, count in risk_counts.items()
            },
            "model_version": model_version,
            "last_scoring_timestamp": last_scoring,
            "last_scoring_age_seconds": (
                max(0.0, (current_time - last_scoring).total_seconds())
                if last_scoring is not None
                else None
            ),
            "drift_window_size": len(observations),
            "drift_features": drift,
            "drift_alert_count": sum(item["status"] == "alert" for item in drift),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRAINING_PROFILE)
    args = parser.parse_args()
    profile = save_training_profile(args.training, args.output)
    print(
        f"Saved training profile for {profile['row_count']} rows and "
        f"{len(profile['features'])} features to {args.output}"
    )


if __name__ == "__main__":
    main()
