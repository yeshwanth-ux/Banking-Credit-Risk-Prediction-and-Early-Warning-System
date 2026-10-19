from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import evaluate_binary_predictions


def test_binary_metrics_include_confusion_and_tradeoff_rates() -> None:
    metrics = evaluate_binary_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.4, 0.9]),
    )

    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.pr_auc == pytest.approx(5 / 6)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1_score == pytest.approx(0.5)
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.false_positive_rate == pytest.approx(0.5)
    assert metrics.false_negative_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("target", "scores", "message"),
    [
        (np.array([]), np.array([]), "empty"),
        (np.array([0, 0]), np.array([0.1, 0.2]), "both binary classes"),
        (np.array([0, 1]), np.array([0.1, 1.2]), "between 0 and 1"),
    ],
)
def test_binary_metrics_reject_invalid_inputs(
    target: np.ndarray, scores: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_binary_predictions(target, scores)


def test_binary_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="Threshold"):
        evaluate_binary_predictions(np.array([0, 1]), np.array([0.1, 0.9]), threshold=1)
