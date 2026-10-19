"""Evaluation metrics and reporting helpers for binary credit-risk models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryModelMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    threshold: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    false_positive_rate: float
    false_negative_rate: float
    predicted_positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_binary_predictions(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.50,
) -> BinaryModelMetrics:
    """Calculate discrimination, classification, and error-trade-off metrics."""

    actual = np.asarray(target, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    if actual.ndim != 1 or scores.ndim != 1 or len(actual) != len(scores):
        raise ValueError("Target and probabilities must be equal-length one-dimensional arrays")
    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if set(np.unique(actual)) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    predicted = (scores >= threshold).astype("int64")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()
    negative_count = true_negative + false_positive
    positive_count = true_positive + false_negative
    return BinaryModelMetrics(
        roc_auc=float(roc_auc_score(actual, scores)),
        pr_auc=float(average_precision_score(actual, scores)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1_score=float(f1_score(actual, predicted, zero_division=0)),
        threshold=threshold,
        true_negatives=int(true_negative),
        false_positives=int(false_positive),
        false_negatives=int(false_negative),
        true_positives=int(true_positive),
        false_positive_rate=float(false_positive / negative_count),
        false_negative_rate=float(false_negative / positive_count),
        predicted_positive_rate=float(predicted.mean()),
    )
