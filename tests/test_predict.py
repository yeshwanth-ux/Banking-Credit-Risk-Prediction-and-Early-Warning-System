from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.predict import (
    FEATURE_COLUMNS,
    RiskBand,
    RiskClassifier,
    RuleInputs,
    classify_probability,
    evaluate_warning_rules,
    select_cost_threshold,
    validate_eligibility,
)


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.probability, self.probability], (len(features), 1))


def eligible_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "duration_months": 24,
        "credit_amount_dm": 4000.0,
        "age_years": 35,
        "credit_amount_per_month": 4000 / 24,
    }
    category_values = {
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
    record.update(category_values)
    return record


def configuration() -> dict[str, Any]:
    references = eligible_record()
    return {
        "configuration_version": "test-1",
        "model": {"version": "test-model"},
        "target_definition": "test prototype target",
        "probability_label": "test risk probability",
        "risk_bands": {"medium_threshold": 0.40, "high_threshold": 0.80},
        "warning_rules": {
            "severe_delinquency_days": 90,
            "worsening_delay_days": 15,
            "high_debt_to_income_ratio": 0.40,
            "debt_to_income_increase": 0.10,
        },
        "explanation_reference_values": references,
    }


def test_probability_band_boundaries() -> None:
    assert classify_probability(0.39, 0.40, 0.80) is RiskBand.LOW
    assert classify_probability(0.40, 0.40, 0.80) is RiskBand.MEDIUM
    assert classify_probability(0.79, 0.40, 0.80) is RiskBand.MEDIUM
    assert classify_probability(0.80, 0.40, 0.80) is RiskBand.HIGH


def test_cost_threshold_prefers_lower_weighted_error() -> None:
    selected = select_cost_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.10, 0.40, 0.45, 0.90]),
        false_positive_cost=1,
        false_negative_cost=5,
        minimum=0.20,
        maximum=0.80,
    )

    assert selected["threshold"] == pytest.approx(0.41)
    assert selected["false_positives"] == 0
    assert selected["false_negatives"] == 0


def test_warning_rules_detect_missed_payments_debt_and_worsening_delay() -> None:
    triggers = evaluate_warning_rules(
        RuleInputs(
            missed_payment_count=2,
            previous_missed_payment_count=1,
            maximum_days_past_due=45,
            previous_maximum_days_past_due=20,
            debt_to_income_ratio=0.55,
            previous_debt_to_income_ratio=0.40,
        ),
        configuration()["warning_rules"],
    )

    assert {trigger.code for trigger in triggers} == {
        "missed_payment",
        "increasing_missed_payments",
        "worsening_repayment_delay",
        "increasing_debt_burden",
    }
    assert all(trigger.severity is RiskBand.MEDIUM for trigger in triggers)


def test_severe_delinquency_and_previous_default_are_high_severity() -> None:
    triggers = evaluate_warning_rules(
        RuleInputs(maximum_days_past_due=95, previous_default_count=1),
        configuration()["warning_rules"],
    )

    assert {trigger.code for trigger in triggers} == {
        "severe_delinquency",
        "previous_default_history",
    }
    assert all(trigger.severity is RiskBand.HIGH for trigger in triggers)


def test_rule_trigger_escalates_model_band_and_reasons_remain_separate() -> None:
    classifier = RiskClassifier(FixedProbabilityModel(0.60), configuration())

    result = classifier.assess(
        eligible_record(),
        RuleInputs(maximum_days_past_due=95),
    )

    assert result.eligible is True
    assert result.model_band is RiskBand.MEDIUM
    assert result.final_band is RiskBand.HIGH
    assert "60.0%" in result.model_reasons[0]
    assert [trigger.code for trigger in result.rule_triggers] == ["severe_delinquency"]
    assert result.model_explanations == ()


def test_missing_or_invalid_features_make_record_ineligible() -> None:
    record = eligible_record()
    record.pop(PROTOTYPE_NUMERIC_FEATURES[0])
    record[PROTOTYPE_CATEGORICAL_FEATURES[0]] = 99

    reasons = validate_eligibility(record)
    result = RiskClassifier(FixedProbabilityModel(0.9), configuration()).assess(record)

    assert f"missing_required_feature:{PROTOTYPE_NUMERIC_FEATURES[0]}" in reasons
    assert f"invalid_category:{PROTOTYPE_CATEGORICAL_FEATURES[0]}" in reasons
    assert result.eligible is False
    assert result.risk_probability is None
    assert result.final_band is None


def test_eligible_record_contains_every_frozen_feature() -> None:
    assert set(eligible_record()) == set(FEATURE_COLUMNS)
    assert validate_eligibility(eligible_record()) == ()
