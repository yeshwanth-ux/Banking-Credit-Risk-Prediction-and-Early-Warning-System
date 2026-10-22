"""Freeze and apply early-warning credit-risk classifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_preparation import (
    DEFAULT_DATA_DIR,
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.ingestion import CATEGORY_RANGES
from src.optimize import training_reference_values
from src.train import TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_MODEL = PROJECT_ROOT / "models" / "optimized_champion.joblib"
DEFAULT_FROZEN_MODEL = PROJECT_ROOT / "models" / "frozen_credit_risk_model.joblib"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "risk_classification.json"
DEFAULT_OPTIMIZATION_RESULTS = PROJECT_ROOT / "reports" / "optimized_model_results.json"
DEFAULT_THRESHOLD_REPORT = PROJECT_ROOT / "reports" / "risk_threshold_analysis.md"
FEATURE_COLUMNS = PROTOTYPE_NUMERIC_FEATURES + PROTOTYPE_CATEGORICAL_FEATURES


class RiskBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RuleInputs:
    missed_payment_count: int = 0
    previous_missed_payment_count: int = 0
    maximum_days_past_due: int = 0
    previous_maximum_days_past_due: int = 0
    debt_to_income_ratio: float | None = None
    previous_debt_to_income_ratio: float | None = None
    previous_default_count: int = 0


@dataclass(frozen=True)
class RuleTrigger:
    code: str
    severity: RiskBand
    reason: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ModelExplanation:
    feature: str
    direction: str
    probability_effect: float
    reason: str


@dataclass(frozen=True)
class WarningResult:
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    risk_probability: float | None
    probability_label: str
    model_band: RiskBand | None
    final_band: RiskBand | None
    model_reasons: tuple[str, ...]
    model_explanations: tuple[ModelExplanation, ...]
    rule_triggers: tuple[RuleTrigger, ...]
    configuration_version: str
    model_version: str
    target_definition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_cost_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
    minimum: float = 0.05,
    maximum: float = 0.95,
) -> dict[str, Any]:
    """Select the lowest-cost validation threshold on a one-point grid."""

    actual = np.asarray(target, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    if len(actual) == 0 or len(actual) != len(scores):
        raise ValueError("Threshold inputs must be non-empty and equal length")
    if false_positive_cost <= 0 or false_negative_cost <= 0:
        raise ValueError("Business costs must be positive")
    if not 0 < minimum < maximum < 1:
        raise ValueError("Threshold search bounds must be inside zero and one")
    candidates = np.round(np.arange(minimum, maximum + 0.001, 0.01), 2)
    evaluated: list[dict[str, Any]] = []
    for threshold in candidates:
        predicted = scores >= threshold
        false_positives = int((predicted & (actual == 0)).sum())
        false_negatives = int((~predicted & (actual == 1)).sum())
        evaluated.append(
            {
                "threshold": float(threshold),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "weighted_cost": float(
                    false_positive_cost * false_positives
                    + false_negative_cost * false_negatives
                ),
            }
        )
    return min(evaluated, key=lambda item: (item["weighted_cost"], item["threshold"]))


def classify_probability(
    probability: float,
    medium_threshold: float,
    high_threshold: float,
) -> RiskBand:
    """Map a probability to an ordered risk band."""

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between zero and one")
    if not 0 < medium_threshold < high_threshold < 1:
        raise ValueError("Risk thresholds must be ordered inside zero and one")
    if probability >= high_threshold:
        return RiskBand.HIGH
    if probability >= medium_threshold:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def evaluate_warning_rules(inputs: RuleInputs, rules: dict[str, Any]) -> tuple[RuleTrigger, ...]:
    """Evaluate configurable point-in-time repayment and debt warning rules."""

    triggers: list[RuleTrigger] = []
    severe_days = int(rules["severe_delinquency_days"])
    delay_days = int(rules["worsening_delay_days"])
    debt_level = float(rules["high_debt_to_income_ratio"])
    debt_change = float(rules["debt_to_income_increase"])

    if inputs.maximum_days_past_due >= severe_days:
        triggers.append(
            RuleTrigger(
                "severe_delinquency",
                RiskBand.HIGH,
                f"Repayment is {inputs.maximum_days_past_due} days past due, meeting the "
                f"{severe_days}-day severe-delinquency rule.",
                {"maximum_days_past_due": inputs.maximum_days_past_due},
            )
        )
    elif inputs.missed_payment_count > 0:
        triggers.append(
            RuleTrigger(
                "missed_payment",
                RiskBand.MEDIUM,
                f"There are {inputs.missed_payment_count} unpaid installments as of the "
                "prediction date.",
                {"missed_payment_count": inputs.missed_payment_count},
            )
        )
    if inputs.missed_payment_count > inputs.previous_missed_payment_count:
        triggers.append(
            RuleTrigger(
                "increasing_missed_payments",
                RiskBand.MEDIUM,
                "Missed payments increased since the prior observation.",
                {
                    "current": inputs.missed_payment_count,
                    "previous": inputs.previous_missed_payment_count,
                },
            )
        )
    delay_increase = (
        inputs.maximum_days_past_due - inputs.previous_maximum_days_past_due
    )
    if (
        30 <= inputs.maximum_days_past_due < severe_days
        and delay_increase >= delay_days
    ):
        triggers.append(
            RuleTrigger(
                "worsening_repayment_delay",
                RiskBand.MEDIUM,
                f"Maximum repayment delay increased by {delay_increase} days.",
                {
                    "current_days": inputs.maximum_days_past_due,
                    "previous_days": inputs.previous_maximum_days_past_due,
                },
            )
        )
    if (
        inputs.debt_to_income_ratio is not None
        and inputs.previous_debt_to_income_ratio is not None
    ):
        increase = inputs.debt_to_income_ratio - inputs.previous_debt_to_income_ratio
        if inputs.debt_to_income_ratio >= debt_level and increase >= debt_change:
            triggers.append(
                RuleTrigger(
                    "increasing_debt_burden",
                    RiskBand.MEDIUM,
                    "Debt-to-income burden is high and increased since the prior observation.",
                    {
                        "current_ratio": inputs.debt_to_income_ratio,
                        "previous_ratio": inputs.previous_debt_to_income_ratio,
                        "increase": increase,
                    },
                )
            )
    if inputs.previous_default_count > 0:
        triggers.append(
            RuleTrigger(
                "previous_default_history",
                RiskBand.HIGH,
                "At least one prior loan met the historical default rule before scoring.",
                {"previous_default_count": inputs.previous_default_count},
            )
        )
    return tuple(triggers)


def _max_band(model_band: RiskBand, triggers: tuple[RuleTrigger, ...]) -> RiskBand:
    order = {RiskBand.LOW: 0, RiskBand.MEDIUM: 1, RiskBand.HIGH: 2}
    bands = [model_band, *(trigger.severity for trigger in triggers)]
    return max(bands, key=order.__getitem__)


def validate_eligibility(record: dict[str, Any]) -> tuple[str, ...]:
    """Return explicit reasons a prototype record cannot be scored."""

    reasons: list[str] = []
    for feature in FEATURE_COLUMNS:
        if feature not in record or record[feature] is None or pd.isna(record[feature]):
            reasons.append(f"missing_required_feature:{feature}")
    numeric_rules = {
        "duration_months": lambda value: float(value) > 0,
        "credit_amount_dm": lambda value: float(value) > 0,
        "age_years": lambda value: 18 <= float(value) <= 120,
        "credit_amount_per_month": lambda value: float(value) > 0,
    }
    for feature, validator in numeric_rules.items():
        if feature not in record or record[feature] is None or pd.isna(record[feature]):
            continue
        try:
            if not validator(record[feature]):
                reasons.append(f"invalid_feature_range:{feature}")
        except (TypeError, ValueError):
            reasons.append(f"invalid_numeric_feature:{feature}")
    for feature in PROTOTYPE_CATEGORICAL_FEATURES:
        if feature not in record or record[feature] is None or pd.isna(record[feature]):
            continue
        if feature in CATEGORY_RANGES:
            try:
                value = int(record[feature])
            except (TypeError, ValueError):
                reasons.append(f"invalid_categorical_feature:{feature}")
                continue
            if value not in CATEGORY_RANGES[feature]:
                reasons.append(f"invalid_category:{feature}")
    for feature in ("young_borrower_flag", "long_duration_flag"):
        if feature not in record or record[feature] is None or pd.isna(record[feature]):
            continue
        if record[feature] not in (0, 1, False, True):
            reasons.append(f"invalid_binary_feature:{feature}")
    return tuple(sorted(set(reasons)))


class RiskClassifier:
    """Apply a frozen model, risk bands, explanations, and warning rules."""

    def __init__(self, model: Any, configuration: dict[str, Any]) -> None:
        self.model = model
        self.configuration = configuration

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_FROZEN_MODEL,
        configuration_path: Path = DEFAULT_CONFIG,
    ) -> RiskClassifier:
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        expected_hash = configuration["model"]["sha256"]
        actual_hash = _sha256(model_path)
        if actual_hash != expected_hash:
            raise ValueError("Frozen model checksum does not match the classification config")
        return cls(joblib.load(model_path), configuration)

    def _model_explanations(self, frame: pd.DataFrame) -> tuple[ModelExplanation, ...]:
        references = self.configuration["explanation_reference_values"]
        original = float(self.model.predict_proba(frame)[:, 1][0])
        effects: list[ModelExplanation] = []
        for feature in FEATURE_COLUMNS:
            perturbed = frame.copy()
            perturbed.loc[:, feature] = references[feature]
            changed = float(self.model.predict_proba(perturbed)[:, 1][0])
            effect = original - changed
            if abs(effect) < 1e-12:
                continue
            direction = "increased" if effect > 0 else "decreased"
            label = feature.replace("_", " ")
            effects.append(
                ModelExplanation(
                    feature=feature,
                    direction=direction,
                    probability_effect=effect,
                    reason=(
                        f"{label.title()} {direction} modeled risk by "
                        f"{abs(effect):.1%} versus its training reference."
                    ),
                )
            )
        effects.sort(key=lambda item: abs(item.probability_effect), reverse=True)
        return tuple(effects[:3])

    def assess(
        self,
        record: dict[str, Any],
        rule_inputs: RuleInputs | None = None,
    ) -> WarningResult:
        eligibility_reasons = validate_eligibility(record)
        metadata = self.configuration
        common = {
            "configuration_version": metadata["configuration_version"],
            "model_version": metadata["model"]["version"],
            "target_definition": metadata["target_definition"],
            "probability_label": metadata["probability_label"],
        }
        if eligibility_reasons:
            return WarningResult(
                eligible=False,
                eligibility_reasons=eligibility_reasons,
                risk_probability=None,
                model_band=None,
                final_band=None,
                model_reasons=(),
                model_explanations=(),
                rule_triggers=(),
                **common,
            )

        frame = pd.DataFrame([{feature: record[feature] for feature in FEATURE_COLUMNS}])
        probability = float(self.model.predict_proba(frame)[:, 1][0])
        thresholds = metadata["risk_bands"]
        model_band = classify_probability(
            probability,
            thresholds["medium_threshold"],
            thresholds["high_threshold"],
        )
        triggers = evaluate_warning_rules(rule_inputs or RuleInputs(), metadata["warning_rules"])
        final_band = _max_band(model_band, triggers)
        medium = thresholds["medium_threshold"]
        high = thresholds["high_threshold"]
        if model_band is RiskBand.LOW:
            band_reason = (
                f"Modeled risk {probability:.1%} is below the {medium:.0%} "
                "review threshold."
            )
        elif model_band is RiskBand.MEDIUM:
            band_reason = (
                f"Modeled risk {probability:.1%} is between the {medium:.0%} review "
                f"threshold and {high:.0%} high-risk threshold."
            )
        else:
            band_reason = (
                f"Modeled risk {probability:.1%} meets the {high:.0%} "
                "high-risk threshold."
            )
        return WarningResult(
            eligible=True,
            eligibility_reasons=(),
            risk_probability=probability,
            model_band=model_band,
            final_band=final_band,
            model_reasons=(band_reason,),
            model_explanations=self._model_explanations(frame),
            rule_triggers=triggers,
            **common,
        )


def _threshold_diagnostics(
    target: np.ndarray,
    probabilities: np.ndarray,
    selected: dict[str, Any],
) -> dict[str, Any]:
    threshold = selected["threshold"]
    predicted = probabilities >= threshold
    true_positives = int((predicted & (target == 1)).sum())
    true_negatives = int((~predicted & (target == 0)).sum())
    return {
        **selected,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "precision": true_positives / max(1, int(predicted.sum())),
        "recall": true_positives / max(1, int((target == 1).sum())),
    }


def freeze_risk_configuration(
    selected_model_path: Path = DEFAULT_SELECTED_MODEL,
    frozen_model_path: Path = DEFAULT_FROZEN_MODEL,
    config_path: Path = DEFAULT_CONFIG,
    optimization_results_path: Path = DEFAULT_OPTIMIZATION_RESULTS,
    training_path: Path = DEFAULT_DATA_DIR / "train.csv",
    validation_path: Path = DEFAULT_DATA_DIR / "validation.csv",
    threshold_report_path: Path = DEFAULT_THRESHOLD_REPORT,
) -> dict[str, Any]:
    """Freeze the selected model, thresholds, rule settings, and references."""

    model = joblib.load(selected_model_path)
    optimization = json.loads(optimization_results_path.read_text(encoding="utf-8"))
    training = pd.read_csv(training_path)
    validation = pd.read_csv(validation_path)
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    target = validation[TARGET_COLUMN].to_numpy(dtype="int64")

    medium = select_cost_threshold(
        target,
        probabilities,
        false_positive_cost=1,
        false_negative_cost=5,
        minimum=0.10,
        maximum=0.70,
    )
    high = select_cost_threshold(
        target,
        probabilities,
        false_positive_cost=4,
        false_negative_cost=1,
        minimum=max(0.50, medium["threshold"] + 0.10),
        maximum=0.95,
    )
    medium_diagnostics = _threshold_diagnostics(target, probabilities, medium)
    high_diagnostics = _threshold_diagnostics(target, probabilities, high)

    frozen_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_model_path, frozen_model_path)
    references = training_reference_values(training[FEATURE_COLUMNS])
    generated_at = datetime.now(tz=UTC).isoformat()
    configuration = {
        "configuration_version": "risk-classification-1.0.0",
        "frozen_at_utc": generated_at,
        "model": {
            "version": "optimized-logistic-regression-1.0.0",
            "family": optimization["champion"],
            "artifact": "models/frozen_credit_risk_model.joblib",
            "sha256": _sha256(frozen_model_path),
        },
        "target_definition": (
            "South German Credit bad_credit contract-compliance outcome; prototype "
            "proxy only, not default within 90 days"
        ),
        "probability_label": "prototype bad-credit risk probability",
        "eligibility": {
            "required_features": FEATURE_COLUMNS,
            "requires_valid_documented_ranges": True,
        },
        "risk_bands": {
            "low": f"probability < {medium['threshold']:.2f}",
            "medium": (
                f"{medium['threshold']:.2f} <= probability < {high['threshold']:.2f}"
            ),
            "high": f"probability >= {high['threshold']:.2f}",
            "medium_threshold": medium["threshold"],
            "high_threshold": high["threshold"],
        },
        "threshold_selection": {
            "dataset": "validation",
            "test_split_used": False,
            "medium_review": {
                "false_positive_cost": 1,
                "false_negative_cost": 5,
                **medium_diagnostics,
            },
            "high_intervention": {
                "false_positive_cost": 4,
                "false_negative_cost": 1,
                **high_diagnostics,
            },
        },
        "warning_rules": {
            "severe_delinquency_days": 90,
            "worsening_delay_days": 15,
            "high_debt_to_income_ratio": 0.40,
            "debt_to_income_increase": 0.10,
        },
        "explanation_reference_values": references,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    _write_threshold_report(threshold_report_path, configuration)
    return configuration


def _write_threshold_report(path: Path, configuration: dict[str, Any]) -> None:
    selection = configuration["threshold_selection"]
    medium = selection["medium_review"]
    high = selection["high_intervention"]
    bands = configuration["risk_bands"]
    lines = [
        "# Early-warning threshold analysis",
        "",
        f"**Configuration:** `{configuration['configuration_version']}`",
        f"**Model:** `{configuration['model']['version']}`",
        "**Selection data:** validation only",
        "**Held-out test used:** no",
        "",
        "## Frozen risk bands",
        "",
        f"- Low: `{bands['low']}`",
        f"- Medium: `{bands['medium']}`",
        f"- High: `{bands['high']}`",
        "",
        "## Business cost assumptions",
        "",
        "The medium threshold represents a manual-review decision. A missed bad-credit "
        "case is assigned five cost units and an unnecessary review one cost unit.",
        "",
        "The high threshold represents a more intensive intervention. An unnecessary "
        "intervention is assigned four cost units and a missed high-band case one cost "
        "unit. This produces a narrow high band; severe rule triggers can independently "
        "escalate the final warning to High.",
        "",
        "| Decision | Threshold | FP | FN | Precision | Recall | Weighted cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Medium review | {medium['threshold']:.2f} | {medium['false_positives']} | "
        f"{medium['false_negatives']} | {medium['precision']:.4f} | "
        f"{medium['recall']:.4f} | {medium['weighted_cost']:.1f} |",
        f"| High intervention | {high['threshold']:.2f} | {high['false_positives']} | "
        f"{high['false_negatives']} | {high['precision']:.4f} | "
        f"{high['recall']:.4f} | {high['weighted_cost']:.1f} |",
        "",
        "## Limitations",
        "",
        "The frozen model predicts the prototype `bad_credit` outcome, not a dated "
        "90-day probability of default. Costs are initial assumptions for portfolio "
        "demonstration and require business, risk, fairness, and compliance review. "
        "Validation has 200 records, so thresholds require monitoring and later-data "
        "confirmation before operational use.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze", help="Freeze model and warning config")
    freeze_parser.add_argument("--selected-model", type=Path, default=DEFAULT_SELECTED_MODEL)
    freeze_parser.add_argument("--frozen-model", type=Path, default=DEFAULT_FROZEN_MODEL)
    freeze_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    assess_parser = subparsers.add_parser("assess", help="Assess one JSON record")
    assess_parser.add_argument("record", type=Path)
    assess_parser.add_argument("--rules", type=Path)
    assess_parser.add_argument("--model", type=Path, default=DEFAULT_FROZEN_MODEL)
    assess_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    if args.command == "freeze":
        configuration = freeze_risk_configuration(
            selected_model_path=args.selected_model,
            frozen_model_path=args.frozen_model,
            config_path=args.config,
        )
        print(f"Frozen configuration: {configuration['configuration_version']}")
        print(
            "Risk thresholds: "
            f"medium={configuration['risk_bands']['medium_threshold']:.2f}, "
            f"high={configuration['risk_bands']['high_threshold']:.2f}"
        )
        return

    record = json.loads(args.record.read_text(encoding="utf-8"))
    rule_inputs = RuleInputs()
    if args.rules:
        rule_inputs = RuleInputs(**json.loads(args.rules.read_text(encoding="utf-8")))
    classifier = RiskClassifier.load(args.model, args.config)
    print(json.dumps(classifier.assess(record, rule_inputs).to_dict(), indent=2))


if __name__ == "__main__":
    main()
