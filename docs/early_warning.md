# Early-warning risk classification

Day 10 freezes the selected Day 9 model with a versioned classification
configuration. The scorer returns model risk, rule triggers, a final escalated
risk band, eligibility findings, and separate human-readable explanations.

## Risk bands and threshold selection

Thresholds are selected from validation predictions only. The initial medium
threshold minimizes a manual-review cost where a missed bad-credit case costs
five units and an unnecessary review costs one. The high threshold models a more
intensive intervention where a false intervention costs four units and a missed
high-band case costs one. Exact frozen thresholds and validation diagnostics are
stored in `config/risk_classification.json` and
`reports/risk_threshold_analysis.md`.

These costs are initial portfolio assumptions. They require approval by business,
risk, compliance, and fairness stakeholders before operational use.

## Warning rules

Point-in-time rule inputs can trigger warnings for:

- one or more missed installments;
- an increase in missed installments;
- repayment delay worsening by the configured number of days;
- high and increasing debt-to-income burden;
- severe delinquency; and
- prior default history known before scoring.

Medium rule triggers raise a Low model result to at least Medium. Severe
delinquency and prior-default triggers raise the final result to High. The
original model band remains in the result so consumers can distinguish score
classification from rule escalation.

## Explanations

`model_reasons` explain the probability threshold classification.
`model_explanations` show the three largest one-feature perturbation effects
against training-set medians or modes. `rule_triggers` separately state observed
historical conditions and evidence. Perturbation effects describe model
sensitivity, not causality or adverse-action reasons.

## Eligibility and target limitation

The frozen prototype model scores only records containing its documented feature
set and valid source ranges. Ineligible records receive reasons and no score.

The returned probability is explicitly labeled a prototype bad-credit risk
probability. The South German Credit outcome is contract compliance and does not
represent default within 90 days. A future operational model must be trained on
the dated target before this interface can produce a true probability of default.

## Freeze and assess

```powershell
.\.venv\Scripts\python.exe -m src.predict freeze
.\.venv\Scripts\python.exe -m src.predict assess input.json --rules rule-inputs.json
```

The frozen model is an ignored local artifact. Its SHA-256 checksum is stored in
the tracked configuration and verified whenever the classifier loads it.
