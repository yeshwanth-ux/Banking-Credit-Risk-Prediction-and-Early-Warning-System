"""Prepare leakage-safe training, validation, and test datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.database import DatabaseSettings
from src.feature_engineering import (
    engineer_prototype_features,
    load_prototype_feature_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
DEFAULT_SPLIT_SUMMARY = PROJECT_ROOT / "reports" / "dataset_split_summary.json"

PROTOTYPE_NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount_dm",
    "age_years",
    "credit_amount_per_month",
]
PROTOTYPE_CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "young_borrower_flag",
    "long_duration_flag",
]


@dataclass(frozen=True)
class SplitDefinition:
    method: str
    random_state: int | None
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    date_column: str | None
    train_end: str | None
    validation_end: str | None
    counts: dict[str, int]
    target_rates: dict[str, float]


def create_split_assignments(
    data: pd.DataFrame,
    *,
    target_column: str,
    id_column: str,
    date_column: str | None = None,
    random_state: int = 42,
) -> tuple[pd.Series, SplitDefinition]:
    """Assign 60/20/20 splits, preferring chronology when dates are reliable."""

    required = {target_column, id_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing split columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Cannot split an empty dataset")
    if data[id_column].isna().any() or data[id_column].duplicated().any():
        raise ValueError("Split identifiers must be populated and unique")
    if data[target_column].isna().any():
        raise ValueError("Split target must be populated")

    assignments = pd.Series(index=data.index, dtype="string", name="split")
    train_end = validation_end = None
    use_time = date_column is not None and date_column in data.columns
    if use_time:
        dates = pd.to_datetime(data[date_column], errors="coerce")
        use_time = dates.notna().all() and dates.nunique() >= 3

    if use_time:
        ordered = data.assign(_split_date=dates).sort_values(
            ["_split_date", id_column], kind="stable"
        )
        train_cut = max(1, int(len(ordered) * 0.60))
        validation_cut = max(train_cut + 1, int(len(ordered) * 0.80))
        if validation_cut >= len(ordered):
            raise ValueError("At least five rows are required for chronological splits")
        assignments.loc[ordered.index[:train_cut]] = "train"
        assignments.loc[ordered.index[train_cut:validation_cut]] = "validation"
        assignments.loc[ordered.index[validation_cut:]] = "test"
        train_end = dates.loc[ordered.index[train_cut - 1]].date().isoformat()
        validation_end = dates.loc[ordered.index[validation_cut - 1]].date().isoformat()
        method = "time_based"
        saved_random_state: int | None = None
    else:
        indices = data.index.to_numpy()
        target = data[target_column]
        train_indices, remaining_indices = train_test_split(
            indices,
            test_size=0.40,
            random_state=random_state,
            stratify=target,
        )
        remaining_target = target.loc[remaining_indices]
        validation_indices, test_indices = train_test_split(
            remaining_indices,
            test_size=0.50,
            random_state=random_state,
            stratify=remaining_target,
        )
        assignments.loc[train_indices] = "train"
        assignments.loc[validation_indices] = "validation"
        assignments.loc[test_indices] = "test"
        method = "stratified_random"
        saved_random_state = random_state

    counts = assignments.value_counts().reindex(["train", "validation", "test"]).to_dict()
    rates = {
        split: float(data.loc[assignments.eq(split), target_column].mean())
        for split in ("train", "validation", "test")
    }
    definition = SplitDefinition(
        method=method,
        random_state=saved_random_state,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        date_column=date_column if use_time else None,
        train_end=train_end,
        validation_end=validation_end,
        counts={key: int(value) for key, value in counts.items()},
        target_rates=rates,
    )
    return assignments, definition


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Create imputation, scaling, and one-hot encoding transformations."""

    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numerical, numerical_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_prototype_datasets(
    output_dir: Path = DEFAULT_DATA_DIR,
    preprocessor_path: Path = DEFAULT_MODEL_PATH,
    split_summary_path: Path = DEFAULT_SPLIT_SUMMARY,
    settings: DatabaseSettings | None = None,
) -> tuple[dict[str, pd.DataFrame], ColumnTransformer, SplitDefinition]:
    """Split prototype records and fit preprocessing using training rows only."""

    data = engineer_prototype_features(load_prototype_feature_source(settings))
    assignments, definition = create_split_assignments(
        data,
        target_column="bad_credit",
        id_column="source_record_id",
        random_state=42,
    )
    feature_columns = PROTOTYPE_NUMERIC_FEATURES + PROTOTYPE_CATEGORICAL_FEATURES
    preprocessor = build_preprocessor(
        PROTOTYPE_NUMERIC_FEATURES,
        PROTOTYPE_CATEGORICAL_FEATURES,
    )
    training_features = data.loc[assignments.eq("train"), feature_columns]
    preprocessor.fit(training_features)

    datasets: dict[str, pd.DataFrame] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        split_data = data.loc[assignments.eq(split)].copy()
        split_data.insert(1, "split", split)
        split_data.to_csv(output_dir / f"{split}.csv", index=False)
        datasets[split] = split_data

    assignment_frame = data[["source_record_id", "bad_credit"]].copy()
    assignment_frame.insert(1, "split", assignments)
    assignment_frame.to_csv(output_dir / "split_assignments.csv", index=False)

    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    summary: dict[str, Any] = asdict(definition)
    summary.update(
        {
            "dataset": "South German Credit research prototype",
            "target": "bad_credit",
            "id_column": "source_record_id",
            "numerical_features": PROTOTYPE_NUMERIC_FEATURES,
            "categorical_features": PROTOTYPE_CATEGORICAL_FEATURES,
            "preprocessor_fit_split": "train",
            "row_level_assignments": "data/processed/split_assignments.csv",
        }
    )
    split_summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return datasets, preprocessor, definition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY)
    args = parser.parse_args()
    datasets, preprocessor, definition = prepare_prototype_datasets(
        args.output_dir,
        args.preprocessor,
        args.split_summary,
    )
    print(f"Split method: {definition.method}")
    print(f"Rows: {definition.counts}")
    print(f"Preprocessed training columns: {len(preprocessor.get_feature_names_out())}")
    print(f"Files written: {', '.join(datasets)}")


if __name__ == "__main__":
    main()
