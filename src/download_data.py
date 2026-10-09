"""Download and validate the UCI South German Credit dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/573/"
    "south%2Bgerman%2Bcredit%2Bupdate.zip"
)
ARCHIVE_SHA256 = "0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551"
SOURCE_SHA256 = "5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f"

SOURCE_COLUMNS = [
    "laufkont",
    "laufzeit",
    "moral",
    "verw",
    "hoehe",
    "sparkont",
    "beszeit",
    "rate",
    "famges",
    "buerge",
    "wohnzeit",
    "verm",
    "alter",
    "weitkred",
    "wohn",
    "bishkred",
    "beruf",
    "pers",
    "telef",
    "gastarb",
    "kredit",
]

COLUMN_NAMES = {
    "laufkont": "checking_account_status",
    "laufzeit": "duration_months",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "credit_amount_dm",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age_years",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "credit-risk-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def validate_dataset(data: pd.DataFrame) -> None:
    """Raise a clear error when the normalized dataset violates source expectations."""

    expected_columns = [COLUMN_NAMES[name] for name in SOURCE_COLUMNS] + ["bad_credit"]
    if list(data.columns) != expected_columns:
        raise ValueError("Dataset columns do not match the documented UCI schema.")
    if data.shape != (1000, 22):
        raise ValueError(f"Expected 1,000 rows and 22 columns; received {data.shape}.")
    if data.isna().any().any():
        raise ValueError("The source declares no missing values, but missing values were found.")
    if set(data["credit_risk"].unique()) != {0, 1}:
        raise ValueError("credit_risk must contain only the documented values 0 and 1.")
    if not data["bad_credit"].equals(1 - data["credit_risk"]):
        raise ValueError("bad_credit must be the inverse of the documented credit_risk label.")


def load_source_bytes(source_bytes: bytes) -> pd.DataFrame:
    """Parse, normalize, and validate the official source file."""

    if _sha256(source_bytes) != SOURCE_SHA256:
        raise ValueError("SouthGermanCredit.asc checksum does not match the verified source.")

    data = pd.read_csv(io.BytesIO(source_bytes), sep=r"\s+")
    if list(data.columns) != SOURCE_COLUMNS:
        raise ValueError("Source headers do not match the documented UCI schema.")

    data = data.rename(columns=COLUMN_NAMES)
    data["bad_credit"] = 1 - data["credit_risk"]
    validate_dataset(data)
    return data


def download_dataset(output_dir: Path, force: bool = False) -> Path:
    """Download the verified archive and write normalized CSV plus source documentation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "banking_data.csv"
    archive_path = output_dir / "south_german_credit_uci.zip"
    source_path = output_dir / "SouthGermanCredit.asc"
    code_table_path = output_dir / "codetable.txt"
    reader_path = output_dir / "read_SouthGermanCredit.R"

    if csv_path.exists() and not force:
        existing = pd.read_csv(csv_path)
        validate_dataset(existing)
        return csv_path

    archive = _download_bytes(DATASET_URL)
    if _sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Downloaded archive checksum does not match the verified UCI archive.")

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        source = bundle.read("SouthGermanCredit.asc")
        code_table = bundle.read("codetable.txt")
        reader = bundle.read("read_SouthGermanCredit.R")

    data = load_source_bytes(source)

    archive_path.write_bytes(archive)
    source_path.write_bytes(source)
    code_table_path.write_bytes(code_table)
    reader_path.write_bytes(reader)
    data.to_csv(csv_path, index=False)
    return csv_path


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = download_dataset(args.output_dir, force=args.force)
    data = pd.read_csv(path)
    print(f"Saved {len(data):,} validated rows to {path}")
    print(f"Bad-credit rate: {data['bad_credit'].mean():.1%}")


if __name__ == "__main__":
    main()
