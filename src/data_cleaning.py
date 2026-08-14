"""Reproducible cleaning pipeline for the Telco Customer Churn dataset.

Design principles:
- Every transformation is a named, documented step that reports how many
  records it affected. Nothing is dropped or altered silently.
- The same functions clean the training data, future unseen data, and the
  rows entered in the Streamlit app — eliminating train/serve skew.
- Unexpected data (schema drift, impossible values) raises an error instead
  of being papered over.

Run as a script to produce the processed dataset:

    python -m src.data_cleaning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_DATA_FILE,
    DATASET_SOURCE_URL,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


@dataclass
class CleaningStep:
    """Record of one cleaning operation, for transparent reporting."""

    name: str
    problem: str
    action: str
    rows_affected: int

    def __str__(self) -> str:
        return (
            f"[{self.name}] {self.problem}\n"
            f"    action: {self.action}\n"
            f"    rows affected: {self.rows_affected}"
        )


def load_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV and verify the schema is the one we expect.

    Raises:
        FileNotFoundError: if the Kaggle CSV has not been downloaded.
        ValueError: if columns differ from the published schema.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Download the dataset from {DATASET_SOURCE_URL} and place it at "
            f"{path} — see data/README.md or the main README's Installation "
            "section for full instructions."
        )
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected columns in raw data. "
            f"Missing: {set(EXPECTED_COLUMNS) - set(df.columns)}; "
            f"unexpected: {set(df.columns) - set(EXPECTED_COLUMNS)}"
        )
    return df


def clean_telco(df: pd.DataFrame) -> tuple[pd.DataFrame, list[CleaningStep]]:
    """Apply the full cleaning pipeline.

    Returns the cleaned frame and a step-by-step report of what changed.
    Row count is preserved unless exact duplicates are found (none exist in
    the published dataset; the guard is for reproducibility on other extracts).
    """
    df = df.copy()
    steps: list[CleaningStep] = []

    # --- 1. Trim whitespace in string columns -------------------------------
    # The published file's only whitespace problem is the blank " " in
    # TotalCharges, but trimming everywhere makes the pipeline robust to
    # future extracts with sloppier formatting.
    str_cols = df.columns[df.dtypes.map(lambda t: t not in ("int64", "float64"))]
    n_ws = int(
        sum((df[c].astype(str) != df[c].astype(str).str.strip()).sum() for c in str_cols)
    )
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    steps.append(CleaningStep(
        name="trim_whitespace",
        problem="String cells may carry leading/trailing whitespace "
                "(in this file: only the blank TotalCharges values).",
        action="str.strip() applied to every string column.",
        rows_affected=n_ws,
    ))

    # --- 2. Exact duplicates ------------------------------------------------
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates()
    steps.append(CleaningStep(
        name="drop_duplicates",
        problem="Exact duplicate rows would double-count customers.",
        action="drop_duplicates() — none exist in the published dataset.",
        rows_affected=n_dup,
    ))

    # --- 3. TotalCharges: string -> numeric, resolve hidden blanks ----------
    # 11 rows hold "" (after trimming); all are tenure-0 customers who have
    # not yet been billed. We impute 0.0 rather than drop them:
    #   * 0 is the economically true value — nothing has been billed yet;
    #   * dropping would silently delete the entire "brand-new customer"
    #     segment, which a deployed model must be able to score;
    #   * 11 rows (0.16%) cannot meaningfully shift any statistic either way.
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = tc.isna()
    inconsistent = df.loc[blank_mask, "tenure"].ne(0)
    if inconsistent.any():
        # A blank bill for a tenured customer would be real missingness, not
        # the structural tenure-0 pattern — refuse to guess.
        raise ValueError(
            f"{int(inconsistent.sum())} rows have unparseable TotalCharges but "
            "tenure > 0; investigate before imputing."
        )
    df["TotalCharges"] = tc.fillna(0.0)
    steps.append(CleaningStep(
        name="fix_total_charges",
        problem="TotalCharges stored as text; 11 blank values hide from isna().",
        action="Coerced to float; blanks (all tenure-0, verified) imputed as 0.0.",
        rows_affected=int(blank_mask.sum()),
    ))

    # --- 4. SeniorCitizen: 0/1 int -> No/Yes categorical --------------------
    # Same information, but consistent with every other demographic flag and
    # far more readable in plots, SHAP output, and the app.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if df["SeniorCitizen"].isna().any():
        raise ValueError("SeniorCitizen contained values other than 0/1.")
    steps.append(CleaningStep(
        name="recode_senior_citizen",
        problem="SeniorCitizen encoded 0/1 while sibling flags are No/Yes.",
        action='Mapped {0: "No", 1: "Yes"}; now treated as categorical.',
        rows_affected=len(df),
    ))

    # --- 5. Validate value ranges ------------------------------------------
    # Cheap invariants that would catch corrupted or drifted future extracts.
    problems = []
    if (df["tenure"] < 0).any():
        problems.append("negative tenure")
    if (df["MonthlyCharges"] <= 0).any():
        problems.append("non-positive MonthlyCharges")
    if (df["TotalCharges"] < 0).any():
        problems.append("negative TotalCharges")
    if not set(df[TARGET_COLUMN].unique()) <= {"Yes", "No"}:
        problems.append("unexpected Churn labels")
    if problems:
        raise ValueError(f"Validation failed: {', '.join(problems)}")
    steps.append(CleaningStep(
        name="validate_ranges",
        problem="Corrupt extracts could carry impossible values.",
        action="Checked tenure >= 0, MonthlyCharges > 0, TotalCharges >= 0, "
               "Churn in {Yes, No}. All passed.",
        rows_affected=0,
    ))

    return df, steps


def encode_target(churn: pd.Series) -> pd.Series:
    """Encode Churn Yes/No -> 1/0.

    Kept separate from clean_telco: the processed CSV stays human-readable
    (and directly usable for SQL/EDA); encoding happens at modeling time.
    """
    if not set(churn.unique()) <= {"Yes", "No"}:
        raise ValueError(f"Unexpected target labels: {set(churn.unique())}")
    return churn.map({"Yes": 1, "No": 0}).astype("int64")


def main() -> None:
    """Clean the raw file, print the report, save the processed CSV."""
    raw = load_raw()
    clean, steps = clean_telco(raw)
    print(f"Raw:   {raw.shape[0]:,} rows x {raw.shape[1]} cols")
    print(f"Clean: {clean.shape[0]:,} rows x {clean.shape[1]} cols\n")
    for step in steps:
        print(step, end="\n\n")
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_DATA_FILE, index=False)
    print(f"Saved -> {CLEAN_DATA_FILE}")


if __name__ == "__main__":
    main()
