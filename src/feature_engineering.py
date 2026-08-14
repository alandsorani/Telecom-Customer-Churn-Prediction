"""Feature engineering for the Telco churn dataset.

Every transformation here is STATELESS and ROW-WISE: it uses no statistic
computed across rows (no means, no quantiles, no fitted encoders). That
property is what makes it leakage-safe to apply before the train/test split —
each row's features depend only on that row — and trivially reproducible on
unseen data, including the single-row inputs from the Streamlit app.

Anything that requires fitting (one-hot encoding, scaling) deliberately lives
in the sklearn Pipeline (src/train.py), where it is fit on training folds only.

Run as a script to produce the feature dataset:

    python -m src.feature_engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_FILE, DATA_PROCESSED

FEATURES_DATA_FILE = DATA_PROCESSED / "telco_churn_features.csv"

# Add-on services grouped by the EDA finding (notebook 03): support/protection
# add-ons show a strong retention association; streaming add-ons show almost none.
PROTECTIVE_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]

# Columns carrying a structural third level that duplicates InternetService/
# PhoneService information (verified exactly redundant in notebook 01).
STRUCTURAL_LEVEL_MAP = {"No internet service": "No", "No phone service": "No"}

# Fixed, a-priori tenure bin edges (months). Fixed — not data-derived quantiles —
# so the binning never shifts between training runs or drifts in production,
# and the open-ended last bin absorbs any future tenure > 72.
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]

# Feature lists consumed by train.py and the app. TotalCharges and customerID
# are deliberately absent — see notebook 05 for the evidence behind each choice.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "num_protective", "num_streaming"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group", "auto_pay",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned Telco frame.

    Pure function: returns a new frame, preserves row count and order, and
    depends on each row in isolation (no cross-row state).
    """
    out = df.copy()

    # Service counts, computed BEFORE structural-level collapsing (either order
    # gives the same result — only "Yes" is counted — but being explicit here
    # makes the invariant obvious).
    out["num_protective"] = sum(
        (out[c] == "Yes").astype("int64") for c in PROTECTIVE_SERVICES
    )
    out["num_streaming"] = sum(
        (out[c] == "Yes").astype("int64") for c in STREAMING_SERVICES
    )

    # Collapse structural levels: "No internet service" -> "No" etc. The
    # information is not lost — InternetService/PhoneService carry it — and the
    # one-hot space shrinks by 7 redundant columns.
    for col in PROTECTIVE_SERVICES + STREAMING_SERVICES + ["MultipleLines"]:
        out[col] = out[col].replace(STRUCTURAL_LEVEL_MAP)

    # Lifecycle stage on fixed edges; string dtype (not pandas Categorical)
    # keeps downstream one-hot encoding simple.
    out["tenure_group"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype("str")

    # Manual vs automatic payment — the divide the EDA found (notebook 03):
    # both automatic methods churn at 15-17%, both manual ones far higher.
    out["auto_pay"] = np.where(
        out["PaymentMethod"].str.contains("automatic"), "Yes", "No"
    )

    return out


def main() -> None:
    """Engineer features for the cleaned dataset and save the result."""
    df = pd.read_csv(CLEAN_DATA_FILE)
    out = engineer_features(df)
    FEATURES_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURES_DATA_FILE, index=False)
    print(f"{out.shape[0]:,} rows, {out.shape[1]} cols -> {FEATURES_DATA_FILE}")
    print(f"Model features: {len(FEATURE_COLUMNS)} "
          f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")


if __name__ == "__main__":
    main()
