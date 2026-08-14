"""Scoring interface for new customers.

Accepts raw customer attributes (the cleaned-data schema, minus customerID,
TotalCharges, and Churn), runs the exact same feature engineering used in
training, and returns churn probability, predicted class, and risk segment.

Used by the Streamlit app; also runnable as a demo:

    python -m src.predict
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, FINAL_MODEL_FILE
from src.feature_engineering import FEATURE_COLUMNS, engineer_features

HIGH_RISK_CUTOFF = 0.65  # Medium/High boundary, as segmented in notebook 07

# Input schema: the cleaned representation a caller must supply.
# TotalCharges is deliberately absent - the model does not use it (notebook 05),
# which conveniently spares the app from asking users to invent a lifetime total.
INPUT_SCHEMA: dict[str, list | tuple] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "tenure": (0, 1200),  # months; open-ended in production, sanity-capped here
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
    "MonthlyCharges": (0.0, 10_000.0),
}

_BUNDLE = None


def load_bundle(path=FINAL_MODEL_FILE) -> dict:
    """Load (and cache) the trained pipeline + threshold bundle."""
    global _BUNDLE
    if _BUNDLE is None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - train the model first: python -m src.train"
            )
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError listing every schema problem, not just the first."""
    problems = []
    for col, allowed in INPUT_SCHEMA.items():
        if col not in df.columns:
            problems.append(f"missing column: {col}")
            continue
        if isinstance(allowed, tuple):
            lo, hi = allowed
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                problems.append(f"{col}: {len(bad)} value(s) outside [{lo}, {hi}]")
        else:
            unknown = set(df[col].unique()) - set(allowed)
            if unknown:
                problems.append(f"{col}: unexpected value(s) {sorted(unknown)}")
    if problems:
        raise ValueError("Invalid input - " + "; ".join(problems))


def risk_segment(probability: float) -> str:
    """Map a churn probability to the operational risk tier (notebook 07)."""
    if probability >= HIGH_RISK_CUTOFF:
        return "High"
    if probability >= DECISION_THRESHOLD:
        return "Medium"
    return "Low"


def predict_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Score a frame of customers. Returns probability, class, and segment."""
    validate_input(raw)
    bundle = load_bundle()
    features = engineer_features(raw)[FEATURE_COLUMNS]
    proba = bundle["pipeline"].predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": proba,
            "predicted_churn": proba >= bundle["threshold"],
            "risk_segment": [risk_segment(p) for p in proba],
        },
        index=raw.index,
    )


def predict_customer(**attributes) -> dict:
    """Score a single customer given keyword attributes; returns a dict."""
    result = predict_customers(pd.DataFrame([attributes]))
    row = result.iloc[0]
    return {
        "churn_probability": float(row["churn_probability"]),
        "predicted_churn": bool(row["predicted_churn"]),
        "risk_segment": str(row["risk_segment"]),
    }


def main() -> None:
    """Demo: score two archetypal customers."""
    high_risk = dict(  # noqa: C408 - kwargs form mirrors predict_customer(**attributes)
        gender="Female", SeniorCitizen="Yes", Partner="No", Dependents="No",
        tenure=2, PhoneService="Yes", MultipleLines="No",
        InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
        StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
        PaymentMethod="Electronic check", MonthlyCharges=95.0,
    )
    low_risk = dict(  # noqa: C408
        gender="Male", SeniorCitizen="No", Partner="Yes", Dependents="Yes",
        tenure=68, PhoneService="Yes", MultipleLines="Yes",
        InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
        StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)", MonthlyCharges=61.0,
    )
    for name, profile in [("high-risk archetype", high_risk),
                          ("low-risk archetype", low_risk)]:
        r = predict_customer(**profile)
        print(f"{name}: p(churn)={r['churn_probability']:.3f} "
              f"predicted={'CHURN' if r['predicted_churn'] else 'stay'} "
              f"segment={r['risk_segment']}")


if __name__ == "__main__":
    main()
