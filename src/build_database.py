"""Load the project data into SQLite for SQL analytics.

Creates data/processed/telco_churn.db with two tables:

- customers    — cleaned dataset + engineered features (one row per customer)
- predictions  — final-model churn probability, risk segment, and an
                 is_test_set flag (1 = row was in the held-out test split).

The flag matters for honest analysis: scores on training rows are in-sample
and look better than reality; queries that report model *performance* should
filter to is_test_set = 1, while operational queries (rank everyone for
outreach) legitimately use all rows.

Run:

    python -m src.build_database
"""

from __future__ import annotations

import sqlite3

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, FINAL_MODEL_FILE, SQLITE_DB_FILE
from src.data_cleaning import clean_telco, load_raw
from src.feature_engineering import FEATURE_COLUMNS, engineer_features
from src.train import split_data

HIGH_RISK_CUTOFF = 0.65  # matches the segmentation in notebook 07


def build_database() -> None:
    df = engineer_features(clean_telco(load_raw())[0])

    bundle = joblib.load(FINAL_MODEL_FILE)
    pipe = bundle["pipeline"]

    # Reproduce the canonical split so test rows can be flagged.
    _, X_test, _, _ = split_data(df)
    is_test = df.index.isin(X_test.index)

    proba = pipe.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    segment = pd.cut(
        proba,
        [0, DECISION_THRESHOLD, HIGH_RISK_CUTOFF, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    predictions = pd.DataFrame(
        {
            "customerID": df["customerID"],
            "churn_probability": proba.round(4),
            "risk_segment": segment.astype(str),
            "is_test_set": is_test.astype(int),
        }
    )

    SQLITE_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_DB_FILE) as con:
        df.to_sql("customers", con, if_exists="replace", index=False)
        predictions.to_sql("predictions", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_pred_id ON predictions(customerID)")
        n_c = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_p = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"{SQLITE_DB_FILE}: customers={n_c:,}, predictions={n_p:,}")


if __name__ == "__main__":
    build_database()
