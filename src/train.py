"""Model construction and training utilities.

The preprocessing lives INSIDE every model pipeline (ColumnTransformer ->
estimator), so encoders and scalers are fit exclusively on whatever data the
pipeline is fit on — training folds during cross-validation, the training set
for the final model. Preprocessing the full dataset up front is the leakage
pattern this design exists to prevent.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    DECISION_THRESHOLD,
    FINAL_MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_cleaning import clean_telco, encode_target, load_raw
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    engineer_features,
)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the engineered feature frame.

    Stratification preserves the 26.5% churn rate in both halves — with a
    moderate class imbalance, an unlucky split could otherwise skew the
    prevalence either side and distort every metric measured later.
    """
    X = df[FEATURE_COLUMNS]
    y = encode_target(df["Churn"])
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Numeric scaling + categorical one-hot, as a fittable transformer.

    - StandardScaler matters for logistic regression (comparable coefficient
      scales, better conditioning); trees are scale-invariant and unharmed.
      One shared preprocessor keeps every model comparison apples-to-apples.
    - OneHotEncoder(drop="if_binary") gives binary columns a single indicator
      (cleaner coefficients); multi-level columns keep all levels.
    - handle_unknown="ignore": an unseen category at scoring time encodes as
      all-zeros instead of crashing production.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model) -> Pipeline:
    """Wrap an estimator with the standard preprocessor."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def candidate_models(class_weighted: bool = False) -> dict[str, Pipeline]:
    """The model lineup, each as a full preprocessing+model pipeline.

    With class_weighted=True, models that support it reweight the minority
    class inversely to prevalence (class_weight="balanced" /
    scale_pos_weight≈2.77). sklearn's GradientBoostingClassifier accepts no
    class_weight parameter, so it appears only in the unweighted lineup; its
    role is covered by XGBoost, which does.
    """
    # ~ (1 - 0.265) / 0.265 from the training prevalence; fixed constant so
    # the pipeline is deterministic and fold-independent.
    scale_pos_weight = 2.77

    weight = "balanced" if class_weighted else None
    models: dict[str, Pipeline] = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight=weight)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=weight)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight=weight,
            )
        ),
        "XGBoost": make_pipeline(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                tree_method="hist",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight if class_weighted else 1.0,
            )
        ),
    }
    if not class_weighted:
        models["Gradient Boosting"] = make_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        )
        models["Majority baseline"] = make_pipeline(
            DummyClassifier(strategy="most_frequent")
        )
    return models


def final_model() -> Pipeline:
    """The selected final model: tuned logistic regression (C=1.0).

    Chosen in notebook 06: statistically tied with tuned gradient boosting and
    XGBoost on CV PR-AUC (all within ~0.004, a fraction of one fold-std) while
    having the lowest fold variance, directly interpretable coefficients, and
    the smallest operational footprint.
    """
    return make_pipeline(LogisticRegression(max_iter=2000, C=1.0))


def main() -> None:
    """Train the final pipeline on the training split and save it.

    Rebuilds features from the raw file so this script is self-contained:
    raw CSV -> clean -> engineer -> split -> fit -> save.
    """
    import joblib

    df = engineer_features(clean_telco(load_raw())[0])
    X_train, _X_test, y_train, _y_test = split_data(df)
    pipe = final_model()
    pipe.fit(X_train, y_train)

    FINAL_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": DECISION_THRESHOLD,
            "feature_columns": FEATURE_COLUMNS,
            "n_train": len(X_train),
        },
        FINAL_MODEL_FILE,
    )
    print(f"Trained on {len(X_train):,} rows; saved -> {FINAL_MODEL_FILE}")
    print(f"Decision threshold: {DECISION_THRESHOLD}")
    print("(Test-set evaluation lives in notebook 06 — run there, reported once.)")


if __name__ == "__main__":
    main()
