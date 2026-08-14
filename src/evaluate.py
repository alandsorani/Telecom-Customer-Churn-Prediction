"""Evaluation utilities: cross-validated model comparison and test scoring."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import RANDOM_STATE

# average_precision = area under the precision-recall curve (PR-AUC).
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    """Stratified K-fold: every fold keeps the 26.5% churn prevalence."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cv_compare(models: dict, X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validate each pipeline on the TRAINING data only.

    Returns mean metrics per model (one row each), sorted by PR-AUC — the
    most informative single number for an imbalanced ranking problem.
    """
    cv = make_cv(n_splits)
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            row[metric] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
    return out


def score_predictions(y_true, y_pred, y_proba) -> dict:
    """All headline metrics for one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
