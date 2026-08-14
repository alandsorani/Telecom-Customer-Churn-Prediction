# Telco Customer Churn — Prediction & Retention Analytics

An end-to-end data science project: identify telecom customers at high risk of churning,
explain the drivers behind that risk, and turn the analysis into specific, testable retention
recommendations — with a reproducible pipeline, SQL analytics, and a Streamlit scoring app.

> **Honesty contract:** every metric in this README was computed by the code in this
> repository on the actual dataset. Associations are never presented as causes, and every
> business assumption is labeled as one.

---

## Business Problem

Telecom is a saturated subscription market: growth mostly means keeping the customers you
have. In this dataset, roughly **1 in 4 customers churned in a single month**. Retention
offers are wasted on customers who were going to stay and too late for customers already
gone — so the company needs to know **who is at risk, why, and what to do about it**:

> Can we identify customers at high risk of churning, understand the factors associated with
> their risk, and provide actionable recommendations to improve retention?

## Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(originally an IBM sample dataset). **7,043 customers × 21 columns**: demographics, subscribed
services, contract/billing attributes, and a Yes/No churn label. Single snapshot — no
timestamps, no cost data (both limitations are handled explicitly; see Limitations).

The CSV itself is not committed (see [`data/README.md`](data/README.md) for the source link,
license note, and expected file layout — that file, plus the `DATASET_SOURCE_URL` constant in
[`src/config.py`](src/config.py), is the version-controlled link to the external dataset).

The CSV is **not** committed to this repo (size/licensing hygiene); see Installation.

## Objectives

1. Quantify and test the factors associated with churn (EDA + formal statistics).
2. Build and compare churn models against honest baselines, without data leakage.
3. Optimize for business-relevant metrics (recall / PR-AUC), including the decision threshold.
4. Explain predictions globally and per customer; segment the base by risk.
5. Deliver evidence-based retention recommendations and a targeting simulation.
6. Ship it reproducibly: pipeline scripts, SQL layer, scoring app, this document.

## Methodology (notebook by notebook)

| Notebook | What happens there |
|---|---|
| `01_data_understanding` | Structure, data dictionary, quality issues (incl. the hidden `TotalCharges` blanks) |
| `02_data_cleaning` | Reproducible cleaning via `src/data_cleaning.py`; every fix justified and counted |
| `03_exploratory_analysis` | Question-driven EDA; 10 portfolio figures; confounds flagged |
| `04_statistical_analysis` | χ² and Mann-Whitney tests with effect sizes + a negative control |
| `05_feature_engineering` | Features added *and rejected*, each with evidence; leakage analysis |
| `06_modeling` | Baselines → 5-model CV comparison → imbalance strategies → tuning → threshold → single test evaluation |
| `07_model_interpretation` | Coefficients, permutation importance, SHAP, risk tiers, retention simulation, recommendations |

### Data cleaning (summary)

7,043 rows in → **7,043 rows out; nothing dropped silently (or at all)**. The dataset's one real
defect: `TotalCharges` is stored as text with 11 blank `" "` values — all brand-new (tenure-0)
customers. Blanks were imputed as `0.0` (the economically true value; dropping would delete the
new-customer segment a deployed model must score). `SeniorCitizen` recoded 0/1 → No/Yes for
consistency. All operations are functions in `src/data_cleaning.py` that report row counts and
validate invariants — the same code cleans training data and app inputs.

### Exploratory analysis — key associations (all computed)

- **Contract:** month-to-month churns at **42.7%**, one-year 11.3%, two-year **2.8%** (15×).
- **Tenure:** first-6-months customers churn at **53%**, falling monotonically to 7% at 61–72 months.
- **Payment:** electronic check churns at **45.3%** vs 15–17% for auto-pay; e-check payers are
  34% of customers but **57% of all churners**.
- **Internet:** fiber churns at **41.9%** — 2.2× DSL (19.0%) — while costing ~$35/month more.
- **Add-on asymmetry:** without TechSupport/OnlineSecurity, internet customers churn at ~42%
  vs ~15% with them; streaming add-ons show no such association (33.5% vs 30.1%).
- **Compound risk:** month-to-month + fiber = **54.6% churn** (n=2,128); two-year + no
  internet = 0.8%.
- **Null results matter:** gender and phone service show no association (verified statistically).

### Statistical analysis

Chi-square tests (with expected-count checks) for categoricals, Mann-Whitney U for the
non-normal numerics, Bonferroni-adjusted α, and effect sizes leading: Contract **V = 0.41**,
InternetService V = 0.32, PaymentMethod V = 0.30, tenure rank-biserial **r = −0.48**,
MonthlyCharges r = +0.24 — and gender as a passing negative control (p = 0.487, V = 0.008).

### Feature engineering

Added (with evidence): `tenure_group` (fixed bins; churn 52.9% → 9.5%), `auto_pay` (16.0% vs
34.7%), `num_protective` / `num_streaming` counts (kept separate because the data shows the
asymmetry). Rejected (with evidence): `avg_charge_per_month` (r = 0.996 with MonthlyCharges),
`charge_growth` (churn correlation 0.002), `TotalCharges` as a model feature (r = 0.83 with
tenure; its unique component carries no signal). All transforms are stateless and row-wise —
provably safe to apply before the split, and verified by a subset-equivalence test.

### Machine learning

Stratified 80/20 split; **the test set was touched exactly once**, after all decisions.
Preprocessing (scaling + one-hot) lives inside every model's sklearn `Pipeline`, so
cross-validation folds never leak. Five algorithms + a majority-class dummy, stratified 5-fold
CV, GridSearchCV on PR-AUC for the finalists:

| Model (tuned) | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| XGBoost | 0.669 ± 0.021 | 0.850 |
| Gradient Boosting | 0.668 ± 0.020 | 0.850 |
| **Logistic Regression (selected)** | 0.665 ± 0.014 | 0.848 |

The three are statistically tied (spread ≈ ⅕ of a fold-std). **Logistic regression was selected
on secondary criteria**: lowest variance, readable coefficients, honest calibration, minimal
operational cost. Class imbalance was handled by evidence, not habit: class weights were shown
to move recall without improving ranking (ROC-AUC unchanged), SMOTE was evaluated and rejected
with reasons, and the decision threshold was optimized directly.

### Model evaluation (held-out test set, n = 1,409)

| Operating point | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Default 0.50 | 0.801 | 0.660 | 0.513 | 0.577 | 0.845 | 0.650 |
| **Adopted 0.338** (max-F1 on out-of-fold train) | 0.775 | 0.558 | **0.730** | 0.633 | 0.845 | 0.650 |

At the adopted threshold the model catches **273 of 374 churners**. Probabilities are
calibrated (top decile predicts 76.5% vs 74.5% actual), which the risk tiers rely on.

## Key Findings

1. **The churn-prone profile:** a newer month-to-month customer on fiber with a high bill,
   paying by electronic check, without support/security add-ons.
2. **Risk tiers separate cleanly on held-out data:** Low 11.0% / Medium 47.7% / High 71.7%
   actual churn; the High tier averages 4.4 months of tenure.
3. **Model-ranked outreach beats random 2.5×:** contacting the top 20% by predicted risk
   reaches 49.7% of all actual churners at 66% precision.
4. **A simple model was enough:** boosted trees could not beat logistic regression here —
   the signal is dominated by additive main effects.

## Business Recommendations (associations → testable interventions)

1. **Early-lifecycle program for month-to-month fiber customers** (54.6% observed churn;
   High tier is 100% month-to-month, 94% fiber) — A/B test onboarding + discounted 12-month
   conversion at months 2–3; measure 6-month churn vs control.
2. **Auto-pay migration campaign** for electronic-check payers (45.3% vs 15–17%) —
   randomized enrollment credit; measure churn of enrolled vs control (selection effects are
   likely part of the raw gap, hence the randomization).
3. **Attach protective add-ons, not streaming, to retention offers** (~42% vs ~15% churn
   asymmetry) — free TechSupport/OnlineSecurity trial for high-risk fiber customers.
4. **Root-cause the fiber experience** before blanket discounts — the data can't separate
   price, quality, and customer mix.
5. **Deploy ranked outreach with a permanent holdout** so real retention lift becomes
   measurable rather than assumed.

## Project Architecture

```
telco-customer-churn/
├── README.md · requirements.txt · .gitignore
├── data/ raw/ (Kaggle CSV, git-ignored) · processed/ (clean CSV, features, SQLite DB)
├── notebooks/ 01…07 (run in order; all executed with real outputs)
├── src/
│   ├── config.py              # paths, seed, decision threshold — single source of truth
│   ├── data_cleaning.py       # cleaning pipeline (python -m src.data_cleaning)
│   ├── feature_engineering.py # stateless features (python -m src.feature_engineering)
│   ├── train.py               # pipelines, model lineup, final training (python -m src.train)
│   ├── evaluate.py            # CV comparison + scoring helpers
│   ├── predict.py             # scoring API + input validation (python -m src.predict)
│   └── build_database.py      # SQLite loader (python -m src.build_database)
├── sql/ customer_analysis.sql · churn_analysis.sql   # 14 executed queries
├── models/ final_churn_pipeline.joblib (git-ignored; regenerate via src.train)
├── reports/figures/ 19 exported charts
└── app/ app.py                # Streamlit demo
```

Design principle: **notebooks narrate, `src/` does the work.** The same functions clean,
engineer, and score in training, in the notebooks, and in the app — no train/serve skew.

## Installation

Requires **Python 3.11+** (developed on 3.12.14).

```bash
git clone https://github.com/alandsorani/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS note:** plain `python`/`pip` aren't on PATH by default — use `python3` to create the
venv; once activated, `python`/`pip` work normally inside it. XGBoost also needs the OpenMP
runtime: `brew install libomp`.

**Dataset** (choose one):
```bash
# Kaggle CLI (needs ~/.kaggle/kaggle.json API token)
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```
or download manually from the Kaggle page https://www.kaggle.com/datasets/blastchar/telco-customer-churn and place
`WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.

## How to Run

Full pipeline, raw CSV to trained model (run in order; each step writes the file noted):

| Command | Produces |
|---|---|
| `python -m src.data_cleaning` | `data/processed/telco_churn_clean.csv` |
| `python -m src.feature_engineering` | `data/processed/telco_churn_features.csv` |
| `python -m src.train` | `models/final_churn_pipeline.joblib` |
| `python -m src.build_database` | `data/processed/telco_churn.db` (SQL layer) |

```bash
python -m src.data_cleaning
python -m src.feature_engineering
python -m src.train
python -m src.build_database
```

Other entry points:

- **Notebooks:** `jupyter lab notebooks/` — run `01` through `07` in order.
- **SQL:** `sqlite3 -column -header data/processed/telco_churn.db` — queries live in `sql/`.
- **App:** `streamlit run app/app.py`

## Example Prediction

```python
from src.predict import predict_customer

predict_customer(
    gender="Female", SeniorCitizen="Yes", Partner="No", Dependents="No",
    tenure=2, PhoneService="Yes", MultipleLines="No",
    InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
    DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
    StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
    PaymentMethod="Electronic check", MonthlyCharges=95.0,
)
# {'churn_probability': 0.858, 'predicted_churn': True, 'risk_segment': 'High'}
```

## Limitations

- **Snapshot data:** no event dates → no temporal validation, no survival analysis, and the
  tenure–churn association is partly mechanical (leaving early truncates tenure).
- **No cost data:** financial scenarios use labeled, adjustable assumptions — never claims.
- **Associations, not causes:** customers self-select contracts and services; every
  recommendation is framed as an A/B-testable hypothesis.
- **Fictional IBM sample:** patterns demonstrate method, not a real carrier's economics.
- **Performance ceiling:** all models plateau near ROC-AUC 0.85 on these features; richer
  behavioral data (usage, support tickets, outages) would be the next unlock.

## Future Improvements

Survival analysis on time-stamped data; cost-sensitive threshold optimization with real
offer economics; drift monitoring + scheduled retraining; uplift modeling (who is
*persuadable*, not just who is at risk); fairness audit of demographic features before
operational use; API-based serving with a model registry.

## Technologies

Python 3.12 · pandas · scikit-learn · XGBoost · SciPy · SHAP · matplotlib · SQLite ·
Streamlit · joblib · Jupyter

---

*Built as a portfolio project. The dataset is public (IBM sample via Kaggle); all analysis
and code are original.*
