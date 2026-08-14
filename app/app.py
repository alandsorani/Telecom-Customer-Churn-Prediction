"""Telco Churn Risk — Streamlit demo application.

Two pages: score an individual customer (with explanation), and a findings
dashboard. Run from the project root:

    streamlit run app/app.py

NOT production-ready — see the deployment notes in the app footer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import DECISION_THRESHOLD, FIGURES_DIR
from src.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURES_DATA_FILE,
    engineer_features,
)
from src.predict import (
    HIGH_RISK_CUTOFF,
    INPUT_SCHEMA,
    load_bundle,
    predict_customer,
)

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#898781"
TIER_STYLE = {
    "Low": ("#0ca30c", "Low risk"),
    "Medium": ("#eda100", "Medium risk"),
    "High": ("#d03b3b", "High risk"),
}
RECOMMENDED_ACTIONS = {
    "Low": [
        "No retention intervention indicated — standard lifecycle marketing.",
        "If on month-to-month, include in routine contract-upgrade campaigns.",
    ],
    "Medium": [
        "Include in the next retention wave (targeted offer, A/B tested).",
        "If paying by electronic check, offer the auto-pay enrollment credit.",
        "If no protective add-ons, offer a free TechSupport/OnlineSecurity trial.",
    ],
    "High": [
        "Priority outreach within the current cycle (72% of this tier churned in testing).",
        "Lead with a discounted 12-month contract conversion offer.",
        "Bundle a free support/security add-on trial into the offer.",
        "Route to the retention specialist queue, not generic marketing.",
    ],
}

st.set_page_config(page_title="Telco Churn Risk", page_icon="📉", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_resource
def get_explainer():
    """SHAP linear explainer over the training-distribution background."""
    import shap

    bundle = get_bundle()
    pre = bundle["pipeline"].named_steps["preprocess"]
    background = pd.read_csv(FEATURES_DATA_FILE)[FEATURE_COLUMNS]
    bg = pre.transform(background)
    bg = np.asarray(bg.toarray() if hasattr(bg, "toarray") else bg)
    explainer = shap.Explainer(
        bundle["pipeline"].named_steps["model"],
        shap.maskers.Independent(bg, max_samples=1000),
        feature_names=list(pre.get_feature_names_out()),
    )
    return explainer, pre


def pretty_feature(name: str) -> str:
    """'Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    for col in ["InternetService", "Contract", "PaymentMethod", "tenure_group"]:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    if name.endswith("_Yes"):
        return name[:-4] + " = Yes"
    return name


def score_page() -> None:
    st.subheader("Score a customer")
    st.caption(
        "Enter the customer's current attributes. The model returns a churn "
        "probability, a risk tier, suggested next actions, and the factors "
        "behind the score."
    )

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account**")
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            contract = st.selectbox("Contract", INPUT_SCHEMA["Contract"])
            monthly = st.number_input("Monthly charges ($)", 18.0, 150.0, 70.0, step=0.5)
            payment = st.selectbox("Payment method", INPUT_SCHEMA["PaymentMethod"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with c2:
            st.markdown("**Services**")
            internet = st.selectbox("Internet service", INPUT_SCHEMA["InternetService"])
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes"])
            security = st.selectbox("Online security", ["No", "Yes"])
            backup = st.selectbox("Online backup", ["No", "Yes"])
            protection = st.selectbox("Device protection", ["No", "Yes"])
            support = st.selectbox("Tech support", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming movies", ["No", "Yes"])
        with c3:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", INPUT_SCHEMA["gender"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        st.info("Fill in the profile and press **Predict churn risk**.")
        return

    # Structural consistency: no internet/phone forces the dependent fields.
    if internet == "No":
        security = backup = protection = support = tv = movies = "No internet service"
    if phone == "No":
        multiple = "No phone service"

    profile = dict(  # noqa: C408 - kwargs form mirrors predict_customer(**attributes)
        gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents,
        tenure=int(tenure), PhoneService=phone, MultipleLines=multiple,
        InternetService=internet, OnlineSecurity=security, OnlineBackup=backup,
        DeviceProtection=protection, TechSupport=support, StreamingTV=tv,
        StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless,
        PaymentMethod=payment, MonthlyCharges=float(monthly),
    )
    result = predict_customer(**profile)
    proba, tier = result["churn_probability"], result["risk_segment"]
    color, label = TIER_STYLE[tier]

    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric("Churn probability", f"{proba:.1%}")
    m2.markdown(
        f"<div style='margin-top:1.4rem;padding:0.45rem 1rem;border-radius:999px;"
        f"display:inline-block;background:{color};color:white;font-weight:700'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )
    m3.progress(min(proba, 1.0), text=f"Decision threshold {DECISION_THRESHOLD:.3f} · "
                                      f"High-risk cutoff {HIGH_RISK_CUTOFF:.2f}")

    st.markdown("**Recommended next actions**")
    for action in RECOMMENDED_ACTIONS[tier]:
        st.markdown(f"- {action}")

    st.markdown("**Why this score** *(model factor contributions — associative, not causal)*")
    try:
        explainer, pre = get_explainer()
        x = pre.transform(engineer_features(pd.DataFrame([profile]))[FEATURE_COLUMNS])
        x = np.asarray(x.toarray() if hasattr(x, "toarray") else x)
        sv = explainer(x)
        contrib = pd.Series(sv.values[0], index=sv.feature_names)
        values = pd.Series(x[0], index=sv.feature_names)
        top = pd.concat([contrib.sort_values().head(4), contrib.sort_values().tail(4)])
        top = top[top.abs() > 0.01]

        def label(f):
            # One-hot indicator at 0 => the customer does NOT have this level;
            # say so, otherwise "Contract = Two year raising risk" reads backwards.
            if "_" in f and values[f] == 0:
                return pretty_feature(f).replace(" = ", " ≠ ")
            return pretty_feature(f)

        fig, ax = plt.subplots(figsize=(7, 0.4 * len(top) + 0.7))
        ax.barh([label(f) for f in top.index], top.values,
                color=[BLUE if v < 0 else ORANGE for v in top.values], height=0.6)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel("Contribution to churn score (log-odds)", fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
    except Exception as exc:  # noqa: BLE001 - explanation is best-effort; scoring already shown
        st.caption(f"Explanation unavailable ({exc}).")


def dashboard_page() -> None:
    st.subheader("Project findings")
    st.caption(
        "Headline numbers from the analysis. All metrics were computed on the "
        "held-out test set (notebooks 06–07); nothing here is estimated or invented."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall churn rate", "26.5%")
    k2.metric("Model ROC-AUC (test)", "0.845")
    k3.metric("Churners caught @ threshold", "73.0%", help="Recall at the 0.338 operating threshold")
    k4.metric("Top-20% targeting captures", "49.7%", help="Share of all churners in the top quintile by predicted risk — 2.5x random outreach")

    st.divider()
    figures = [
        ("02_churn_by_contract.png", "Churn by contract — the 15x gradient"),
        ("10_contract_x_internet.png", "Where risk concentrates"),
        ("18_risk_segments.png", "Risk tiers on held-out data"),
        ("19_gains_curve.png", "Model targeting vs random outreach"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figures):
        path = FIGURES_DIR / fname
        if path.exists():
            cols[i % 2].image(str(path), caption=caption, use_container_width=True)

    st.divider()
    st.markdown(
        """
**The churn-prone profile** (composite of observed associations): a newer month-to-month
customer on fiber internet with a high monthly bill, paying by electronic check, without
support/security add-ons.

**Top recommendations** (each phrased as a testable hypothesis — see notebook 07):
1. Early-lifecycle program for month-to-month fiber customers (54.6% observed churn).
2. Randomized auto-pay migration campaign for electronic-check payers (45.3% vs 15–17%).
3. Attach protective add-ons — not streaming — to retention offers (42% vs 15% churn).
"""
    )


st.title("📉 Telco Customer Churn — Risk Scoring")
page = st.sidebar.radio("Page", ["Score a customer", "Project findings"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** logistic regression pipeline  \n"
    f"**Test ROC-AUC:** 0.845 · **PR-AUC:** 0.650  \n"
    f"**Threshold:** {DECISION_THRESHOLD:.3f} (max-F1, out-of-fold)  \n"
    f"**Trained on:** {get_bundle()['n_train']:,} customers"
)

if page == "Score a customer":
    score_page()
else:
    dashboard_page()

with st.expander("⚠️ This is a demo, not a production system"):
    st.markdown(
        """
A real deployment would additionally require: authentication and audit logging; a model
registry with versioning and rollback; input validation against live billing-system schemas
(not hand-entered forms); scheduled retraining with **drift monitoring** (input distributions
and calibration); an A/B measurement framework so retention lift is measured against holdout
controls rather than assumed; latency/SLA guarantees behind an API rather than a notebook-grade
app; and review of fairness implications before using demographic attributes operationally.
The dataset is also a single IBM sample snapshot — patterns here demonstrate method, not any
real carrier's economics.
"""
    )
