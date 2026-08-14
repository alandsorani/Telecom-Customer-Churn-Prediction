"""Telco Churn Risk — Streamlit demo application.

Two pages: score an individual customer (with explanation), and a findings
dashboard. Run from the project root:

    streamlit run app/app.py

NOT production-ready — see the notes at the bottom of the app.
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

BLUE, ORANGE, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781"

TIER_PILL = {
    "Low": ("pill pill-low", "Low risk"),
    "Medium": ("pill pill-medium", "Medium risk"),
    "High": ("pill pill-high", "High risk"),
}
TIER_SUMMARY = {
    "Low": "This profile scores below the operating threshold — no retention "
           "intervention is indicated.",
    "Medium": "This profile is above the operating threshold. In testing, "
              "customers in this tier churned at 48%.",
    "High": "This profile is in the highest tier. In testing, customers in "
            "this tier churned at 72% — priority outreach is warranted.",
}
RECOMMENDED_ACTIONS = {
    "Low": [
        "Standard lifecycle marketing only.",
        "If on month-to-month, include in routine contract-upgrade campaigns.",
    ],
    "Medium": [
        "Include in the next retention wave (targeted offer, A/B tested).",
        "If paying by electronic check, offer the auto-pay enrollment credit.",
        "If no protective add-ons, offer a free TechSupport / OnlineSecurity trial.",
    ],
    "High": [
        "Priority outreach within the current cycle.",
        "Lead with a discounted 12-month contract conversion offer.",
        "Bundle a free support/security add-on trial into the offer.",
        "Route to the retention specialist queue, not generic marketing.",
    ],
}

st.set_page_config(page_title="Telco Churn Risk", layout="wide")

st.markdown(
    """
<style>
/* ---- chrome & typography ------------------------------------------------ */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1150px; }
h1, h2, h3 { letter-spacing: -0.01em; }

.app-header h1 { font-size: 1.85rem; margin-bottom: 0.1rem; }
.app-header p  { color: #52514e; margin: 0 0 0.4rem 0; font-size: 1rem; }

/* ---- risk pill ---------------------------------------------------------- */
.pill { display: inline-block; padding: 0.3rem 0.95rem; border-radius: 999px;
        font-weight: 600; font-size: 0.95rem; }
.pill-low    { background: rgba(12, 163, 12, 0.12);  color: #067806; }
.pill-medium { background: rgba(237, 161, 0, 0.16);  color: #8a5d00; }
.pill-high   { background: rgba(208, 59, 59, 0.12);  color: #b02a2a; }

/* ---- probability card --------------------------------------------------- */
.prob-label  { color: #52514e; font-size: 0.88rem; margin-bottom: 0.1rem; }
.prob-number { font-size: 2.7rem; font-weight: 700; color: #0b0b0b; line-height: 1.05; }

.scale { position: relative; height: 10px; border-radius: 999px; margin: 1.1rem 0 0.35rem;
         background: linear-gradient(90deg, #9ec5f4 0%, #f3cd7e 45%, #e79191 100%); }
.scale .tick   { position: absolute; top: -4px; width: 2px; height: 18px;
                 background: rgba(11, 11, 11, 0.3); }
.scale .marker { position: absolute; top: -5px; width: 20px; height: 20px; border-radius: 50%;
                 background: #0b0b0b; border: 3px solid #ffffff;
                 box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); transform: translateX(-50%); }
.scale-labels  { display: flex; justify-content: space-between; color: #898781;
                 font-size: 0.78rem; }

/* ---- images inside bordered containers collapse to 16px in Streamlit 1.50;
       force the image wrapper chain back to full width ---------------------- */
[data-testid="stLayoutWrapper"]:has([data-testid="stImage"]),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stImage"]),
[data-testid="stElementContainer"]:has([data-testid="stImage"]),
[data-testid="stFullScreenFrame"], [data-testid="stImage"],
[data-testid="stImageContainer"] { width: 100% !important; }
[data-testid="stImageContainer"] img { width: 100% !important; }

/* ---- stat cards (dashboard) --------------------------------------------- */
.stat-label { color: #52514e; font-size: 0.85rem; margin-bottom: 0.15rem; }
.stat-value { font-size: 1.9rem; font-weight: 700; color: #0b0b0b; line-height: 1.1; }
.stat-note  { color: #898781; font-size: 0.8rem; margin-top: 0.15rem; }
</style>
""",
    unsafe_allow_html=True,
)


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


def risk_scale_html(probability: float) -> str:
    """Gradient scale with threshold ticks and a marker at the customer's score."""
    p = min(max(probability, 0.0), 1.0) * 100
    t1, t2 = DECISION_THRESHOLD * 100, HIGH_RISK_CUTOFF * 100
    return (
        f'<div class="scale">'
        f'<div class="tick" style="left:{t1:.1f}%"></div>'
        f'<div class="tick" style="left:{t2:.1f}%"></div>'
        f'<div class="marker" style="left:{p:.1f}%"></div>'
        f"</div>"
        f'<div class="scale-labels"><span>0%</span>'
        f"<span>threshold {DECISION_THRESHOLD:.0%}</span>"
        f"<span>high risk {HIGH_RISK_CUTOFF:.0%}</span><span>100%</span></div>"
    )


def score_page() -> None:
    st.markdown("#### Customer profile")
    st.caption(
        "Fill in the customer's current details and select **Predict churn risk**. "
        "Nothing is stored — each prediction is computed on the spot."
    )

    with st.form("customer"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Account**")
            tenure = st.number_input(
                "Tenure (months)", 0, 120, 12,
                help="How long this customer has been with the company.",
            )
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
        st.caption("Results will appear here after you submit the profile.")
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
    pill_class, pill_label = TIER_PILL[tier]

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                f'<div class="prob-label">Predicted churn probability</div>'
                f'<div class="prob-number">{proba:.1%}</div>'
                f'<div style="margin-top:0.55rem"><span class="{pill_class}">{pill_label}</span></div>'
                f"{risk_scale_html(proba)}",
                unsafe_allow_html=True,
            )
            st.caption(TIER_SUMMARY[tier])

        with st.container(border=True):
            st.markdown("**Suggested next steps**")
            for action in RECOMMENDED_ACTIONS[tier]:
                st.markdown(f"- {action}")

    with right:
        with st.container(border=True):
            st.markdown("**What drives this score**")
            st.caption("Model factor contributions — associative, not causal.")
            try:
                explainer, pre = get_explainer()
                x = pre.transform(engineer_features(pd.DataFrame([profile]))[FEATURE_COLUMNS])
                x = np.asarray(x.toarray() if hasattr(x, "toarray") else x)
                sv = explainer(x)
                contrib = pd.Series(sv.values[0], index=sv.feature_names)
                values = pd.Series(x[0], index=sv.feature_names)
                top = pd.concat(
                    [contrib.sort_values().head(4), contrib.sort_values().tail(4)]
                )
                top = top[top.abs() > 0.01]

                def label(f):
                    # One-hot indicator at 0 => the customer does NOT have this
                    # level; "Contract = Two year raising risk" would read backwards.
                    if "_" in f and values[f] == 0:
                        return pretty_feature(f).replace(" = ", " ≠ ")
                    return pretty_feature(f)

                fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(top) + 0.7))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                ax.barh(
                    [label(f) for f in top.index], top.values,
                    color=[BLUE if v < 0 else ORANGE for v in top.values], height=0.6,
                )
                ax.axvline(0, color=MUTED, lw=1)
                ax.set_xlabel("Contribution to churn score (log-odds)", fontsize=9, color=INK2)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.tick_params(labelsize=9, colors=INK2)
                fig.tight_layout()
                st.pyplot(fig)
                st.caption(
                    "Orange bars push the score toward churn; blue bars pull it away."
                )
            except Exception as exc:  # noqa: BLE001 - explanation is best-effort; scoring already shown
                st.caption(f"Explanation unavailable ({exc}).")


def stat_card(column, label: str, value: str, note: str) -> None:
    with column:
        with st.container(border=True):
            st.markdown(
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div>'
                f'<div class="stat-note">{note}</div>',
                unsafe_allow_html=True,
            )


def dashboard_page() -> None:
    st.markdown("#### Project findings")
    st.caption(
        "Headline numbers from the analysis. All metrics were computed on the "
        "held-out test set (notebooks 06–07); nothing here is estimated or invented."
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    stat_card(k1, "Overall churn rate", "26.5%", "1,869 of 7,043 customers")
    stat_card(k2, "Model ROC-AUC", "0.845", "held-out test set")
    stat_card(k3, "Churners caught", "73.0%", "recall at the 0.338 threshold")
    stat_card(k4, "Top-20% capture", "49.7%", "of all churners; 2.5x random outreach")

    st.write("")
    figures = [
        ("02_churn_by_contract.png", "Churn by contract — the 15x gradient"),
        ("10_contract_x_internet.png", "Where risk concentrates"),
        ("18_risk_segments.png", "Risk tiers on held-out data"),
        ("19_gains_curve.png", "Model targeting vs random outreach"),
    ]
    cols = st.columns(2, gap="medium")
    for i, (fname, caption) in enumerate(figures):
        path = FIGURES_DIR / fname
        if path.exists():
            with cols[i % 2]:
                with st.container(border=True):
                    st.image(str(path), width="stretch")
                    st.caption(caption)

    st.write("")
    with st.container(border=True):
        st.markdown(
            """
**The churn-prone profile** (composite of observed associations): a newer month-to-month
customer on fiber internet with a high monthly bill, paying by electronic check, without
support or security add-ons.

**Top recommendations** (each phrased as a testable hypothesis — see notebook 07):

1. Early-lifecycle program for month-to-month fiber customers (54.6% observed churn).
2. Randomized auto-pay migration campaign for electronic-check payers (45.3% vs 15–17%).
3. Attach protective add-ons — not streaming — to retention offers (42% vs 15% churn).
"""
        )


st.markdown(
    '<div class="app-header"><h1>Telco Customer Churn — Risk Scoring</h1>'
    "<p>Estimate a customer's churn risk, understand the factors behind it, "
    "and get suggested next steps.</p></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Page", ["Score a customer", "Project findings"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model summary**")
st.sidebar.markdown(
    f"Logistic regression pipeline  \n"
    f"Test ROC-AUC 0.845 · PR-AUC 0.650  \n"
    f"Threshold {DECISION_THRESHOLD:.3f} (max-F1, out-of-fold)  \n"
    f"Trained on {get_bundle()['n_train']:,} customers"
)

if page == "Score a customer":
    score_page()
else:
    dashboard_page()

st.write("")
with st.expander("About this demo — and what a production version would require"):
    st.markdown(
        """
This application demonstrates the project's model; it is **not** a production system.
A real deployment would additionally require: authentication and audit logging; a model
registry with versioning and rollback; input validation against live billing-system schemas
(not hand-entered forms); scheduled retraining with **drift monitoring** (input distributions
and calibration); an A/B measurement framework so retention lift is measured against holdout
controls rather than assumed; latency/SLA guarantees behind an API rather than a
notebook-grade app; and review of fairness implications before using demographic attributes
operationally. The dataset is also a single IBM sample snapshot — patterns here demonstrate
method, not any real carrier's economics.
"""
    )
