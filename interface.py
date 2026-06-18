"""
Fraud Detection — Scenario Demo
Run with: streamlit run fraud_demo.py
Requires: streamlit, pandas, numpy, joblib, xgboost, scikit-learn, matplotlib
Place xgb_fraud_model.pkl and ohe_encoder.pkl in the same directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from xgboost import XGBClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0a0e1a; color: #e2e8f0; }

  .metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .value {
    font-size: 2rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .tx-card {
    background: #111827;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border-left: 4px solid #1e293b;
    transition: border-color 0.3s;
  }
  .tx-card.fraud  { border-left-color: #ef4444; }
  .tx-card.legit  { border-left-color: #10b981; }
  .tx-card.warn   { border-left-color: #f59e0b; }

  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .badge-fraud  { background: #450a0a; color: #fca5a5; }
  .badge-legit  { background: #052e16; color: #6ee7b7; }
  .badge-warn   { background: #451a03; color: #fcd34d; }

  .score-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .section-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
  }

  .feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.85rem;
  }
  .feature-name { color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
  .feature-val  { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 500; }

  div[data-testid="stButton"] button {
    width: 100%;
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
  }
  div[data-testid="stButton"] button:hover {
    background: #334155;
    border-color: #475569;
    color: #f1f5f9;
  }
</style>
""", unsafe_allow_html=True)


# ── Model & OHE loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = XGBClassifier()
        model.load_model('xgb_fraud_model.json')
    except FileNotFoundError:
        model = None
    try:
        ohe = joblib.load('ohe_encoder.pkl')
    except FileNotFoundError:
        ohe = None
    return model, ohe

model, ohe = load_artifacts()

# Exact feature order the model was trained on
MODEL_FEATURES = [
    'amt', 'time_of_day', 'day_of_week', 'day_of_month', 'month', 'age',
    'tx_count_1h', 'tx_count_24h', 'amt_sum_24h',
    'category_entertainment', 'category_food_dining', 'category_gas_transport',
    'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
    'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos',
    'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
    'category_travel'
]

# Decision threshold from cost-based optimization (C_FN/C_FP = 10)
THRESHOLD = 0.1149

# ── Feature store (session state) ─────────────────────────────────────────────
if "feature_store" not in st.session_state:
    st.session_state.feature_store = {}
if "tx_log" not in st.session_state:
    st.session_state.tx_log = []


# ── Feature engineering ───────────────────────────────────────────────────────
def build_feature_vector(cc_num, timestamp, amt, category, age=35):
    """
    Replicates the preprocessing pipeline for a single live transaction.
    Velocity features are computed from the in-memory feature store,
    mirroring the rolling 1h/24h windows used during training.
    age defaults to 35 since dob is not available in the demo context.
    """
    history = st.session_state.feature_store.get(cc_num, [])
    cutoff_24h = timestamp - timedelta(hours=24)
    cutoff_1h  = timestamp - timedelta(hours=1)

    # filter to 24h window (matches training rolling window)
    history = [(t, a) for t, a in history if t >= cutoff_24h]

    tx_count_24h = len(history)
    tx_count_1h  = sum(1 for t, _ in history if t >= cutoff_1h)
    amt_sum_24h  = sum(a for _, a in history)

    # amt_zscore is kept as a display metric only — not fed to the model
    amounts    = [a for _, a in history]
    mean_amt   = np.mean(amounts) if amounts else amt
    std_amt    = np.std(amounts)  if len(amounts) > 1 else 1.0
    amt_zscore = (amt - mean_amt) / (std_amt + 1e-8)

    # append current transaction to history for future calls
    history.append((timestamp, amt))
    st.session_state.feature_store[cc_num] = history

    # OHE encoding via saved encoder — exact same transform as training
    if ohe is not None:
        cat_df     = pd.DataFrame([[category]], columns=['category'])
        cat_encoded = ohe.transform(cat_df)
        cat_cols   = ohe.get_feature_names_out(['category'])
        ohe_dict   = dict(zip(cat_cols, cat_encoded[0].astype(int)))
    else:
        # fallback if encoder not found: manual one-hot
        all_cats = [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net',
            'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net',
            'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel'
        ]
        ohe_dict = {f"category_{c}": int(c == category) for c in all_cats}

    features = {
        'amt':          amt,
        'time_of_day':  timestamp.hour,
        'day_of_week':  timestamp.weekday(),
        'day_of_month': timestamp.day,
        'month':        timestamp.month,
        'age':          age,
        'tx_count_1h':  tx_count_1h,
        'tx_count_24h': tx_count_24h,
        'amt_sum_24h':  amt_sum_24h,
        **ohe_dict,
    }

    display = {
        'tx_count_1h':  tx_count_1h,
        'tx_count_24h': tx_count_24h,
        'amt_sum_24h':  f"${amt_sum_24h:,.2f}",
        'amt_zscore':   round(amt_zscore, 2),  # display only
    }

    return features, display


def predict(features):
    """
    Build a DataFrame in the exact column order the model was trained on,
    then apply the cost-based threshold.
    """
    if model is None:
        # deterministic mock when model file is absent
        score = min(1.0, (
            features['amt'] / 2000 * 0.4 +
            features['tx_count_24h'] / 10 * 0.35 +
            features['amt_sum_24h'] / 5000 * 0.25
        ) + np.random.normal(0, 0.03))
        return float(np.clip(score, 0, 1))

    X = pd.DataFrame([features])

    # ensure all expected columns are present and in the right order
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[MODEL_FEATURES]

    return float(model.predict_proba(X)[0][1])


def score_to_state(score):
    if score >= THRESHOLD:
        return "fraud"
    elif score >= THRESHOLD * 0.6:
        return "warn"
    return "legit"


def state_label(state):
    return {"fraud": "🚨 Flagged", "warn": "⚠️ Review", "legit": "✅ Approved"}[state]


def score_color(score):
    if score >= THRESHOLD:      return "#ef4444"
    elif score >= THRESHOLD * 0.6: return "#f59e0b"
    return "#10b981"


# ── Scenarios ─────────────────────────────────────────────────────────────────
BASE_TIME = datetime(2024, 3, 15, 14, 0, 0)

SCENARIOS = {
    "velocity_burst": {
        "title": "High-frequency burst",
        "subtitle": "10 transactions in under an hour",
        "icon": "⚡",
        "why_fraud": "Rapid successive transactions on a single card are a primary fraud signal — stolen cards are often tested with small amounts then drained quickly.",
        "transactions": [
            {"cc": "CARD_001", "offset_min": 0,  "amt": 45.00,   "cat": "misc_pos",     "label": "Small test charge"},
            {"cc": "CARD_001", "offset_min": 3,  "amt": 38.50,   "cat": "misc_pos",     "label": "Second test"},
            {"cc": "CARD_001", "offset_min": 7,  "amt": 120.00,  "cat": "shopping_net", "label": "First escalation"},
            {"cc": "CARD_001", "offset_min": 12, "amt": 340.00,  "cat": "shopping_net", "label": "Larger purchase"},
            {"cc": "CARD_001", "offset_min": 18, "amt": 780.00,  "cat": "misc_net",     "label": "High value"},
            {"cc": "CARD_001", "offset_min": 25, "amt": 999.00,  "cat": "shopping_net", "label": "Near limit"},
            {"cc": "CARD_001", "offset_min": 31, "amt": 450.00,  "cat": "misc_net",     "label": "Continued drain"},
            {"cc": "CARD_001", "offset_min": 40, "amt": 890.00,  "cat": "shopping_net", "label": "Large charge"},
            {"cc": "CARD_001", "offset_min": 48, "amt": 1100.00, "cat": "misc_net",     "label": "Exceeds history"},
            {"cc": "CARD_001", "offset_min": 55, "amt": 1450.00, "cat": "shopping_net", "label": "Final drain"},
        ]
    },
    "large_single": {
        "title": "Single large transaction",
        "subtitle": "Unusual amount with no prior history",
        "icon": "💸",
        "why_fraud": "A transaction far above a card's typical spend pattern has a high z-score. Combined with no 24h history, the model has little context to trust it.",
        "transactions": [
            {"cc": "CARD_002", "offset_min": 0, "amt": 4200.00, "cat": "shopping_net", "label": "Anomalous charge"},
        ]
    },
    "legitimate_shopping": {
        "title": "Legitimate shopping spree",
        "subtitle": "High spend but spread across hours",
        "icon": "🛍️",
        "why_fraud": "Same total spend as a fraud scenario but distributed naturally across a day — velocity stays low, amounts escalate gradually, matching real shopping behaviour.",
        "transactions": [
            {"cc": "CARD_003", "offset_min": 0,   "amt": 55.00,  "cat": "food_dining",   "label": "Lunch"},
            {"cc": "CARD_003", "offset_min": 90,  "amt": 320.00, "cat": "shopping_pos",  "label": "Clothing store"},
            {"cc": "CARD_003", "offset_min": 210, "amt": 180.00, "cat": "shopping_pos",  "label": "Electronics"},
            {"cc": "CARD_003", "offset_min": 380, "amt": 95.00,  "cat": "personal_care", "label": "Pharmacy"},
            {"cc": "CARD_003", "offset_min": 540, "amt": 430.00, "cat": "shopping_net",  "label": "Online order"},
        ]
    },
    "bust_out": {
        "title": "Bust-out pattern",
        "subtitle": "Normal use then sudden account drain",
        "icon": "📉",
        "why_fraud": "Several days of legitimate low-value transactions establish a normal baseline. Then suddenly high-value purchases cause the z-score to spike — the model sees the deviation.",
        "transactions": [
            {"cc": "CARD_004", "offset_min": -2880, "amt": 42.00,   "cat": "grocery_pos",  "label": "Day -2: Groceries"},
            {"cc": "CARD_004", "offset_min": -2700, "amt": 18.50,   "cat": "food_dining",  "label": "Day -2: Coffee"},
            {"cc": "CARD_004", "offset_min": -1440, "amt": 65.00,   "cat": "grocery_pos",  "label": "Day -1: Groceries"},
            {"cc": "CARD_004", "offset_min": -1320, "amt": 12.00,   "cat": "food_dining",  "label": "Day -1: Snack"},
            {"cc": "CARD_004", "offset_min": 0,     "amt": 890.00,  "cat": "shopping_net", "label": "Day 0: Sudden spike"},
            {"cc": "CARD_004", "offset_min": 15,    "amt": 1200.00, "cat": "misc_net",     "label": "Day 0: Continued"},
            {"cc": "CARD_004", "offset_min": 30,    "amt": 1500.00, "cat": "shopping_net", "label": "Day 0: Account drain"},
        ]
    },
    "night_fraud": {
        "title": "Late night transactions",
        "subtitle": "High-value purchases at 2–4 AM",
        "icon": "🌙",
        "why_fraud": "Fraudsters operate at night when cardholders are asleep and less likely to notice alerts. The model learns this temporal signal from the training data distribution.",
        "transactions": [
            {"cc": "CARD_005", "offset_min": 0,  "amt": 340.00, "cat": "shopping_net", "label": "2:00 AM purchase",   "hour": 2},
            {"cc": "CARD_005", "offset_min": 12, "amt": 560.00, "cat": "misc_net",     "label": "2:12 AM escalation", "hour": 2},
            {"cc": "CARD_005", "offset_min": 25, "amt": 890.00, "cat": "shopping_net", "label": "2:25 AM large buy",  "hour": 2},
        ]
    },
}


# ── Feature contribution chart ─────────────────────────────────────────────────
def contribution_bar_chart(features, score):
    """
    Approximate feature contributions using known feature importances.
    Uses only features actually in the model — amt_zscore excluded.
    For real SHAP values, pass model to shap.TreeExplainer instead.
    """
    raw = {
        'amt':          features['amt'] / 500 * 0.30,
        'tx_count_24h': features['tx_count_24h'] / 5 * 0.35,
        'amt_sum_24h':  features['amt_sum_24h'] / 2000 * 0.25,
        'tx_count_1h':  features['tx_count_1h'] / 3 * 0.10,
        'time_of_day':  0.05 if features['time_of_day'] < 6 else -0.02,
    }
    total = sum(abs(v) for v in raw.values())
    contribs = {k: v / total * score for k, v in raw.items()} if total > 0 else raw

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    keys   = list(contribs.keys())
    values = list(contribs.values())
    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

    ax.barh(keys, values, color=colors, height=0.5)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Contribution to fraud score", color="#64748b", fontsize=8)
    plt.tight_layout()
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px;">
  <div style="font-size:0.72rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">
    PFA · Fraud Detection System
  </div>
  <h1 style="font-size:1.8rem;font-weight:600;color:#f1f5f9;margin:0 0 6px;">
    Real-time Transaction Scoring
  </h1>
  <p style="color:#64748b;font-size:0.9rem;margin:0;">
    Scenario-based demonstration of XGBoost + velocity feature fraud detection
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.info("⚠️  `xgb_fraud_model.pkl` not found — running with mock scoring.", icon="ℹ️")
if ohe is None:
    st.info("⚠️  `ohe_encoder.pkl` not found — using fallback one-hot encoding.", icon="ℹ️")

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="section-title">Scenarios</div>', unsafe_allow_html=True)

    for key, s in SCENARIOS.items():
        if st.button(f"{s['icon']}  {s['title']}\n{s['subtitle']}", key=f"btn_{key}"):
            st.session_state.feature_store = {}
            st.session_state.tx_log = []
            st.session_state.active_scenario = key

    st.divider()
    if st.button("🔄  Reset all", key="reset"):
        st.session_state.feature_store = {}
        st.session_state.tx_log = []
        st.session_state.pop("active_scenario", None)
        st.rerun()

    if st.session_state.tx_log:
        st.markdown('<div class="section-title" style="margin-top:24px;">Session summary</div>', unsafe_allow_html=True)
        total   = len(st.session_state.tx_log)
        flagged = sum(1 for t in st.session_state.tx_log if t["state"] == "fraud")
        review  = sum(1 for t in st.session_state.tx_log if t["state"] == "warn")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total",      total)
        c2.metric("🚨 Flagged", flagged)
        c3.metric("⚠️ Review",  review)


with right:
    if "active_scenario" not in st.session_state:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #1e293b;border-radius:12px;
                    padding:60px 40px;text-align:center;color:#475569;">
          <div style="font-size:2rem;margin-bottom:12px;">👈</div>
          <div style="font-size:0.9rem;">Select a scenario to begin</div>
          <div style="font-size:0.78rem;margin-top:6px;color:#334155;">
            Each scenario runs a sequence of transactions through the live feature store
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        scenario = SCENARIOS[st.session_state.active_scenario]

        st.markdown(f"""
        <div style="background:#111827;border-radius:12px;padding:20px 24px;margin-bottom:20px;
                    border:1px solid #1e293b;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:1.5rem;">{scenario['icon']}</span>
            <div>
              <div style="font-weight:600;color:#f1f5f9;">{scenario['title']}</div>
              <div style="font-size:0.8rem;color:#64748b;">{scenario['subtitle']}</div>
            </div>
          </div>
          <div style="font-size:0.82rem;color:#94a3b8;border-top:1px solid #1e293b;
                      padding-top:12px;margin-top:4px;">
            <span style="color:#475569;font-size:0.72rem;text-transform:uppercase;
                         letter-spacing:0.08em;">Why the model flags this · </span>
            {scenario['why_fraud']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # process transactions once per scenario load
        if not st.session_state.tx_log:
            for tx in scenario["transactions"]:
                ts = BASE_TIME + timedelta(minutes=tx["offset_min"])
                if "hour" in tx:
                    ts = ts.replace(hour=tx["hour"])

                features, display = build_feature_vector(
                    tx["cc"], ts, tx["amt"], tx["cat"]
                )
                score = predict(features)
                state = score_to_state(score)

                st.session_state.tx_log.append({
                    "label":    tx["label"],
                    "amt":      tx["amt"],
                    "cat":      tx["cat"],
                    "time":     ts.strftime("%H:%M"),
                    "score":    score,
                    "state":    state,
                    "display":  display,
                    "features": features,
                })

        st.markdown('<div class="section-title">Transaction feed</div>', unsafe_allow_html=True)

        for tx in st.session_state.tx_log:
            score     = tx["score"]
            state     = tx["state"]
            bar_color = score_color(score)
            bar_pct   = int(score * 100)

            st.markdown(f"""
            <div class="tx-card {state}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div style="font-weight:500;color:#f1f5f9;font-size:0.9rem;">{tx['label']}</div>
                  <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">
                    {tx['time']} · {tx['cat'].replace('_',' ').title()}
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                              color:#f1f5f9;font-weight:500;">${tx['amt']:,.2f}</div>
                  <span class="badge badge-{state}">{state_label(state)}</span>
                </div>
              </div>
              <div style="margin-top:12px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.72rem;color:#475569;margin-bottom:4px;">
                  <span>Fraud score</span>
                  <span style="font-family:'JetBrains Mono',monospace;color:{bar_color};">
                    {score:.3f}
                  </span>
                </div>
                <div class="score-bar-bg">
                  <div class="score-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
                </div>
              </div>
              <div style="margin-top:12px;display:flex;gap:20px;flex-wrap:wrap;">
                <span style="font-size:0.72rem;color:#475569;">
                  <span style="color:#64748b;">tx/1h </span>
                  <span style="font-family:'JetBrains Mono',monospace;color:#94a3b8;">
                    {tx['display']['tx_count_1h']}
                  </span>
                </span>
                <span style="font-size:0.72rem;color:#475569;">
                  <span style="color:#64748b;">tx/24h </span>
                  <span style="font-family:'JetBrains Mono',monospace;color:#94a3b8;">
                    {tx['display']['tx_count_24h']}
                  </span>
                </span>
                <span style="font-size:0.72rem;color:#475569;">
                  <span style="color:#64748b;">24h spend </span>
                  <span style="font-family:'JetBrains Mono',monospace;color:#94a3b8;">
                    {tx['display']['amt_sum_24h']}
                  </span>
                </span>
                <span style="font-size:0.72rem;color:#475569;">
                  <span style="color:#64748b;">z-score </span>
                  <span style="font-family:'JetBrains Mono',monospace;
                        color:{'#ef4444' if tx['display']['amt_zscore'] > 2 else '#94a3b8'};">
                    {tx['display']['amt_zscore']}
                  </span>
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.tx_log:
            last = st.session_state.tx_log[-1]
            st.markdown(
                '<div class="section-title" style="margin-top:24px;">'
                'Feature contributions — last transaction</div>',
                unsafe_allow_html=True
            )
            fig = contribution_bar_chart(last["features"], last["score"])
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("""
            <div style="font-size:0.75rem;color:#475569;margin-top:8px;">
              <span style="color:#ef4444;">■</span> pushes toward fraud &nbsp;·&nbsp;
              <span style="color:#10b981;">■</span> pushes toward legitimate
            </div>
            """, unsafe_allow_html=True)