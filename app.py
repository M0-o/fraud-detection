"""
app.py  –  Interface Streamlit pour FraudShield
À placer dans le dossier fraud-detection-main/ (là où se trouvent
X_train.csv, y_train.csv, X_test.csv, y_test.csv générés par preprocessing.py)

Workflow :
  1. Lancer preprocessing.py  →  génère les CSV
  2. Lancer app.py             →  entraîne, évalue, prédit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve,
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');
:root {
    --bg:#f0f4ff; --surface:#ffffff; --border:#dde3f5;
    --accent:#4361ee; --accent-light:#eef0fd; --accent3:#7209b7;
    --text:#1a1f36; --text2:#4a5378; --muted:#8b93b8;
    --success:#0caf60; --success-light:#e8faf2;
    --danger:#e63946; --danger-light:#fef0f1;
    --shadow:0 2px 12px rgba(67,97,238,.08);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background:var(--bg)!important;color:var(--text)!important}
.stApp{background:var(--bg)!important}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1.5px solid var(--border)!important}
h1,h2,h3{font-family:'DM Sans',sans-serif!important;font-weight:700!important}
[data-testid="metric-container"]{background:var(--surface)!important;border:1.5px solid var(--border)!important;border-radius:16px!important;padding:1.1rem 1.3rem!important;box-shadow:var(--shadow)!important}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'DM Mono',monospace!important;font-size:1.55rem!important}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent3))!important;color:#fff!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:.6rem 1.5rem!important;box-shadow:0 4px 15px rgba(67,97,238,.28)!important}
.stButton>button:hover{transform:translateY(-2px)!important}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:12px!important;padding:5px!important;border:1.5px solid var(--border)!important}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:#fff!important;border-radius:9px!important}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-weight:600!important}
.fraud-alert{background:var(--danger-light);border:1.5px solid #f5b8bc;border-left:5px solid var(--danger);border-radius:12px;padding:1rem 1.4rem;margin:.5rem 0}
.legit-alert{background:var(--success-light);border:1.5px solid #9ae8c3;border-left:5px solid var(--success);border-radius:12px;padding:1rem 1.4rem;margin:.5rem 0}
.info-card{background:var(--surface);border:1.5px solid var(--border);border-radius:16px;padding:1.4rem;margin:.5rem 0;box-shadow:var(--shadow)}
.mono{font-family:'DM Mono',monospace;font-size:.83rem;color:var(--accent)}
.hero-title{font-size:1.55rem;font-weight:800;color:var(--accent)}
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    'entertainment','food_dining','gas_transport','grocery_net',
    'grocery_pos','health_fitness','home','kids_pets','misc_net',
    'misc_pos','personal_care','shopping_net','shopping_pos','travel'
]

MODEL_OPTIONS = {
    "XGBoost":        lambda: XGBClassifier(eval_metric='aucpr', n_jobs=-1, random_state=42, verbosity=0),
    "Random Forest":  lambda: RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42),
    "LightGBM":       lambda: LGBMClassifier(is_unbalance=True, n_jobs=-1, random_state=42, verbose=-1),
}

MODEL_FILE  = "model.pkl"
META_FILE   = "model_meta.pkl"   # stocke ohe, cat_cols, feature_names, threshold, metrics

# ── Helpers ────────────────────────────────────────────────────────────────────
def plot_cfg():
    return dict(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a5378', family='DM Sans'),
        xaxis=dict(gridcolor='#dde3f5', linecolor='#dde3f5'),
        yaxis=dict(gridcolor='#dde3f5', linecolor='#dde3f5'),
    )

def load_csv_data():
    """Charge les 4 CSV générés par preprocessing.py"""
    missing = [f for f in ["X_train.csv","y_train.csv","X_test.csv","y_test.csv"] if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    X_test  = pd.read_csv("X_test.csv")
    y_test  = pd.read_csv("y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test, []

def train_model(model_name, use_smote, X_train, y_train, X_test, y_test):
    model = MODEL_OPTIONS[model_name]()
    if use_smote:
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_train, y_train)
    else:
        X_tr, y_tr = X_train, y_train

    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1.argmax()
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (y_proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "auc_pr":    average_precision_score(y_test, y_proba),
        "report":    report,
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "precision": precision.tolist(),
        "recall":    recall.tolist(),
        "cm":        cm.tolist(),
        "threshold": threshold,
        "y_test":    y_test.tolist(),
        "y_proba":   y_proba.tolist(),
    }
    return model, metrics

def load_saved_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(META_FILE):
        model = joblib.load(MODEL_FILE)
        meta  = joblib.load(META_FILE)
        return model, meta
    return None, None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🛡️ FraudDetector</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b93b8;font-size:.85rem;margin-bottom:1.2rem;">Détection de fraude bancaire par ML</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("###  Algorithme")
    selected_model = st.selectbox("Modèle", list(MODEL_OPTIONS.keys()), index=0, label_visibility="collapsed")
    use_smote = st.toggle("Utiliser SMOTE (rééchantillonnage)", value=True)

    st.divider()
    st.markdown("###  Entraînement")

    # Vérifier si les CSV existent
    X_train, y_train, X_test, y_test, missing = load_csv_data()
    if missing:
        st.warning(f"CSV manquants : `{'`, `'.join(missing)}`\n\nLancez d'abord `preprocessing.py`.")
        csv_ok = False
    else:
        n_train, n_cols = X_train.shape
        st.success(f" CSV chargés — {n_train:,} lignes · {n_cols} features")
        csv_ok = True

    train_btn = st.button("🏋️ Entraîner le modèle", use_container_width=True, disabled=not csv_ok)

    # Charger un modèle sauvegardé existant
    st.divider()
    st.markdown("### 💾 Charger un modèle existant")
    if os.path.exists(MODEL_FILE):
        if st.button("📂 Charger model.pkl", use_container_width=True):
            model, meta = load_saved_model()
            if model:
                st.session_state.model        = model
                st.session_state.meta         = meta
                st.session_state.model_trained = True
                st.session_state.model_name   = meta.get("model_name", "Inconnu")
                st.session_state.metrics      = meta["metrics"]
                st.session_state.feature_names = meta["feature_names"]
                st.session_state.threshold    = meta["threshold"]
                st.rerun()
    else:
        st.caption("Aucun `model.pkl` trouvé dans le dossier.")

    if "model_trained" in st.session_state and st.session_state.model_trained:
        m = st.session_state.metrics
        st.divider()
        st.success(f" {st.session_state.model_name}")
        st.markdown(f'<div class="mono">AUC-ROC : {m["auc_roc"]:.4f}<br>AUC-PR  : {m["auc_pr"]:.4f}<br>Seuil   : {m["threshold"]:.4f}</div>', unsafe_allow_html=True)

# ── Entraînement ───────────────────────────────────────────────────────────────
if train_btn and csv_ok:
    with st.spinner(f"Entraînement {selected_model}{'  + SMOTE' if use_smote else ''}…"):
        model, metrics = train_model(selected_model, use_smote, X_train, y_train, X_test, y_test)

        # Sauvegarder sur disque
        meta = {
            "model_name":    selected_model,
            "feature_names": X_train.columns.tolist(),
            "threshold":     metrics["threshold"],
            "metrics":       metrics,
        }
        joblib.dump(model, MODEL_FILE)
        joblib.dump(meta,  META_FILE)

        st.session_state.model         = model
        st.session_state.meta          = meta
        st.session_state.model_trained = True
        st.session_state.model_name    = selected_model
        st.session_state.metrics       = metrics
        st.session_state.feature_names = X_train.columns.tolist()
        st.session_state.threshold     = metrics["threshold"]
    st.rerun()

# ── Onglets ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Tableau de bord", " Prédiction manuelle", " Prédiction par fichier"])

# ═══════════════════════════════ TAB 1 — DASHBOARD ════════════════════════════
with tab1:
    if "model_trained" not in st.session_state or not st.session_state.model_trained:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🛡️</div>
            <div class="hero-title" style="font-size:2rem;">Prêt à analyser vos transactions</div>
            <p style="color:#8b93b8;margin-top:.5rem;">
                Lancez d'abord <code>preprocessing.py</code> pour générer les CSV,<br>
                puis cliquez sur <strong>Entraîner le modèle</strong> dans la barre latérale.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        r = m["report"]

        st.markdown(f"## 📊 Performance — {st.session_state.model_name}")
        st.markdown(f'<p class="mono">Seuil optimal (max F1) : {m["threshold"]:.4f}</p>', unsafe_allow_html=True)
        st.divider()

        # KPIs
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("AUC-ROC",           f'{m["auc_roc"]:.4f}')
        c2.metric("AUC-PR",            f'{m["auc_pr"]:.4f}')
        c3.metric("Précision (fraude)", f'{r["1"]["precision"]:.2%}')
        c4.metric("Rappel (fraude)",    f'{r["1"]["recall"]:.2%}')
        c5.metric("F1 (fraude)",        f'{r["1"]["f1-score"]:.2%}')

        st.divider()
        col_l, col_r = st.columns(2)

        # Courbe ROC
        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode='lines',
                line=dict(color='#4361ee', width=2.5), name=f'AUC={m["auc_roc"]:.4f}',
                fill='tozeroy', fillcolor='rgba(67,97,238,.08)'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                line=dict(color='#dde3f5', dash='dash'), showlegend=False))
            fig.update_layout(title="Courbe ROC", xaxis_title="Taux Faux Positifs",
                yaxis_title="Taux Vrais Positifs", height=350, **plot_cfg())
            st.plotly_chart(fig, use_container_width=True)

        # Courbe PR
        with col_r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["recall"], y=m["precision"], mode='lines',
                line=dict(color='#7209b7', width=2.5), name=f'AUC-PR={m["auc_pr"]:.4f}',
                fill='tozeroy', fillcolor='rgba(114,9,183,.08)'))
            fig.update_layout(title="Courbe Précision-Rappel", xaxis_title="Rappel",
                yaxis_title="Précision", height=350, **plot_cfg())
            st.plotly_chart(fig, use_container_width=True)

        col_cm, col_fi = st.columns(2)

        # Matrice de confusion
        with col_cm:
            cm_arr = np.array(m["cm"])
            fig = go.Figure(go.Heatmap(
                z=cm_arr, x=["Légitime","Fraude"], y=["Légitime","Fraude"],
                colorscale=[[0,'#eef0fd'],[1,'#4361ee']],
                text=cm_arr, texttemplate="%{text}", textfont=dict(size=18),
                showscale=False))
            fig.update_layout(title="Matrice de Confusion", height=350,
                xaxis_title="Prédit", yaxis_title="Réel", **plot_cfg())
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        with col_fi:
            mdl = st.session_state.model
            feats = st.session_state.feature_names
            if hasattr(mdl, 'feature_importances_'):
                fi_df = pd.DataFrame({'feature': feats, 'importance': mdl.feature_importances_})
                fi_df = fi_df.sort_values('importance', ascending=True).tail(12)
                fig = go.Figure(go.Bar(
                    x=fi_df['importance'], y=fi_df['feature'], orientation='h',
                    marker=dict(color=fi_df['importance'],
                        colorscale=[[0,'#eef0fd'],[1,'#4361ee']])))
                fig.update_layout(title="Importance des Variables", height=350, **plot_cfg())
                st.plotly_chart(fig, use_container_width=True)

        # Distribution des scores
        st.divider()
        yt = np.array(m["y_test"]); yp = np.array(m["y_proba"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=yp[yt==0], name='Légitime', opacity=.7,
            marker_color='#0caf60', nbinsx=80, histnorm='probability density'))
        fig.add_trace(go.Histogram(x=yp[yt==1], name='Fraude', opacity=.7,
            marker_color='#e63946', nbinsx=80, histnorm='probability density'))
        fig.add_vline(x=m["threshold"], line_dash="dash", line_color="#f59e0b",
            annotation_text=f'Seuil: {m["threshold"]:.3f}')
        fig.update_layout(title="Distribution des Scores de Fraude", barmode='overlay',
            xaxis_title="Score de probabilité", yaxis_title="Densité",
            height=300, **plot_cfg())
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════ TAB 2 — PRÉDICTION MANUELLE ══════════════════
with tab2:
    st.markdown("## 🔍 Analyser une Transaction")
    if "model_trained" not in st.session_state or not st.session_state.model_trained:
        st.info("⚠️ Entraînez d'abord le modèle dans la barre latérale.")
    else:
        col_form, col_res = st.columns([1,1], gap="large")

        with col_form:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 💳 Détails de la transaction")
            amt       = st.number_input("Montant ($)", min_value=0.01, max_value=100000.0, value=150.0, step=0.01)
            category  = st.selectbox("Catégorie marchande", CATEGORIES)
            trans_date = st.date_input("Date", value=datetime.today())
            trans_time = st.time_input("Heure", value=datetime.now().time())
            age       = st.number_input("Âge du titulaire", min_value=18, max_value=100, value=35)
            predict_btn = st.button("⚡ Analyser", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            if predict_btn:
                mdl           = st.session_state.model
                feature_names = st.session_state.feature_names
                threshold     = st.session_state.threshold

                # Reconstruire le vecteur de features exactement comme preprocessing.py
                # OHE sur category manuellement (colonnes connues via feature_names)
                row = {'amt': amt, 'time_of_day': trans_time.hour,
                       'day_of_week': trans_date.weekday(), 'month': trans_date.month, 'age': age}

                # Ajouter colonnes OHE (category_xxx)
                for f in feature_names:
                    if f.startswith('category_'):
                        cat_val = f.replace('category_', '')
                        row[f] = 1 if category == cat_val else 0

                df_row = pd.DataFrame([row])
                for f in feature_names:
                    if f not in df_row.columns:
                        df_row[f] = 0
                df_row = df_row[feature_names]

                proba    = mdl.predict_proba(df_row)[0][1]
                is_fraud = proba >= threshold
                risk_pct = proba * 100

                # Jauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk_pct,
                    number={'suffix':'%','font':{'size':38,'family':'DM Mono','color':'#4361ee'}},
                    gauge={
                        'axis':{'range':[0,100],'tickcolor':'#8b93b8'},
                        'bar':{'color':'#e63946' if is_fraud else '#0caf60','thickness':.25},
                        'bgcolor':'#f0f4ff','bordercolor':'#dde3f5',
                        'steps':[
                            {'range':[0,30],'color':'rgba(12,175,96,.12)'},
                            {'range':[30,60],'color':'rgba(245,158,11,.12)'},
                            {'range':[60,100],'color':'rgba(230,57,70,.12)'},
                        ],
                        'threshold':{'line':{'color':'#f59e0b','width':2},'thickness':.75,'value':threshold*100}
                    },
                    title={'text':"Score de Risque",'font':{'size':14,'color':'#8b93b8'}}
                ))
                fig.update_layout(height=280, **plot_cfg())
                st.plotly_chart(fig, use_container_width=True)

                if is_fraud:
                    st.markdown(f'''<div class="fraud-alert">
                        <strong style="color:#e63946;font-size:1.1rem;"> TRANSACTION SUSPECTE</strong><br>
                        <span style="color:#4a5378;">Probabilité de fraude : <strong style="color:#e63946;">{risk_pct:.1f}%</strong></span><br>
                        <span style="color:#8b93b8;font-size:.8rem;">Seuil de décision : {threshold:.4f}</span>
                    </div>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''<div class="legit-alert">
                        <strong style="color:#0caf60;font-size:1.1rem;"> TRANSACTION LÉGITIME</strong><br>
                        <span style="color:#4a5378;">Probabilité de fraude : <strong style="color:#0caf60;">{risk_pct:.1f}%</strong></span><br>
                        <span style="color:#8b93b8;font-size:.8rem;">Seuil de décision : {threshold:.4f}</span>
                    </div>''', unsafe_allow_html=True)
            else:
                st.markdown('''<div style="display:flex;align-items:center;justify-content:center;height:300px;">
                    <div style="text-align:center;color:#8b93b8;">
                        <div style="font-size:4rem;">⚡</div>
                        <p style="margin-top:1rem;">Remplissez le formulaire et cliquez sur Analyser</p>
                    </div></div>''', unsafe_allow_html=True)

# ═══════════════════════════════ TAB 3 — PRÉDICTION FICHIER ═══════════════════
with tab3:
    st.markdown("## 📁 Prédiction par Fichier CSV")
    if "model_trained" not in st.session_state or not st.session_state.model_trained:
        st.info("⚠️ Entraînez d'abord le modèle dans la barre latérale.")
    else:
        st.markdown("""
        <div class="info-card">
        <strong>Format attendu</strong> : même format que <code>fraudTrain.csv</code> 
        (avec les colonnes brutes : <code>trans_date_trans_time</code>, <code>amt</code>, <code>category</code>, <code>dob</code>…).
        La colonne <code>is_fraud</code> est optionnelle.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Uploader un fichier de transactions", type=["csv"], key="predict_file")

        if uploaded:
            df_raw = pd.read_csv(uploaded, index_col=0)
            st.markdown(f'<p class="mono">{len(df_raw):,} transactions chargées</p>', unsafe_allow_html=True)

            if st.button("🔎 Lancer les prédictions", use_container_width=False):
                with st.spinner("Analyse en cours…"):
                    mdl           = st.session_state.model
                    feature_names = st.session_state.feature_names
                    threshold     = st.session_state.threshold

                    # Prétraitement identique à preprocessing.py
                    df = df_raw.copy()
                    if 'trans_date_trans_time' in df.columns:
                        dt = pd.to_datetime(df['trans_date_trans_time'])
                        df['time_of_day'] = dt.dt.hour
                        df['day_of_week']  = dt.dt.dayofweek
                        df['month']        = dt.dt.month
                    if 'dob' in df.columns:
                        df['age'] = (pd.Timestamp.today() - pd.to_datetime(df['dob'])).dt.days // 365

                    if 'category' in df.columns:
                        for cat in CATEGORIES:
                            df[f'category_{cat}'] = (df['category'] == cat).astype(int)

                    DROP = ['trans_date_trans_time','cc_num','merchant','first','last',
                            'street','city','state','zip','lat','long','merch_lat','merch_long',
                            'city_pop','job','dob','trans_num','unix_time','gender','category','is_fraud']
                    df = df.drop(columns=[c for c in DROP if c in df.columns])

                    for f in feature_names:
                        if f not in df.columns:
                            df[f] = 0
                    X_pred = df[feature_names]

                    probas = mdl.predict_proba(X_pred)[:, 1]
                    preds  = (probas >= threshold).astype(int)

                result_df = df_raw.copy()
                result_df['score_fraude'] = np.round(probas, 4)
                result_df['prediction']   = preds
                result_df['statut']       = result_df['prediction'].map({0:' Légitime', 1:' Fraude'})

                n_fraud = preds.sum()
                n_legit = len(preds) - n_fraud

                c1, c2, c3 = st.columns(3)
                c1.metric("Total",            f"{len(preds):,}")
                c2.metric(" Fraudes",        f"{n_fraud:,}", f"{n_fraud/len(preds):.2%}")
                c3.metric(" Légitimes",       f"{n_legit:,}")

                st.divider()
                display_cols = ['amt','statut','score_fraude']
                if 'category' in df_raw.columns:
                    display_cols = ['amt','category','statut','score_fraude']
                st.dataframe(result_df[display_cols].head(500), use_container_width=True, height=350)

                st.download_button(
                    " Télécharger les résultats",
                    data=result_df.to_csv(index=True),
                    file_name="predictions_fraude.csv",
                    mime="text/csv",
                )
