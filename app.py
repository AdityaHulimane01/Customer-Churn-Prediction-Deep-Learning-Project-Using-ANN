import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import sys

# ─── NumPy 1.x / 2.x Compatibility Shim ─────────────────────────────────────────
try:
    import numpy._core.numeric
except ModuleNotFoundError:
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.numeric'] = _core.numeric
    except Exception:
        pass

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ---- Header banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #3B3A8F 60%, #0F0F1A 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem 1.8rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: #c5c3f5;
    font-size: 1.05rem;
    margin: 0;
}

/* ---- Section card ---- */
.section-card {
    background: #1A1A2E;
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ---- Predict button ---- */
div.stButton > button {
    background: linear-gradient(90deg, #6C63FF, #3B3A8F);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.75rem 2.5rem;
    width: 100%;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 18px rgba(108,99,255,0.4);
    letter-spacing: 0.5px;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(108,99,255,0.55);
}
div.stButton > button:active {
    transform: translateY(0px);
}

/* ---- Result cards ---- */
.result-card-safe {
    background: linear-gradient(135deg, #0d3321 0%, #1a4d33 100%);
    border: 1.5px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(46,204,113,0.2);
    animation: popIn 0.4s ease;
}
.result-card-churn {
    background: linear-gradient(135deg, #3d0d0d 0%, #5a1a1a 100%);
    border: 1.5px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(231,76,60,0.2);
    animation: popIn 0.4s ease;
}
.result-card-medium {
    background: linear-gradient(135deg, #3d2d00 0%, #5a4400 100%);
    border: 1.5px solid #f39c12;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(243,156,18,0.2);
    animation: popIn 0.4s ease;
}
.result-label {
    font-size: 2rem;
    font-weight: 900;
    margin: 0.5rem 0;
}
.result-sub {
    font-size: 1rem;
    opacity: 0.85;
    margin-top: 0.3rem;
}
.prob-value {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
}

/* ---- Sidebar ---- */
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #6C63FF;
    margin-bottom: 0.5rem;
}
.sidebar-info {
    background: #1A1A2E;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.88rem;
    color: #c5c3f5;
    line-height: 1.6;
    border: 1px solid #2a2a4a;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #2a2a4a;
    font-size: 0.85rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #9e9ebe; }
.stat-val { color: #e0e0e0; font-weight: 600; }

/* ---- Pop-in animation ---- */
@keyframes popIn {
    0%   { transform: scale(0.92); opacity: 0; }
    100% { transform: scale(1);    opacity: 1; }
}

/* ---- Divider ---- */
.custom-divider {
    border: none;
    border-top: 1px solid #2a2a4a;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model('churn_model.keras')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('one_hot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler


model, label_encoder_gender, one_hot_encoder_geo, scaler = load_artifacts()


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏦 About This App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        This app uses a trained <b>Artificial Neural Network (ANN)</b> to predict
        whether a bank customer is likely to churn (leave the bank).<br><br>
        Fill in the customer details on the main page and click
        <b>Predict Churn</b> to get an instant prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🧠 Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        <div class="stat-row"><span class="stat-key">Architecture</span><span class="stat-val">ANN (Deep Learning)</span></div>
        <div class="stat-row"><span class="stat-key">Framework</span><span class="stat-val">TensorFlow / Keras</span></div>
        <div class="stat-row"><span class="stat-key">Input Features</span><span class="stat-val">11</span></div>
        <div class="stat-row"><span class="stat-key">Output</span><span class="stat-val">Churn Probability</span></div>
        <div class="stat-row"><span class="stat-key">Threshold</span><span class="stat-val">0.50</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎯 Risk Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info">
        <div class="stat-row"><span class="stat-key">🟢 Low Risk</span><span class="stat-val">0% – 40%</span></div>
        <div class="stat-row"><span class="stat-key">🟡 Medium Risk</span><span class="stat-val">40% – 60%</span></div>
        <div class="stat-row"><span class="stat-key">🔴 High Risk</span><span class="stat-val">60% – 100%</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color:#2a2a4a'>", unsafe_allow_html=True)
    st.caption("Built with ❤️ using Streamlit & TensorFlow")


# ─── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🏦 Customer Churn Predictor</h1>
    <p>Enter customer details below to predict the likelihood of churn using a trained ANN model.</p>
</div>
""", unsafe_allow_html=True)


# ─── Input Form ─────────────────────────────────────────────────────────────────
with st.form("churn_form"):

    # --- Row 1: Personal Info ---
    st.markdown('<div class="section-card"><div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        geography = st.selectbox(
            "🌍 Geography",
            options=one_hot_encoder_geo.categories_[0],
            help="Country where the customer is located"
        )
    with col2:
        gender = st.selectbox(
            "⚧ Gender",
            options=label_encoder_gender.classes_,
            help="Customer's gender"
        )
    with col3:
        age = st.slider(
            "🎂 Age",
            min_value=18, max_value=92, value=35,
            help="Customer's age in years"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Row 2: Financial Info ---
    st.markdown('<div class="section-card"><div class="section-title">💰 Financial Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        credit_score = st.number_input(
            "📊 Credit Score",
            min_value=300, max_value=900, value=650, step=1,
            help="Customer's credit score (300–900)"
        )
    with col5:
        balance = st.number_input(
            "🏧 Account Balance ($)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f",
            help="Current account balance. Enter 0 if no balance."
        )
    with col6:
        estimated_salary = st.number_input(
            "💵 Estimated Salary ($)",
            min_value=0.0, value=50000.0, step=500.0, format="%.2f",
            help="Customer's estimated annual salary"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Row 3: Account Info ---
    st.markdown('<div class="section-card"><div class="section-title">🏦 Account Details</div>', unsafe_allow_html=True)
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        tenure = st.slider(
            "📅 Tenure (Years)",
            min_value=0, max_value=10, value=5,
            help="Number of years the customer has been with the bank"
        )
    with col8:
        num_of_products = st.slider(
            "📦 No. of Products",
            min_value=1, max_value=4, value=1,
            help="Number of bank products the customer uses"
        )
    with col9:
        has_cr_card_input = st.radio(
            "💳 Has Credit Card?",
            options=["Yes", "No"],
            index=0,
            horizontal=True,
            help="Does the customer hold a credit card?"
        )
        has_cr_card = 1 if has_cr_card_input == "Yes" else 0
    with col10:
        is_active_input = st.radio(
            "✅ Active Member?",
            options=["Yes", "No"],
            index=0,
            horizontal=True,
            help="Is the customer an active bank member?"
        )
        is_active_member = 1 if is_active_input == "Yes" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Submit Button ---
    submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True)


# ─── Prediction Logic ───────────────────────────────────────────────────────────
if submitted:
    with st.spinner("🧠 Running prediction..."):

        # Build input dataframe
        input_data = pd.DataFrame({
            'CreditScore':     [credit_score],
            'Gender':          [label_encoder_gender.transform([gender])[0]],
            'Age':             [age],
            'Tenure':          [tenure],
            'Balance':         [balance],
            'NumOfProducts':   [num_of_products],
            'HasCrCard':       [has_cr_card],
            'IsActiveMember':  [is_active_member],
            'EstimatedSalary': [estimated_salary],
        })

        # One-hot encode Geography
        geo_encoded    = one_hot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # Merge & scale
        input_data       = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction_proba = float(model.predict(input_data_scaled, verbose=0)[0][0])

    # ── Determine risk tier ──────────────────────────────────────────────────────
    if prediction_proba < 0.40:
        risk_tier  = "low"
        risk_emoji = "🟢"
        risk_label = "Low Risk"
        risk_color = "#2ecc71"
        card_class = "result-card-safe"
        verdict    = "Customer is likely to <b>STAY</b>"
        advice     = "This customer shows strong retention signals. Keep up the engagement!"
    elif prediction_proba < 0.60:
        risk_tier  = "medium"
        risk_emoji = "🟡"
        risk_label = "Medium Risk"
        risk_color = "#f39c12"
        card_class = "result-card-medium"
        verdict    = "Customer is <b>AT RISK</b>"
        advice     = "Consider proactive outreach — a small nudge could prevent churn."
    else:
        risk_tier  = "high"
        risk_emoji = "🔴"
        risk_label = "High Risk"
        risk_color = "#e74c3c"
        card_class = "result-card-churn"
        verdict    = "Customer is likely to <b>CHURN</b>"
        advice     = "Immediate intervention recommended — offer retention incentives."

    pct = round(prediction_proba * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Result")

    res_col, gauge_col = st.columns([1, 1])

    # ── Result card ──────────────────────────────────────────────────────────────
    with res_col:
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:1.1rem; color:{risk_color}; font-weight:700; margin-bottom:0.4rem;">
                {risk_emoji} {risk_label}
            </div>
            <div class="prob-value" style="color:{risk_color};">{pct}%</div>
            <div style="font-size:0.95rem; color:#ccc; margin-top:0.3rem;">Churn Probability</div>
            <hr class="custom-divider" style="border-color:{risk_color}33; margin:1rem 0;">
            <div class="result-sub">{verdict}</div>
            <div style="font-size:0.85rem; color:#aaa; margin-top:0.6rem;">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gauge / breakdown metrics ────────────────────────────────────────────────
    with gauge_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Probability Breakdown</div>', unsafe_allow_html=True)

        st.markdown(f"**Churn probability:** `{pct}%`")
        st.progress(prediction_proba)

        st.markdown(f"**Retention probability:** `{round((1 - prediction_proba) * 100, 1)}%`")
        st.progress(1 - prediction_proba)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Churn %",     f"{pct}%",                        delta=None)
        m2.metric("Retention %", f"{round((1-prediction_proba)*100,1)}%", delta=None)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary expander ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            "Feature": [
                "Geography", "Gender", "Age", "Credit Score",
                "Balance ($)", "Estimated Salary ($)", "Tenure (Years)",
                "No. of Products", "Has Credit Card", "Active Member"
            ],
            "Value": [
                geography, gender, age, credit_score,
                f"${balance:,.2f}", f"${estimated_salary:,.2f}", tenure,
                num_of_products,
                "Yes" if has_cr_card else "No",
                "Yes" if is_active_member else "No"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # Placeholder when no prediction yet
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Fill in the customer details above and click **Predict Churn** to see the result.", icon="ℹ️")
