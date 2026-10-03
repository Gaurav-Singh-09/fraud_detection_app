import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Real-Time Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ----------------- LOAD ARTIFACTS -----------------
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, expected_features = load_model()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    threshold = st.slider(
        "Operational Alert Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help="Scores at or above this cutoff are flagged for investigation."
    )

# ----------------- MAIN HEADER -----------------
st.title("🛡️ Real-Time Financial Fraud Detection")
st.write("Simulate transaction parameters to evaluate potential laundering or unauthorized account drains.")

# ----------------- QUICK LOAD PRESETS -----------------
st.markdown("#### ⚡ Quick Load Simulation Scenarios")

if "tx_type" not in st.session_state:
    st.session_state.tx_type = "TRANSFER"
    st.session_state.amount = 250000.00
    st.session_state.old_orig = 300000.00
    st.session_state.new_orig = 50000.00
    st.session_state.old_dest = 0.00
    st.session_state.new_dest = 50000.00
    st.session_state.hour = 3

col_p1, col_p2, col_p3 = st.columns(3)

if col_p1.button("🟢 Standard Grocery Payment", use_container_width=True):
    st.session_state.tx_type = "PAYMENT"
    st.session_state.amount = 65.20
    st.session_state.old_orig = 1500.00
    st.session_state.new_orig = 1434.80
    st.session_state.old_dest = 45000.00
    st.session_state.new_dest = 45065.20
    st.session_state.hour = 14
    st.rerun()

if col_p2.button("🚨 Midnight Account Drain", use_container_width=True):
    st.session_state.tx_type = "TRANSFER"
    st.session_state.amount = 300000.00
    st.session_state.old_orig = 300000.00
    st.session_state.new_orig = 0.00
    st.session_state.old_dest = 0.00
    st.session_state.new_dest = 0.00
    st.session_state.hour = 3
    st.rerun()

if col_p3.button("🟡 High-Value Cash-Out", use_container_width=True):
    st.session_state.tx_type = "CASH_OUT"
    st.session_state.amount = 90000.00
    st.session_state.old_orig = 100000.00
    st.session_state.new_orig = 10000.00
    st.session_state.old_dest = 2000.00
    st.session_state.new_dest = 92000.00
    st.session_state.hour = 21
    st.rerun()

st.markdown("---")

# ----------------- INPUT CONTROLS -----------------
col_sender, col_recipient = st.columns(2)

types_list = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
current_type_idx = types_list.index(st.session_state.tx_type) if st.session_state.tx_type in types_list else 0

with col_sender:
    st.subheader("Sender & Transaction Details")
    tx_type = st.selectbox("Transaction Type", types_list, index=current_type_idx)
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(st.session_state.amount), step=1000.0)
    oldbalanceOrg = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=float(st.session_state.old_orig), step=1000.0)
    newbalanceOrig = st.number_input("Sender Balance After Transaction ($)", min_value=0.0, value=float(st.session_state.new_orig), step=1000.0)
    hour = st.slider("Hour of Transaction (0 - 23)", min_value=0, max_value=23, value=int(st.session_state.hour))

with col_recipient:
    st.subheader("Recipient Details")
    oldbalanceDest = st.number_input("Recipient Initial Balance ($)", min_value=0.0, value=float(st.session_state.old_dest), step=1000.0)
    newbalanceDest = st.number_input("Recipient Balance After Transaction ($)", min_value=0.0, value=float(st.session_state.new_dest), step=1000.0)

# ----------------- GAUGE FUNCTION -----------------
def create_gauge(prob, thresh):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'size': 38, 'color': "white"}},
        title={'text': "Calculated Risk Score", 'font': {'size': 20, 'color': "#94a3b8"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#ef4444" if prob >= thresh else "#22c55e"},
            'bgcolor': "#0f172a",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, thresh * 100], 'color': '#064e3b'},
                {'range': [thresh * 100, 100], 'color': '#7f1d1d'}
            ],
            'threshold': {
                'line': {'color': "#facc15", 'width': 4},
                'thickness': 0.8,
                'value': thresh * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig

# ----------------- PREDICTION PIPELINE -----------------
st.markdown("###")
if st.button("Evaluate Transaction Risk", type="primary", use_container_width=True):
    input_dict = {
        'amount': float(amount),
        'oldbalanceOrg': float(oldbalanceOrg),
        'newbalanceOrig': float(newbalanceOrig),
        'oldbalanceDest': float(oldbalanceDest),
        'newbalanceDest': float(newbalanceDest),
        'hour': int(hour),
        'orig_balance_err': float((oldbalanceOrg - amount) - newbalanceOrig),
        'dest_balance_err': float((oldbalanceDest + amount) - newbalanceDest),
    }

    # One-hot encode if model expected dummy columns
    for t in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']:
        col_name = f"type_{t}"
        if col_name in expected_features:
            input_dict[col_name] = 1 if tx_type == t else 0

    input_df = pd.DataFrame([input_dict])

    # If the model used categorical 'type' feature directly
    if 'type' in expected_features:
        input_df['type'] = pd.Categorical([tx_type], categories=types_list)

    # Ensure all expected columns exist
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure strict column ordering matching training
    input_df = input_df[expected_features]

    # Convert non-categorical columns to numeric float/int
    for col in input_df.columns:
        if col != 'type':
            input_df[col] = pd.to_numeric(input_df[col])

    # Run inference
    prob = float(model.predict_proba(input_df)[0][1])
    is_fraud = prob >= threshold

    st.markdown("---")

    g_col, stat_col = st.columns([1.2, 1])

    with g_col:
        st.plotly_chart(create_gauge(prob, threshold), use_container_width=True)

    with stat_col:
        st.markdown("### Risk Evaluation Verdict")
        if is_fraud:
            st.error("🚨 **ALERT: High Risk of Fraud Detected!**")
            st.write(f"The transaction scored **{prob * 100:.2f}%**, exceeding your operational alert threshold of **{threshold * 100:.0f}%**.")
            st.markdown("#### Detected Risk Anomalies:")
            if newbalanceOrig == 0 and amount > 10000:
                st.markdown("• **Complete Account Liquidation**: Sender balance completely zeroed.")
            if oldbalanceDest == 0 and newbalanceDest == 0:
                st.markdown("• **Mule/Pass-Through Endpoint**: Destination retained zero funds.")
            if hour in [0, 1, 2, 3, 4, 5]:
                st.markdown("• **Off-Peak Hours**: Initiated during non-standard early-morning hours.")
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                st.markdown(f"• **High-Risk Channel**: Executed through `{tx_type}`.")
        else:
            st.success("✅ **CLEARED: Transaction Appears Legitimate.**")
            st.write(f"The transaction scored **{prob * 100:.2f}%**, remaining safely below the **{threshold * 100:.0f}%** alert threshold.")
            st.balloons()