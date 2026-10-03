import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom header styling
st.title("🛡️ Real-Time Transaction Fraud Detection")
st.markdown("Enter transaction parameters below to evaluate fraud risk in real-time.")

# Load artifacts
@st.cache_resource
def load_model():
    model = joblib.load('fraud_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, expected_features = load_model()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Layout: Two input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sender & Transaction Details")
    txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150000.0, step=1000.0)
    oldbalanceOrg = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=150000.0, step=1000.0)
    newbalanceOrig = st.number_input("Sender Balance After Transaction ($)", min_value=0.0, value=0.0, step=1000.0)
    hourOfDay = st.slider("Hour of Transaction (0 - 23)", min_value=0, max_value=23, value=14)

with col2:
    st.subheader("Recipient Details")
    oldbalanceDest = st.number_input("Recipient Initial Balance ($)", min_value=0.0, value=0.0, step=1000.0)
    newbalanceDest = st.number_input("Recipient Balance After Transaction ($)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    threshold = st.slider("Operational Alert Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
                          help="Transactions with a probability above this threshold will trigger an alert.")

# Prediction logic
st.markdown("---")
if st.button("Evaluate Transaction Risk", type="primary", use_container_width=True):
    # Map type to numeric if your model expects numeric, or leave as string
    # (Matches standard Transfer=1 / Cash_Out=0 convention)
    type_numeric = 1 if txn_type == "TRANSFER" else 0

    input_data = {
        'type': type_numeric if 'type' in expected_features and model.feature_name_ and isinstance(expected_features[0], str) else txn_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'hourOfDay': hourOfDay
    }

    # Format as DataFrame aligning exactly with expected feature order
    df_input = pd.DataFrame([input_data])
    df_input = df_input[expected_features]

    # Run prediction
    fraud_prob = model.predict_proba(df_input)[0][1]
    is_fraud = fraud_prob >= threshold

    # Results display
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Fraud Risk Score", value=f"{fraud_prob * 100:.2f}%")

    with res_col2:
        if is_fraud:
            st.error("🚨 **ALERT: High Risk of Fraudulent Activity Detected!**")
            st.write("This transaction exhibits patterns consistent with account takeover or drain attacks. Flagged for review.")
        else:
            st.success("✅ **CLEARED: Transaction Appears Legitimate.**")
            st.write("Risk probability falls within safe parameters.")
