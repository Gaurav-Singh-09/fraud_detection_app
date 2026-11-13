# 🛡️ Real-Time Financial Fraud Detection

An interactive **Streamlit** dashboard powered by **LightGBM** and a hybrid rule-based risk calibration engine to detect unauthorized account drains, money laundering, and anomalous financial transactions in real time.

---

## ✨ Key Features

- **Interactive Transaction Simulator:** Configure sender/recipient balances, transaction channels (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`), amounts, and timestamps.
- **One-Click Simulation Presets:** Instantly test common scenarios:
  - 🟢 **Standard Grocery Payment** (Low-risk baseline)
  - 🟡 **Borderline Suspicious Cash-Out** (Elevated risk requiring manual review)
  - 🚨 **Midnight Account Drain** (High-risk account liquidation)
- **Dynamic Operational Threshold:** Adjust the alert cutoff slider (`5%` – `95%`) in real time to balance false positives against fraud capture rates.
- **Hybrid Risk Scoring & Anomaly Detection:** Combines LightGBM probability predictions with heuristic behavioral checks (account liquidation, pass-through mule accounts, off-peak hours, and ledger discrepancies) to provide transparent, explainable alerts.
- **Visual Risk Gauge:** Real-time Plotly gauge displaying the calculated risk score against your active threshold.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Visualization:** [Plotly](https://plotly.com/)
- **Machine Learning:** [LightGBM](https://lightgbm.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit application
├── fraud_model.pkl         # Pre-trained LightGBM classification model
├── model_features.pkl      # Serialized list of expected model feature names
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
