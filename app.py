from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="VaultGuard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(PROJECT_ROOT / "models" / "vaultguard_fraud_model.joblib")
    with open(PROJECT_ROOT / "models" / "model_config.json", encoding="utf-8") as file:
        config = json.load(file)
    return model, config


def get_risk_level(probability, threshold):
    if probability >= max(threshold, 0.70):
        return "High"
    if probability >= threshold:
        return "Medium"
    return "Low"


def create_reasons(row):
    checks = [
        (row.get("is_foreign_transaction", False), "Foreign transaction"),
        (row.get("is_new_merchant", False), "New merchant"),
        (row.get("used_vpn", False), "VPN usage detected"),
        (row.get("ip_country_mismatch", False), "IP-country mismatch"),
        (row.get("billing_shipping_mismatch", False), "Billing and shipping mismatch"),
        (row.get("cvv_retry_count", 0) >= 2, "Multiple CVV retries"),
        (row.get("velocity_score", 0) >= 70, "High transaction velocity"),
        (row.get("merchant_risk_score", 0) >= 70, "High merchant risk score"),
        (row.get("is_ai_generated_scam_attempt", False), "Possible AI-generated scam attempt"),
        (row.get("prior_disputes", 0) >= 1, "Customer has prior disputes"),
    ]
    reasons = [reason for condition, reason in checks if bool(condition)]
    return ", ".join(reasons) if reasons else "Model-based risk pattern"


st.title("🛡️ VaultGuard")
st.caption("Credit Card Fraud Detection and Transaction Risk Scoring")

try:
    model, config = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run `python -m src.train` from the project root.")
    st.stop()

st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Fraud classification threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(config["threshold"]),
    step=0.01,
)
st.sidebar.caption(f"Model: {config['model_name']}")
st.sidebar.caption(f"Target: {config['target_column']}")

st.header("Upload Transactions")
uploaded_file = st.file_uploader("Upload a CSV for fraud scoring", type=["csv"])
st.caption("Upload all training features. Do not include `is_fraud` or `transaction_id`.")

if uploaded_file is None:
    st.info("Use `data/sample_transactions.csv` to test the dashboard after training.")
    st.stop()

input_df = pd.read_csv(uploaded_file)
required_features = config["feature_columns"]
missing_columns = [column for column in required_features if column not in input_df.columns]
if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

X_input = input_df[required_features].copy()
probabilities = model.predict_proba(X_input)[:, 1]
results = input_df.copy()
results["fraud_probability"] = probabilities
results["fraud_risk_percent"] = (probabilities * 100).round(2)
results["prediction"] = (probabilities >= threshold).astype(int)
results["risk_level"] = results["fraud_probability"].apply(
    lambda value: get_risk_level(value, threshold)
)
results["flag_reasons"] = results.apply(create_reasons, axis=1)

total_transactions = len(results)
fraud_alerts = int(results["prediction"].sum())
alert_rate = fraud_alerts / total_transactions * 100 if total_transactions else 0
amount_at_risk = results.loc[results["prediction"] == 1, "amount_usd"].sum()

metric_columns = st.columns(4)
metric_columns[0].metric("Transactions analyzed", f"{total_transactions:,}")
metric_columns[1].metric("Fraud alerts", f"{fraud_alerts:,}")
metric_columns[2].metric("Alert rate", f"{alert_rate:.2f}%")
metric_columns[3].metric("Amount at risk", f"${amount_at_risk:,.2f}")

st.subheader("Fraud Probability Distribution")
chart = px.histogram(
    results,
    x="fraud_risk_percent",
    color="risk_level",
    nbins=30,
    category_orders={"risk_level": ["Low", "Medium", "High"]},
    color_discrete_map={"Low": "#3b82f6", "Medium": "#f59e0b", "High": "#dc2626"},
)
chart.update_layout(xaxis_title="Fraud Risk Score (%)", yaxis_title="Transactions")
st.plotly_chart(chart, use_container_width=True)

st.subheader("Fraud Alerts")
alerts = results[results["prediction"] == 1].sort_values("fraud_probability", ascending=False)
display_columns = [
    column
    for column in [
        "amount_usd", "merchant_category", "card_type", "auth_method", "channel",
        "is_foreign_transaction", "is_new_merchant", "velocity_score",
        "merchant_risk_score", "fraud_risk_percent", "risk_level", "flag_reasons",
    ]
    if column in alerts.columns
]
if alerts.empty:
    st.success("No transactions crossed the selected fraud threshold.")
else:
    st.dataframe(alerts[display_columns], use_container_width=True, hide_index=True)

st.subheader("All Scored Transactions")
st.dataframe(results.sort_values("fraud_probability", ascending=False), use_container_width=True, hide_index=True)
st.download_button(
    "Download Scored Results",
    results.to_csv(index=False).encode("utf-8"),
    file_name="vaultguard_scored_transactions.csv",
    mime="text/csv",
)
