from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_IMPORTANCE_IMAGE = REPORTS_DIR / "figures" / "feature_importance.png"

NUMERIC_FEATURES = {
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
}

# ---------------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VaultGuard | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS  — dark banking theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg:       #08111f;
    --surface:  #111c2e;
    --surface2: #17253b;
    --text:     #eef6ff;
    --muted:    #9fb3c8;
    --cyan:     #25d8f4;
    --purple:   #9b7bff;
    --green:    #30e39a;
    --yellow:   #ffca5f;
    --red:      #ff5c7a;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(37,216,244,0.10), transparent 28%),
        radial-gradient(circle at 85% 10%, rgba(155,123,255,0.14), transparent 28%),
        linear-gradient(135deg, #07111f 0%, #0c1728 55%, #0c1221 100%);
    color: var(--text);
    font-family: "DM Sans", sans-serif;
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", sans-serif;
    color: #f4f8ff;
    letter-spacing: -0.4px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091524 0%, #111f35 100%);
    border-right: 1px solid rgba(37,216,244,0.14);
}
[data-testid="stSidebar"] * { color: #eaf5ff; }

.hero-card {
    padding: 1.8rem 2rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(37,216,244,0.13), rgba(155,123,255,0.16));
    border: 1px solid rgba(112,213,255,0.22);
    box-shadow: 0 16px 44px rgba(0,0,0,0.25);
    animation: floatIn 0.65s ease-out;
}
.hero-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.hero-subtitle { color: #b6cae0; font-size: 1rem; }

.metric-card {
    background: linear-gradient(145deg, rgba(22,40,65,0.93), rgba(12,24,43,0.97));
    padding: 1.2rem 1.3rem;
    border-radius: 20px;
    border: 1px solid rgba(161,197,235,0.15);
    min-height: 140px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(37,216,244,0.48);
}
.metric-label  { color: #a9bfd4; font-size: 0.86rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-value  { font-family: "Space Grotesk", sans-serif; font-size: 2rem; font-weight: 700; margin-top: 0.35rem; }
.metric-note   { color: #7fdff0; font-size: 0.77rem; margin-top: 0.4rem; }

.status-pill  { padding: 0.3rem 0.8rem; border-radius: 99px; font-size: 0.79rem; font-weight: 700; display: inline-block; }
.low-risk     { background: rgba(48,227,154,0.13); color: #68efb7; border: 1px solid rgba(48,227,154,0.28); }
.medium-risk  { background: rgba(255,202,95,0.14); color: #ffd77f; border: 1px solid rgba(255,202,95,0.30); }
.high-risk    { background: rgba(255,92,122,0.15); color: #ff8aa1; border: 1px solid rgba(255,92,122,0.35); }

.section-title {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
    color: #d4e8ff;
}

[data-testid="stFileUploader"] {
    background: rgba(20,38,62,0.75);
    border: 1.5px dashed rgba(37,216,244,0.42);
    border-radius: 16px;
    padding: 0.75rem;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(161,197,235,0.16);
}

[data-testid="stTabs"] button {
    font-family: "DM Sans", sans-serif;
    font-weight: 600;
}

@keyframes floatIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared chart layout defaults
# ---------------------------------------------------------------------------
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#eef6ff",
    font_family="DM Sans",
    margin=dict(t=44, b=28, l=16, r=16),
)
RISK_COLORS = {"Low": "#30e39a", "Medium": "#ffca5f", "High": "#ff5c7a"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(PROJECT_ROOT / "models" / "vaultguard_fraud_model.joblib")
    with open(PROJECT_ROOT / "models" / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


@st.cache_data
def load_report(filename: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / filename)


def validate_input(input_df: pd.DataFrame, required_features: list[str]) -> str | None:
    if input_df.empty:
        return "The uploaded CSV has no transactions to score."
    dupes = input_df.columns[input_df.columns.duplicated()].unique().tolist()
    if dupes:
        return "Duplicate column names are not supported: " + ", ".join(dupes)
    missing = [c for c in required_features if c not in input_df.columns]
    if missing:
        return "Missing required columns: " + ", ".join(missing)
    bad_numeric = []
    for col in NUMERIC_FEATURES.intersection(required_features):
        vals = pd.to_numeric(input_df[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals).all():
            bad_numeric.append(col)
    if bad_numeric:
        return "Numeric values are missing or invalid in: " + ", ".join(sorted(bad_numeric))
    return None


def get_risk_level(prob: float, threshold: float) -> str:
    if prob >= max(threshold, 0.70):
        return "High"
    if prob >= threshold:
        return "Medium"
    return "Low"


def get_fraud_reasons(row: pd.Series) -> str:
    checks = [
        (row.get("is_foreign_transaction",      False), "Foreign transaction"),
        (row.get("is_new_merchant",             False), "New merchant"),
        (row.get("used_vpn",                    False), "VPN usage detected"),
        (row.get("ip_country_mismatch",         False), "IP-country mismatch"),
        (row.get("billing_shipping_mismatch",   False), "Billing / shipping mismatch"),
        (row.get("cvv_retry_count",                 0) >= 2, "Multiple CVV retries"),
        (row.get("velocity_score",                  0) >= 70, "High transaction velocity"),
        (row.get("merchant_risk_score",             0) >= 70, "High merchant risk score"),
        (row.get("is_ai_generated_scam_attempt", False), "Possible AI-generated scam attempt"),
        (row.get("prior_disputes",                  0) >= 1, "Customer has prior disputes"),
    ]
    reasons = [r for cond, r in checks if bool(cond)]
    return "; ".join(reasons) if reasons else "Model-detected risk pattern"


def score_transactions(
    input_df: pd.DataFrame,
    model,
    required_features: list[str],
    threshold: float,
) -> pd.DataFrame:
    probs = model.predict_proba(input_df[required_features].copy())[:, 1]
    out = input_df.copy()
    out["fraud_probability"]  = probs
    out["fraud_risk_percent"] = (probs * 100).round(2)
    out["prediction"]         = (probs >= threshold).astype(int)
    out["risk_level"]         = out["fraud_probability"].apply(lambda v: get_risk_level(v, threshold))
    out["risk_reasons"]       = out.apply(get_fraud_reasons, axis=1)
    return out


def metric_card(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
try:
    model, config = load_artifacts()
except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as err:
    st.error("Model artifacts could not be loaded. Run `python -m src.train` from the project root.")
    st.caption(f"Detail: {err}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ VaultGuard")
st.sidebar.caption("Fraud Intelligence Console")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Fraud Scanner", "Dashboard", "Model Intelligence", "About"],
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Monitoring Controls")

live_mode = st.sidebar.toggle(
    "Live monitoring mode",
    value=True,
    help="Visual indicator for real-time fraud-monitoring simulation.",
)

threshold = st.sidebar.slider(
    "Fraud alert threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(config.get("threshold", 0.25)),
    step=0.01,
    help="Lower values catch more fraud but raise more alerts.",
)

show_all_columns = st.sidebar.toggle(
    "Show all transaction columns",
    value=False,
)

st.sidebar.divider()
st.sidebar.caption(f"Active model: **{config.get('model_name', 'Saved Model')}**")
st.sidebar.caption(f"Risk threshold: **{threshold:.0%}**")
if live_mode:
    st.sidebar.success("🟢 Live monitoring active")
else:
    st.sidebar.warning("⏸️ Live monitoring paused")


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🛡️ VaultGuard</div>
        <div class="hero-subtitle">
            AI-powered credit-card fraud intelligence and transaction monitoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE — Fraud Scanner
# ===========================================================================
if page == "Fraud Scanner":

    st.markdown('<div class="section-title">Upload Transaction File</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a CSV containing transaction data",
        type=["csv"],
        help="File must contain all feature columns used during model training.",
    )

    if uploaded_file is None:
        st.info("Upload `data/sample_transactions.csv` to begin fraud scoring.")
    else:
        with st.spinner("VaultGuard is scoring transactions…"):
            try:
                input_df = pd.read_csv(uploaded_file)
            except Exception as err:
                st.error(f"Could not read the uploaded CSV: {err}")
                st.stop()

            validation_error = validate_input(input_df, config["feature_columns"])
            if validation_error:
                st.error(validation_error)
                st.stop()

            try:
                results = score_transactions(input_df, model, config["feature_columns"], threshold)
            except (TypeError, ValueError) as err:
                st.error("The uploaded data could not be scored with the trained model.")
                st.caption(f"Detail: {err}")
                st.stop()

        st.session_state["scored_results"]   = results
        st.session_state["scored_file_name"] = uploaded_file.name
        st.success(f"Scoring complete — {len(results):,} transactions analysed.")

        total      = len(results)
        alerts     = int(results["prediction"].sum())
        alert_rate = alerts / total * 100 if total else 0
        at_risk    = results.loc[results["prediction"] == 1, "amount_usd"].sum()
        high_risk  = int((results["risk_level"] == "High").sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Transactions analysed", f"{total:,}",        "Current uploaded file",       "#25d8f4")
        with c2:
            metric_card("Fraud alerts",          f"{alerts:,}",       f"{alert_rate:.2f}% of batch", "#ff5c7a")
        with c3:
            metric_card("High-risk cases",       f"{high_risk:,}",    "Immediate review priority",   "#ffca5f")
        with c4:
            metric_card("Amount at risk",        f"${at_risk:,.2f}",  "Value of flagged payments",   "#30e39a")

        tab_overview, tab_alerts, tab_analytics, tab_export = st.tabs([
            "📊 Overview", "🚨 Alert Queue", "📈 Analytics", "⬇️ Export",
        ])

        # ── Overview ──────────────────────────────────────────────────────
        with tab_overview:
            col_l, col_r = st.columns(2)

            with col_l:
                rc = results["risk_level"].value_counts().reset_index()
                rc.columns = ["risk_level", "count"]
                donut = px.pie(
                    rc, names="risk_level", values="count",
                    hole=0.62, color="risk_level",
                    color_discrete_map=RISK_COLORS,
                    title="Risk-Level Distribution",
                )
                donut.update_layout(**CHART_LAYOUT, legend_title_text="")
                st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

            with col_r:
                hist = px.histogram(
                    results, x="fraud_risk_percent", color="risk_level",
                    nbins=35, color_discrete_map=RISK_COLORS,
                    title="Fraud-Risk Score Distribution",
                )
                hist.update_layout(**CHART_LAYOUT,
                                   xaxis_title="Fraud Risk Score (%)",
                                   yaxis_title="Transactions")
                st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-title">Top 10 High-Risk Transactions</div>',
                        unsafe_allow_html=True)
            top10 = results.sort_values("fraud_probability", ascending=False).head(10)
            overview_cols = ["transaction_id", "amount_usd", "merchant_category",
                             "channel", "fraud_risk_percent", "risk_level", "risk_reasons"]
            overview_cols = [c for c in overview_cols if c in top10.columns]
            st.dataframe(top10[overview_cols], use_container_width=True, hide_index=True)

        # ── Alert Queue ───────────────────────────────────────────────────
        with tab_alerts:
            st.markdown('<div class="section-title">Fraud Investigation Queue</div>',
                        unsafe_allow_html=True)
            alert_df = (results[results["prediction"] == 1]
                        .sort_values("fraud_probability", ascending=False))

            if alert_df.empty:
                st.success("No transactions exceed the selected fraud threshold.")
            else:
                risk_filter = st.multiselect(
                    "Filter by risk level",
                    ["High", "Medium", "Low"],
                    default=["High", "Medium"],
                )
                filtered = alert_df[alert_df["risk_level"].isin(risk_filter)]

                alert_cols = [
                    "transaction_id", "amount_usd", "merchant_category",
                    "card_type", "auth_method", "channel", "device_type",
                    "is_foreign_transaction", "is_new_merchant", "used_vpn",
                    "ip_country_mismatch", "velocity_score", "merchant_risk_score",
                    "fraud_risk_percent", "risk_level", "risk_reasons",
                ]
                if show_all_columns:
                    alert_cols = list(filtered.columns)
                alert_cols = [c for c in alert_cols if c in filtered.columns]
                st.dataframe(filtered[alert_cols], use_container_width=True, hide_index=True)

                if not filtered.empty:
                    st.markdown('<div class="section-title">Investigate a Transaction</div>',
                                unsafe_allow_html=True)
                    sel_idx = st.selectbox(
                        "Select a transaction",
                        filtered.index.tolist(),
                        format_func=lambda i: (
                            f"{filtered.loc[i, 'transaction_id'] if 'transaction_id' in filtered.columns else f'Row {i}'}"
                            f"  —  {filtered.loc[i, 'fraud_risk_percent']:.2f}% risk"
                        ),
                    )
                    txn = filtered.loc[sel_idx]
                    dl, dr = st.columns(2)
                    dl.metric("Fraud risk",   f"{txn['fraud_risk_percent']:.2f}%")
                    dl.metric("Amount (USD)", f"${txn['amount_usd']:,.2f}")
                    dr.write("**Risk signals:**")
                    dr.write(txn["risk_reasons"])

        # ── Analytics ─────────────────────────────────────────────────────
        with tab_analytics:
            acol_l, acol_r = st.columns(2)

            with acol_l:
                if "merchant_category" in results.columns:
                    cat_risk = (
                        results.groupby("merchant_category")["fraud_probability"]
                        .mean().reset_index()
                        .sort_values("fraud_probability", ascending=False)
                    )
                    cat_chart = px.bar(
                        cat_risk, x="merchant_category", y="fraud_probability",
                        color="fraud_probability",
                        color_continuous_scale=["#30e39a", "#ffca5f", "#ff5c7a"],
                        title="Avg Fraud Risk by Merchant Category",
                    )
                    cat_chart.update_layout(
                        **CHART_LAYOUT,
                        xaxis_title="Merchant Category",
                        yaxis_title="Avg Fraud Probability",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(cat_chart, use_container_width=True,
                                    config={"displayModeBar": False})

            with acol_r:
                if "time_of_day_hour" in results.columns:
                    hourly = (
                        results.groupby("time_of_day_hour")["fraud_probability"]
                        .mean().reset_index()
                    )
                    hourly_chart = px.line(
                        hourly, x="time_of_day_hour", y="fraud_probability",
                        markers=True, title="Avg Fraud Risk by Hour of Day",
                    )
                    hourly_chart.update_traces(line_color="#25d8f4", marker_color="#9b7bff")
                    hourly_chart.update_layout(
                        **CHART_LAYOUT,
                        xaxis_title="Hour of Day",
                        yaxis_title="Avg Fraud Probability",
                    )
                    st.plotly_chart(hourly_chart, use_container_width=True,
                                    config={"displayModeBar": False})

            if "amount_usd" in results.columns:
                hover_extra = [c for c in ["transaction_id", "merchant_category", "channel"]
                               if c in results.columns]
                scatter = px.scatter(
                    results, x="amount_usd", y="fraud_risk_percent",
                    color="risk_level", hover_data=hover_extra,
                    color_discrete_map=RISK_COLORS, opacity=0.72,
                    title="Transaction Amount vs Fraud Risk",
                )
                scatter.update_layout(
                    **CHART_LAYOUT,
                    xaxis_title="Transaction Amount (USD)",
                    yaxis_title="Fraud Risk Score (%)",
                )
                st.plotly_chart(scatter, use_container_width=True,
                                config={"displayModeBar": False})

        # ── Export ────────────────────────────────────────────────────────
        with tab_export:
            st.markdown('<div class="section-title">Export Fraud Review Report</div>',
                        unsafe_allow_html=True)
            st.write(
                "Download all scored transactions including fraud probability, "
                "risk level, and investigation signals."
            )
            st.download_button(
                "⬇️ Download VaultGuard Fraud Report",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="vaultguard_scored_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if live_mode:
            st.success("🟢 Live monitoring mode is active — ready for the next transaction batch.")


# ===========================================================================
# PAGE — Dashboard
# ===========================================================================
elif page == "Dashboard":

    st.markdown('<div class="section-title">Fraud Monitoring Dashboard</div>',
                unsafe_allow_html=True)
    results = st.session_state.get("scored_results")

    if results is None:
        st.info("No batch scored yet. Open **Fraud Scanner** and upload a CSV file first.")
    else:
        fname = st.session_state.get("scored_file_name", "uploaded CSV")
        st.success(f"Showing latest scored batch: **{fname}** — {len(results):,} transactions.")

        total     = len(results)
        alerts    = int(results["prediction"].sum())
        at_risk   = results.loc[results["prediction"] == 1, "amount_usd"].sum()
        high_risk = int((results["risk_level"] == "High").sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Transactions",  f"{total:,}",        "Current batch",           "#25d8f4")
        with c2:
            metric_card("Fraud alerts",  f"{alerts:,}",       f"{alerts/total*100:.1f}%","#ff5c7a")
        with c3:
            metric_card("High-risk",     f"{high_risk:,}",    "Needs immediate review",  "#ffca5f")
        with c4:
            metric_card("At risk (USD)", f"${at_risk:,.2f}",  "Flagged payment value",   "#30e39a")

        col_l, col_r = st.columns(2)

        with col_l:
            risk_summary = (
                results.groupby("risk_level").size().reset_index(name="transactions")
            )
            bar = px.bar(
                risk_summary, x="risk_level", y="transactions",
                color="risk_level", color_discrete_map=RISK_COLORS,
                title="Transaction Risk Summary",
            )
            bar.update_layout(**CHART_LAYOUT,
                              xaxis_title="Risk Level", yaxis_title="Transactions")
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        with col_r:
            donut2 = px.pie(
                risk_summary, names="risk_level", values="transactions",
                hole=0.60, color="risk_level", color_discrete_map=RISK_COLORS,
                title="Risk-Level Share",
            )
            donut2.update_layout(**CHART_LAYOUT, legend_title_text="")
            st.plotly_chart(donut2, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">All Scored Transactions</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            results.sort_values("fraud_probability", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ===========================================================================
# PAGE — Model Intelligence
# ===========================================================================
elif page == "Model Intelligence":

    st.markdown('<div class="section-title">Model Intelligence</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Active model",       config.get("model_name", "Saved Model"),
                    "Production pipeline",            "#25d8f4")
    with c2:
        metric_card("Decision threshold", f"{threshold:.0%}",
                    "Probability required for alert", "#ffca5f")
    with c3:
        metric_card("Target variable",    config.get("target_column", "is_fraud"),
                    "Fraud label used in training",   "#30e39a")

    st.markdown('<div class="section-title">Evaluation Reports</div>', unsafe_allow_html=True)
    try:
        comparison = load_report("model_comparison.csv")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        thresholds_df = load_report("threshold_analysis.csv")
        thr_chart = px.line(
            thresholds_df, x="threshold",
            y=["precision", "recall", "f1_score"],
            markers=True,
            title="Precision–Recall Trade-off by Alert Threshold",
            color_discrete_sequence=["#25d8f4", "#30e39a", "#ffca5f"],
        )
        thr_chart.update_layout(**CHART_LAYOUT,
                                xaxis_title="Fraud alert threshold",
                                yaxis_title="Score")
        st.plotly_chart(thr_chart, use_container_width=True)
    except (FileNotFoundError, pd.errors.ParserError) as err:
        st.warning(f"Evaluation reports unavailable: {err}")

    if FEATURE_IMPORTANCE_IMAGE.exists():
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        st.image(str(FEATURE_IMPORTANCE_IMAGE), caption="Permutation Feature Importance")
    else:
        st.info("Feature-importance chart not yet generated. Run `python -m src.feature_importance`.")

    st.markdown('<div class="section-title">How VaultGuard Scores Risk</div>',
                unsafe_allow_html=True)
    st.markdown("""
1. The analyst uploads a credit-card transaction CSV file.
2. VaultGuard applies the preprocessing pipeline saved alongside the ML model.
3. The model produces a fraud probability (0–1) for every transaction.
4. Transactions whose probability exceeds the chosen **threshold** enter the alert queue.
5. Supporting signals — new merchant, VPN usage, foreign transaction, high velocity,
   elevated merchant risk — are surfaced as investigation reasons for the analyst.
""")

    st.markdown('<div class="section-title">Key Metrics Explained</div>', unsafe_allow_html=True)
    st.markdown("""
- **Precision** — of transactions flagged as fraud, how many were actually fraud.
- **Recall** — of actual fraud transactions, how many the model caught.
- **F1-score** — harmonic mean of precision and recall.
- **ROC-AUC** — ability to rank fraud above legitimate transactions across all thresholds.
- **PR-AUC** — especially informative for rare-event classification.
""")

    st.warning(
        "VaultGuard is an educational decision-support prototype. "
        "It must not automatically approve, decline, or block real customer transactions."
    )


# ===========================================================================
# PAGE — About
# ===========================================================================
else:
    st.markdown('<div class="section-title">About VaultGuard</div>', unsafe_allow_html=True)

    st.markdown("""
### AI-Powered Fraud Intelligence

**VaultGuard** is a credit-card fraud-detection and transaction-monitoring prototype built with
machine learning. It scores uploaded transaction batches, identifies suspicious payments,
prioritises high-risk cases, and provides clear investigation signals for fraud analysts.

### Technology Stack
- Python · Pandas · NumPy
- Scikit-learn · Imbalanced-learn · XGBoost
- Plotly · Streamlit · Joblib

### Key Data Signals
| Signal | Description |
|---|---|
| `amount_usd` | Transaction value |
| `merchant_category` | Business type |
| `card_type` | Credit / Debit / Prepaid |
| `auth_method` | PIN, Chip, Contactless, etc. |
| `channel` | Online / POS / ATM |
| `device_type` | Mobile, Desktop, etc. |
| `is_foreign_transaction` | Cross-border flag |
| `is_new_merchant` | First-time merchant flag |
| `used_vpn` | VPN / proxy detection |
| `ip_country_mismatch` | IP vs billing country |
| `billing_shipping_mismatch` | Address inconsistency |
| `cvv_retry_count` | Failed CVV attempts |
| `velocity_score` | Recent transaction frequency |
| `merchant_risk_score` | Merchant-level risk rating |
| `prior_disputes` | Historical chargebacks |
| `is_ai_generated_scam_attempt` | Synthetic fraud indicator |

### Disclaimer
VaultGuard supports human analysts and must never autonomously block real customer transactions.
""")
