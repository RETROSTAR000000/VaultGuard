# VaultGuard

*Detect suspicious card transactions before they cause financial loss.*

## Project overview

VaultGuard is an end-to-end credit-card fraud-detection system that uses
machine learning to predict transaction fraud risk, flag suspicious payments,
and provide an interactive monitoring dashboard.

## Project details

| Item | Value |
| --- | --- |
| Project type | Credit Card Fraud Detection System |
| Goal | Identify and prioritize suspicious credit-card transactions |
| Target column | `is_fraud` |
| Technology | Python, pandas, scikit-learn, Streamlit, Plotly |

## Current project structure

```text
VaultGuard/
├── app.py
├── data/
│   ├── credit_card_fraud_2026.csv
│   └── sample_transactions.csv
├── models/
│   ├── vaultguard_fraud_model.joblib
│   └── model_config.json
├── notebooks/
├── reports/
├── src/
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Run locally

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

The training script performs a stratified split, compares logistic regression
and random forest models, selects the model with the best PR-AUC, tunes a
fraud-alert threshold, and writes metrics to `reports/`.

To test the dashboard, upload `data/sample_transactions.csv`. Do not include
`is_fraud` or `transaction_id` in files submitted for scoring.

## Resume description

**VaultGuard — Credit Card Fraud Detection System**  
Built an end-to-end machine-learning system to detect potentially fraudulent
credit-card transactions. Performed exploratory data analysis, handled class
imbalance, compared classification models, optimized fraud-alert thresholds,
and developed a Streamlit dashboard for transaction risk scoring and
investigation.
