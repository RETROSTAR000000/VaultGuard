# VaultGuard — CreditShield AI

CreditShield AI is a machine-learning project for identifying potentially
fraudulent credit-card transactions from anonymized transaction data and
prioritizing risky transactions for review.

## Project goal

The project will compare fraud-detection models that address severe class
imbalance, select an appropriate fraud-risk threshold, and present transaction
scores in a Streamlit dashboard.

## Current project structure

```text
VaultGuard/
├── data/
│   ├── raw/                 # Place creditcard.csv here; it is not committed
│   └── processed/
├── models/                  # Saved model artifacts will be created later
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── figures/
├── src/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Dataset setup

Download Kaggle's *Credit Card Fraud Detection* dataset and save it as:

```text
data/raw/creditcard.csv
```

The raw CSV is excluded from Git. It must contain `Time`, `Amount`, `V1`
through `V28`, and the `Class` target column, where `1` denotes fraud and `0`
denotes a legitimate transaction.

## Run the exploratory analysis

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

Once the dataset is in place, run all notebook cells. The notebook checks data
quality, duplicate records, class imbalance, and transaction amount/time
distributions. The anonymized `V1`–`V28` fields are not interpreted as specific
real-world banking attributes.

## Next steps

1. Verify the dataset and class-distribution analysis.
2. Add stratified data preprocessing and a logistic-regression baseline.
3. Compare models, tune the decision threshold, and save the selected model.
4. Build the Streamlit scoring dashboard and document measured results.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, Jupyter, Plotly, and
Streamlit.
