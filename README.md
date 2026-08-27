# VaultGuard
Duo Project of Arnab & Aritra
creditshield-ai/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Place dataset here (do not commit)
│   └── processed/
│       └── processed_creditcard.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_threshold_tuning.ipynb
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── fraud_model.joblib
│   └── model_config.json
│
├── reports/
│   ├── figures/
│   └── model_metrics.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore