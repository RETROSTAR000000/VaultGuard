"""Train baseline fraud models and save the selected production artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cross-platform sys.path fix
# Ensures `from src.xxx import ...` works when the script is executed directly
# (e.g. `python src/train.py`) without `pip install -e .`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "is_fraud"
ID_COLUMN = "transaction_id"


def train_and_save():
    """Fit baselines, select by PR-AUC, tune F1 threshold, and save artifacts."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "credit_card_fraud_2026.csv")
    df = df.drop_duplicates().dropna(subset=[TARGET_COLUMN]).copy()
    if pd.api.types.is_bool_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]
    categorical_columns = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", create_preprocessor(numerical_columns, categorical_columns)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", create_preprocessor(numerical_columns, categorical_columns)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    comparison = []
    probabilities = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.50).astype(int)
        probabilities[name] = probability
        comparison.append(
            {
                "Model": name,
                "Precision": precision_score(y_test, prediction, zero_division=0),
                "Recall": recall_score(y_test, prediction, zero_division=0),
                "F1 Score": f1_score(y_test, prediction, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probability),
                "PR-AUC": average_precision_score(y_test, probability),
            }
        )

    comparison_df = pd.DataFrame(comparison).sort_values("PR-AUC", ascending=False)
    model_name = comparison_df.iloc[0]["Model"]
    final_model = models[model_name]
    final_probability = probabilities[model_name]

    threshold_results = []
    for threshold in (round(value * 0.05, 2) for value in range(1, 20)):
        prediction = (final_probability >= threshold).astype(int)
        threshold_results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1_score": f1_score(y_test, prediction, zero_division=0),
                "alerts": int(prediction.sum()),
            }
        )
    threshold_df = pd.DataFrame(threshold_results)
    final_threshold = float(
        threshold_df.loc[threshold_df["f1_score"].idxmax(), "threshold"]
    )

    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    threshold_df.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    X_test.head(100).to_csv(PROJECT_ROOT / "data" / "sample_transactions.csv", index=False)
    joblib.dump(final_model, models_dir / "vaultguard_fraud_model.joblib")
    with open(models_dir / "model_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": model_name,
                "target_column": TARGET_COLUMN,
                "threshold": final_threshold,
                "feature_columns": X.columns.tolist(),
                "features_excluded": [ID_COLUMN, TARGET_COLUMN],
            },
            file,
            indent=4,
        )

    return comparison_df, threshold_df, model_name, final_threshold


if __name__ == "__main__":
    comparison, thresholds, name, threshold = train_and_save()
    print(comparison.to_string(index=False))
    print(f"Selected model: {name}")
    print(f"Selected threshold: {threshold:.2f}")
