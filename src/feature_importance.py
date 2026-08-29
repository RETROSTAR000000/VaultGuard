"""Create a permutation-importance report for the saved VaultGuard model."""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cross-platform sys.path fix
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "is_fraud"
ID_COLUMN = "transaction_id"


def create_feature_importance() -> pd.DataFrame:
    """Calculate and save PR-AUC permutation importance on the hold-out split."""
    data = pd.read_csv(PROJECT_ROOT / "data" / "credit_card_fraud_2026.csv")
    data = data.drop_duplicates().dropna(subset=[TARGET_COLUMN]).copy()
    if pd.api.types.is_bool_dtype(data[TARGET_COLUMN]):
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = data[TARGET_COLUMN]
    _, test_features, _, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )
    model = joblib.load(PROJECT_ROOT / "models" / "vaultguard_fraud_model.joblib")
    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    result = permutation_importance(
        model, test_features, test_target, n_repeats=5, random_state=42,
        scoring=scorer, n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": test_features.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(PROJECT_ROOT / "reports" / "feature_importance.csv", index=False)
    top_features = importance.head(10).sort_values("importance_mean")
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"])
    plt.title("Top 10 Features Influencing Fraud Detection")
    plt.xlabel("Permutation Importance (PR-AUC decrease)")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    return importance


if __name__ == "__main__":
    print(create_feature_importance().head(10).to_string(index=False))
