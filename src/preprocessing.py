"""Feature preprocessing shared by fraud-detection experiments."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def create_preprocessor(numerical_columns, categorical_columns):
    """Scale numeric features and one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", RobustScaler(), numerical_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
