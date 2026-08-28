"""Feature preprocessing shared by fraud-detection experiments."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def create_preprocessor(numerical_columns, categorical_columns):
    """Scale numeric features and one-hot encode categorical features."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Supports scikit-learn releases prior to the sparse_output argument.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("numeric", RobustScaler(), numerical_columns),
            (
                "categorical",
                encoder,
                categorical_columns,
            ),
        ]
    )
