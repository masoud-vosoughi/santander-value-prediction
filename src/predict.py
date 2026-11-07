from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import add_engineered_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "santander_lgbm.joblib"


def predict_values(X: pd.DataFrame) -> np.ndarray:

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not X.columns.is_unique:
        raise ValueError("Input contains duplicate column names.")
    if X.isna().any().any():
        raise ValueError("Input contains missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("All input columns must be numeric.")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Input contains infinite values.")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    expected_columns = artifact["feature_columns"]

    X_processed = add_engineered_features(X)

    if set(X_processed.columns) != set(expected_columns):
        raise ValueError("Input feature schema does not match the model.")

    X_processed = X_processed[expected_columns]

    predictions_log = model.predict(X_processed)

    predictions = np.clip(
        np.expm1(predictions_log),
        0,
        None,
    )

    return predictions
