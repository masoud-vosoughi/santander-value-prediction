from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import add_engineered_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "santander_lgbm.joblib"


def predict_values(X: pd.DataFrame) -> np.ndarray:
    """Predict target values from raw input features."""

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
