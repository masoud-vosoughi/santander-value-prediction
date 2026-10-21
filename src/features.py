import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X_new = X.copy()

    row_sum = X.sum(axis=1)
    row_max = X.max(axis=1)

    X_new["row_sum"] = row_sum
    X_new["row_max"] = row_max
    X_new["row_max_share"] = (row_max / row_sum.replace(0, np.nan)).fillna(0)

    return X_new
