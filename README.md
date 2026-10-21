# Santander Transaction Value Prediction

An end-to-end tabular regression project exploring **data analysis, model selection, feature engineering, and reusable inference** using the [Santander Value Prediction Challenge](https://www.kaggle.com/competitions/santander-value-prediction-challenge) dataset.

## Problem and data

Predict transaction values from anonymized numerical features. The labeled data is obtained from [OpenML (dataset 42572)](https://www.openml.org/search?id=42572&type=data), not the original Kaggle competition test set.

- **Observations:** 4,459; **input features:** 4,991; **target:** transaction value.
- **Data characteristics:** no missing values, 256 constant-zero columns, and approximately 96.85% zero entries.
- **Metric:** root mean squared logarithmic error (RMSLE). Models are trained on `log1p(target)` and predictions are converted back with `expm1`.

## Approach

1. Explore sparsity, target skewness, and data quality.
2. Establish constant-prediction baselines and compare tree-based regressors.
3. Select model settings using **5-fold cross-validation on an 80% development partition**.
4. Test row-level features individually and retain changes that improve cross-validation performance.
5. Serialize the selected LightGBM model and implement reusable prediction code with basic input-schema validation.

The selected engineered features are **`row_sum`**, **`row_max`**, and **`row_max_share = row_max / row_sum`** (with a safe zero-denominator fallback). They are computed from the original feature columns.

## Results

| Model | Mean 5-fold CV RMSLE ↓ |
|---|---:|
| Log-mean baseline | 1.7644 |
| Decision Tree | 1.6458 |
| Random Forest | 1.5427 |
| Gradient Boosting | 1.5306 |
| LightGBM (original features) | 1.4668 |
| **LightGBM + engineered features** | **1.4304** |

Adding the three engineered features improved the selected LightGBM model's CV RMSLE from **1.4668 to 1.4304**. The corresponding **exploratory holdout RMSLE** improved from **1.4504 to 1.4147**.

**Evaluation caveat:** The same CV folds were used repeatedly for model selection, and the 20% holdout was inspected during earlier development. These scores are **development results**, not an unbiased final test estimate or a Kaggle leaderboard result.

## Repository structure

```text
notebooks/
  01_data_understanding.ipynb
  02_baseline_models.ipynb
  03_decision_tree_experiments.ipynb
  04_ensemble_models.ipynb
  05_feature_engineering.ipynb
src/
  __init__.py
  features.py
  predict.py
models/
  santander_lgbm.joblib
requirements.txt
README.md
```

## Setup and usage

Python **3.12** is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, LightGBM also requires the OpenMP runtime: `brew install libomp` (Homebrew).

Run the notebooks in numbered order, or open each independently in VS Code and select the project's `.venv` kernel. A network connection is needed for the initial OpenML download.

To predict with the saved model using the **original 4,991-column input schema**:

```python
from sklearn.datasets import fetch_openml
from src.predict import predict_values

X = fetch_openml(data_id=42572, as_frame=True).data
X = X.drop(columns=["ID"], errors="ignore")
print(predict_values(X.head(3)))
```

The prediction module adds engineered features, validates the expected columns, loads the saved model, and returns nonnegative predictions in the original target scale. Load serialized models only from trusted sources.

## Limitations

Feature names are anonymized, so feature importance cannot establish business meaning or causality. The dataset is small relative to its dimensionality, model selection has reused the same folds, and a fully untouched test set is not available in this workflow. Raw data is not redistributed here; check the source's applicable usage terms.

## References

- [Santander Value Prediction Challenge (Kaggle)](https://www.kaggle.com/competitions/santander-value-prediction-challenge)
- [Dataset 42572 (OpenML)](https://www.openml.org/search?id=42572&type=data)
