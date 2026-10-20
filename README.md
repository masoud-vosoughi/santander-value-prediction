# Santander Transaction Value Prediction

**A reproducible, experiment-driven regression project** based on the 2018 [Kaggle Santander Value Prediction Challenge](https://www.kaggle.com/competitions/santander-value-prediction-challenge).

> **Project stage:** Data exploration, constant baselines, and initial Decision Tree experiments. **Not** a production-ready model or a claim to leaderboard performance.

## What problem does this solve?

Given anonymized numeric features for a potential banking customer, estimate the value of their transaction. This is **supervised regression**, not churn prediction, a classification problem, or customer lifetime value estimation.

**Primary metric — RMSLE**:

$$
\operatorname{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left[\ln(1+\hat{y}_i)-\ln(1+y_i)\right]^2}
$$

RMSLE evaluates deviations in logarithmic space. It reduces the dominance of large-value targets compared with RMSE in original units. Predictions and targets must be non-negative.

## Dataset

| Property | Description |
|---|---|
| Source | [OpenML dataset 42572](https://www.openml.org/search?id=42572&type=data) |
| Original challenge | [Santander Value Prediction](https://www.kaggle.com/competitions/santander-value-prediction-challenge) |
| Training observations | 4,459 |
| Feature columns | 4,991 numerical, anonymized |
| Target | `target` (transaction value) |
| Availability | Labeled training data from OpenML; **not** the original unlabeled Kaggle test file |

Source data is **not** committed here. Review any applicable dataset redistribution and usage terms before using the data. OpenML access/cache is required to run the notebooks.

## Research questions and what we learned

| Question | Evidence | Interpretation / action |
|---|---|---|
| Are there missing values? | 0 missing cells | No imputation currently needed |
| Are there constant columns? | 256, all zero | Candidate for removal in a training-fitted preprocessing pipeline |
| Are zeroes common? | 96.85% of feature values | Do not drop zero entries indiscriminately |
| Could non-zero row counts help? | Mean 157, max 1,989; Spearman ρ ≈ −0.0447 with target | Try as a feature only if CV justifies it |
| Is target skewed? | Mean ≈ 5.94M vs median 2.26M; max 40M | Consider log-target modeling and RMSLE |
| Does using features beat a constant prediction? | Tree better than baseline on all five training-CV folds | Continue model experiments |
| Does a deeper tree always improve CV? | Improvements level off past depth 10 | Track generalization gap and choose reasonable complexity |
| Does regularization help? | Leaf size 50 outperforms 5 and 20 under five-fold CV | Keep as current best **tested** tree setting |

**Important:** The original input fields are anonymized. We do not claim to know the underlying banking meaning of individual columns.

## Results so far

### Initial holdout experiments (80% train / 20% validation)

| Constant baseline | Validation RMSLE |
|---|---:|
| Mean | 2.0351 |
| Median | 1.7021 |
| **Log-Mean** | **1.6963** |

A Decision Tree with `max_depth=5`, `min_samples_leaf=20` obtained train RMSLE **1.5782** and holdout validation RMSLE **1.5862**. These are **preliminary** holdout readings, not untouched test performance.

### Paired five-fold CV on the 80% training portion

| Approach | Mean RMSLE |
|---|---:|
| Log-Mean constant baseline | 1.7644 |
| Initial tree (depth 5, leaf 20) | 1.6935 |
| **Best tested tree (depth 10, leaf 50)** | **1.6458** |

The best tested tree reduces the baseline's mean CV RMSLE by approximately **6.72%** on these folds. The baseline is recomputed inside each training fold, and every compared model uses the same folds.

**Scope/limitations:** Only a small set of hyperparameters has been explored. Repeatedly comparing models on the same folds can make the selected model's CV score optimistic. The original 20% holdout has already been used for exploratory checks, so we **do not** claim it is an untouched final test set. No Kaggle leaderboard score is claimed.

## Repository layout

```text
santander-value-prediction/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_decision_tree_experiments.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── SETUP_AND_VERIFICATION.md
```

**Narrative flow:**
1. **Data understanding** — shape, feature dtypes, zero prevalence, row-level activity, target distribution.
2. **Baseline models** — mean, median, and RMSLE-aligned log-mean constants.
3. **Decision Tree experiments** — holdout check, paired CV, depth/leaf-size studies, model-selection caveats.

Each notebook repeats the **minimal loading and split setup** intentionally so it can be run independently in VS Code. We will centralize reusable logic in `src/` only when doing so actually reduces duplication.

## Installation and execution

Tested originally on macOS with Python **3.12** in a project `.venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
code .
```

In VS Code, ensure the **Python** and **Jupyter** extensions are installed. Open any `.ipynb`, select the `.venv` kernel, then choose **Restart Kernel and Run All**. Initial OpenML download requires internet access. Read [SETUP_AND_VERIFICATION.md](SETUP_AND_VERIFICATION.md) for copying these files into an existing repo and for acceptance checks.

## Methodological guardrails

- Targets are log-transformed for RMSE training/scoring where RMSLE equivalence holds; `np.expm1()` restores the original scale for values shown to users.
- Preprocessing, feature selection, and fitted transformations must be learned from **training folds**, not from validation targets/data.
- Do not compare a holdout score to a CV score as if they used identical data partitions.
- Do not infer causal relations or business semantics from anonymized numeric columns.
- Preserve the distinction between **observed CV improvement** and **unbiased new-data performance**.

## Planned next steps

- Compare Random Forest and gradient boosting under consistent folds and metric.
- Evaluate zero-based/statistical feature engineering and training-fitted constant-column removal.
- Use defensible final evaluation procedures and error analysis.
- Refactor reusable modeling code into modules and add meaningful tests, without unnecessary infrastructure.
- Review and attribute public competition solution ideas, separately from our own experimental findings.

## Attribution

- [Original Kaggle competition](https://www.kaggle.com/competitions/santander-value-prediction-challenge)
- [OpenML data catalog entry](https://www.openml.org/search?id=42572&type=data)

The code and write-up here are a guided independent portfolio exercise, not an official Santander solution.
