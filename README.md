# Machine Learning Prediction System

A reusable, end-to-end tabular ML project for both classification and regression. It accepts a CSV and target column, validates the data, preprocesses numeric and categorical features without leaking test data, compares several estimators with cross-validation, evaluates the winner on a holdout set, saves a versioned model bundle, and generates plots and machine-readable reports.

## Features

- Feature engineering through numeric median imputation and scaling plus categorical mode imputation and one-hot encoding. Entirely missing training columns are retained safely, and unknown categories at inference time are supported. Unknown categories at inference time are supported.
- Automatic task inference (or explicit classification/regression selection).
- Model comparison using stratified cross-validation for classification and standard K-fold cross-validation for regression.
- Classification candidates: logistic regression, random forest, and extra trees. Regression candidates: ridge, random forest, and extra trees.
- Selection metrics: weighted F1, accuracy, or ROC AUC for classification; R², negative RMSE, or negative MAE for regression.
- Holdout evaluation metrics, classification report, confusion matrix or actual-versus-predicted chart, dataset distributions, and model comparison plot.
- Reproducible seeds, version metadata, input schema metadata, structured logs, artifact format versioning, and a CLI for batch inference.

## Quick start

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python data\generate_sample.py
ml-predict train --data data\sample_classification.csv --target purchased --task classification
ml-predict predict --model artifacts\model.joblib --data data\sample_classification.csv --output artifacts\sample_predictions.csv
```

On macOS/Linux, activate with `source .venv/bin/activate` and use forward slashes in paths. You can also run the CLI without installing its entry point: `python -m ml_prediction_system.cli ...` after installing the package.

The sample generator creates a deterministic synthetic classification dataset with numeric and categorical features, missing feature values, and 600 rows. Use this dataset only to exercise the pipeline; it is not a real-world benchmark.

## Train on your data

```powershell
ml-predict train --data path\to\data.csv --target target_column --task auto --output-dir artifacts --test-size 0.2 --cv 5 --random-state 42
```

Set `--task classification` or `--task regression` when automatic inference is not appropriate. Task inference treats text/boolean targets and integer targets with at most 20 unique values as classification. For classification, the default selection metric is weighted F1; for regression, it is R². Override with `--metric f1_weighted`, `accuracy`, `roc_auc`, `r2`, `neg_root_mean_squared_error`, or `neg_mean_absolute_error` as appropriate. ROC AUC selection uses one-vs-rest weighted scoring.

The pipeline drops rows whose target is missing and reports the count. Missing/infinite feature values are imputed. Feature preprocessing is fit inside each cross-validation fold and only on the training partition. Classification requires at least two examples per class; cross-validation folds are reduced when the smallest class has fewer training examples than requested.

## Batch inference

Pass a CSV with feature columns to the model's `predict` command:

```powershell
ml-predict predict --model artifacts\model.joblib --data new_rows.csv --output predictions.csv
```

The model stores the training feature schema. Missing columns are filled with missing values and imputed by the fitted pipeline; unexpected columns are ignored with a warning. Classification output includes predicted labels and probability columns when the selected model supports probability estimates.

## Output artifacts

Training writes the following under the output directory:

| Artifact | Contents |
|---|---|
| `model.joblib` | Fitted preprocessing + estimator and versioned metadata |
| `metrics.json` | Holdout metrics, cross-validation scores, classification report, schema, and library versions |
| `model_comparison.csv` | Candidate model CV means, standard deviations, and fold scores |
| `data_overview.png` | Target and numeric feature distributions |
| `model_comparison.png` | Cross-validation score comparison |
| `confusion_matrix.png` | Classification test-set result; regression instead writes `predictions_vs_actual.png` |

Joblib artifacts use Python pickle internally. Load only model files you trust, and keep training and inference environments on compatible scikit-learn versions. Retain the original dataset, source revision, and artifact together when reproducibility and auditability matter.

## Docker

```powershell
docker build -t ml-prediction-system .
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/artifacts:/app/artifacts" ml-prediction-system train --data /app/data/sample_classification.csv --target purchased --task classification
```

## Project layout

```text
src/ml_prediction_system/   Training and inference package
 data/generate_sample.py    Deterministic local demo data generator
```

## Operational notes

This project provides reproducible batch training and inference; it does not expose a hosted prediction API. A deployment owner should configure data access controls, data retention, monitoring and retraining criteria, and model approval for the actual environment. Choose evaluation metrics and split strategy that reflect the costs and time structure of your real use case. Random holdout splitting is not suitable for time-dependent or grouped observations; those need domain-specific splitting.

