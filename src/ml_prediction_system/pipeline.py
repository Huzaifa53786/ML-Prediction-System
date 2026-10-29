"""Training, evaluation, reporting, and inference for tabular ML datasets."""
from __future__ import annotations
import json
import logging
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
ARTIFACT_FORMAT_VERSION = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    return value


def load_dataset(path: str | Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV, remove rows with missing targets, and validate the feature/target split."""
    source = Path(path)
    if not source.is_file(): raise FileNotFoundError(f"Dataset not found: {source}")
    frame = pd.read_csv(source)
    if frame.empty: raise ValueError("Dataset has no rows.")
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing. Available columns: {list(frame.columns)}")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing_targets = int(frame[target].isna().sum())
    if missing_targets:
        LOGGER.warning("Dropping %d row(s) with a missing target.", missing_targets)
        frame = frame.loc[frame[target].notna()].copy()
    if frame.empty: raise ValueError("No rows remain after dropping missing target values.")
    y = frame.pop(target)
    if frame.shape[1] == 0: raise ValueError("Dataset must contain feature columns besides the target.")
    if y.nunique(dropna=True) < 2: raise ValueError("Target must contain at least two distinct values.")
    return frame, y


def infer_task(y: pd.Series, task: str) -> str:
    if task != "auto": return task
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if y.nunique() <= 20 and pd.api.types.is_integer_dtype(y): return "classification"
    return "regression"


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical))
    if not transformers: raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _candidate_models(task: str, seed: int) -> dict[str, Any]:
    if task == "classification":
        return {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
            "extra_trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)}
    return {"ridge": Ridge(), "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1)}


def _metrics(task: str, actual: pd.Series, predicted: np.ndarray) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if task == "regression":
        return {"r2": r2_score(actual, predicted), "mae": mean_absolute_error(actual, predicted),
            "rmse": mean_squared_error(actual, predicted) ** 0.5}, None
    return {"accuracy": accuracy_score(actual, predicted),
        "precision_weighted": precision_score(actual, predicted, average="weighted", zero_division=0),
        "recall_weighted": recall_score(actual, predicted, average="weighted", zero_division=0),
        "f1_weighted": f1_score(actual, predicted, average="weighted", zero_division=0)}, \
        classification_report(actual, predicted, output_dict=True, zero_division=0)


def _plot_data(X: pd.DataFrame, y: pd.Series, task: str, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    numeric = X.select_dtypes(include=[np.number])
    cols = min(3, max(1, len(numeric)))
    rows = int(np.ceil((len(numeric) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    axes = axes.ravel()
    if task == "classification":
        y.astype(str).value_counts().plot(kind="bar", ax=axes[0], color="#3b82f6")
        axes[0].set_title("Target class counts")
    else:
        sns.histplot(y, ax=axes[0], kde=True, color="#3b82f6")
        axes[0].set_title("Target distribution")
    for i, column in enumerate(numeric.columns, start=1):
        sns.histplot(X[column].dropna(), ax=axes[i], kde=True, color="#14b8a6")
        axes[i].set_title(f"{column} distribution")
    for axis in axes[len(numeric) + 1:]: axis.set_visible(False)
    fig.suptitle("Dataset overview", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(results: pd.DataFrame, metric: str, output: Path) -> None:
    display = results.sort_values("selection_score", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=display, x="model", y="selection_score", hue="model", legend=False, ax=ax, palette="crest")
    ax.set_title(f"Cross-validation model comparison ({metric})")
    ax.set_xlabel("")
    ax.set_ylabel(f"CV score ({metric}; higher is better)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(data_path: str | Path, target: str, output_dir: str | Path = "artifacts", task: str = "auto",
          test_size: float = 0.2, cv: int = 5, random_state: int = 42,
          primary_metric: str | None = None) -> dict[str, Any]:
    """Compare candidate models using CV, evaluate the winner on a held-out test set, and save it."""
    if not 0.05 <= test_size <= 0.5: raise ValueError("test_size must be between 0.05 and 0.5.")
    if cv < 2: raise ValueError("cv must be at least 2.")
    X, y = load_dataset(data_path, target)
    task = infer_task(y, task)
    if task not in {"classification", "regression"}: raise ValueError("task must be auto, classification, or regression.")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    if task == "classification" and stratify is None:
        raise ValueError("Each class needs at least two rows for a stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
        random_state=random_state, stratify=stratify)
    metric = primary_metric or ("f1_weighted" if task == "classification" else "r2")
    scoring = {"f1_weighted": "f1_weighted", "accuracy": "accuracy", "roc_auc": "roc_auc_ovr_weighted",
        "r2": "r2", "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error"}
    allowed = {"f1_weighted", "accuracy", "roc_auc"} if task == "classification" else {"r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"}
    if metric not in allowed: raise ValueError(f"Metric {metric!r} is invalid for {task}; choose from {sorted(allowed)}")
    folds = min(cv, int(y_train.value_counts().min())) if task == "classification" else min(cv, len(y_train))
    if folds < 2: raise ValueError("Not enough training rows per class for cross-validation.")
    if folds < cv: LOGGER.warning("Reducing CV folds from %d to %d due to the smallest class size.", cv, folds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preprocessor = _make_preprocessor(X_train)
    comparison = []
    for name, estimator in _candidate_models(task, random_state).items():
        candidate = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        try:
            scores = cross_validate(candidate, X_train, y_train, cv=folds,
                scoring={"selection": scoring[metric]}, n_jobs=1, error_score="raise")
            vals = scores["test_selection"]
            comparison.append({"model": name, "selection_score": float(np.mean(vals)),
                "selection_std": float(np.std(vals)), "fold_scores": vals.tolist()})
            LOGGER.info("%s CV %s: %.4f +/- %.4f", name, metric, np.mean(vals), np.std(vals))
        except Exception as exc:
            LOGGER.exception("Skipping model %s after cross-validation error.", name)
            comparison.append({"model": name, "selection_score": None, "selection_std": None, "error": str(exc)})
    viable = [row for row in comparison if row["selection_score"] is not None and np.isfinite(row["selection_score"])]
    if not viable: raise RuntimeError("Every candidate model failed. Review the input data and logged errors.")
    best = max(viable, key=lambda row: row["selection_score"])
    model_name = best["model"]
    final_model = Pipeline([("preprocessor", preprocessor),
        ("model", _candidate_models(task, random_state)[model_name])])
    final_model.fit(X_train, y_train)
    predicted = final_model.predict(X_test)
    metrics, report = _metrics(task, y_test, predicted)
    if task == "classification" and hasattr(final_model, "predict_proba"):
        try:
            proba = final_model.predict_proba(X_test)
            if len(final_model.classes_) == 2: metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
            else: metrics["roc_auc_ovr_weighted"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        except ValueError: pass
    pd.DataFrame(comparison).to_csv(output / "model_comparison.csv", index=False)
    _plot_data(X, y, task, output / "data_overview.png")
    _plot_comparison(pd.DataFrame(viable), metric, output / "model_comparison.png")
    if task == "classification":
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, predicted, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} test confusion matrix")
        fig.tight_layout()
        fig.savefig(output / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, predicted, alpha=0.7, color="#2563eb")
        limits = [min(float(y_test.min()), float(np.min(predicted))), max(float(y_test.max()), float(np.max(predicted)))]
        ax.plot(limits, limits, "--", color="#dc2626")
        ax.set(xlabel="Actual", ylabel="Predicted", title=f"{model_name} test predictions")
        fig.tight_layout()
        fig.savefig(output / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    metadata = {"format_version": ARTIFACT_FORMAT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "target": target, "selected_model": model_name, "selection_metric": metric,
        "selection_cv_mean": best["selection_score"], "selection_cv_std": best["selection_std"],
        "test_metrics": metrics, "n_rows": len(X), "n_features": X.shape[1], "feature_columns": list(X.columns),
        "numeric_columns": numeric, "categorical_columns": [c for c in X.columns if c not in numeric],
        "missing_values_by_column": X.isna().sum().to_dict(),
        "class_labels": final_model.named_steps["model"].classes_.tolist() if task == "classification" else None,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__}}
    joblib.dump({"model": final_model, "metadata": metadata}, output / "model.joblib", compress=3)
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe({"metadata": metadata, "model_comparison": comparison,
            "classification_report": report}), file, indent=2, allow_nan=False)
    LOGGER.info("Saved selected %s model and reports to %s", model_name, output.resolve())
    return metadata


def predict(model_path: str | Path, data_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Predict from a saved artifact; unknown input features are ignored and missing ones imputed."""
    artifact_path = Path(model_path)
    if not artifact_path.is_file(): raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    bundle = joblib.load(artifact_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError("Unsupported model artifact format.")
    metadata = bundle["metadata"]
    if metadata.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"Unsupported artifact version: {metadata.get('format_version')}")
    X = pd.read_csv(data_path).replace([np.inf, -np.inf], np.nan)
    if X.empty: raise ValueError("Prediction input has no rows.")
    expected = metadata["feature_columns"]
    missing = [column for column in expected if column not in X.columns]
    extra = [column for column in X.columns if column not in expected]
    if missing:
        warnings.warn(f"Missing feature columns will be imputed: {missing}", UserWarning, stacklevel=2)
        for column in missing: X[column] = np.nan
    if extra: LOGGER.warning("Ignoring unexpected input columns: %s", extra)
    X = X[expected]
    estimator = bundle["model"]
    result = pd.DataFrame({"prediction": estimator.predict(X)})
    if metadata["task"] == "classification" and hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        labels = metadata.get("class_labels") or list(range(proba.shape[1]))
        for index, label in enumerate(labels): result[f"probability_{label}"] = proba[:, index]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    LOGGER.info("Wrote %d predictions to %s", len(result), destination.resolve())
    return result

