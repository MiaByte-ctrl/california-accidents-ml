"""Model evaluation (notebooks 02–03).

What is here:
1. compute_metrics() — all metrics for one set of predictions.
2. cross_validate_model() — one model on all folds.
3. random_search() + saving and loading the tuning results.
4. compare_models() — several models on the same folds; summarize() — mean ± SD.
5. Metrics per group (e.g. per source), bootstrap and confusion matrix — for the test sets.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import ParameterSampler

from src.data import CLASS_NAMES, RESULTS_DIR

# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """macro-F1 (main metric), balanced accuracy, QWK and the recall of each class.

    A class that does not occur in y_true (possible in a small group, e.g. one data source
    in one period) gets a missing recall (NaN) and is left out of macro-F1 and balanced
    accuracy, instead of being counted as 0.
    """
    all_labels = list(range(len(class_names)))
    present = [k for k in all_labels if np.any(y_true == k)]
    recalls = recall_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    result = {
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        # balanced accuracy = mean recall over the classes that occur
        "balanced_acc": np.mean([recalls[k] for k in present]),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    for k, name in enumerate(class_names):
        result[f"recall_{name}"] = recalls[k] if k in present else np.nan
    return result


# ---------------------------------------------------------------------------
# 2. Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(fit_predict, df, y, folds, params=None):
    """Trains and evaluates a model on every fold.

    fit_predict is one of the functions from src/models.py. Preprocessing happens
    inside it, so it only sees the training data of the given fold.

    Returns (metrics table: one row per fold, probabilities for all rows of df).
    The probability for a row comes from a model that did not see that row
    (so-called out-of-fold predictions).
    """
    params = params or {}
    rows = []
    oof_proba = None
    for fold, (train_idx, val_idx) in enumerate(folds):
        start = time.time()
        proba = fit_predict(df.iloc[train_idx], y[train_idx], df.iloc[val_idx], params)
        if oof_proba is None:
            oof_proba = np.zeros((len(df), proba.shape[1]))
        oof_proba[val_idx] = proba
        metrics = compute_metrics(y[val_idx], proba.argmax(axis=1))
        metrics["fold"] = fold
        metrics["seconds"] = time.time() - start
        rows.append(metrics)
    return pd.DataFrame(rows), oof_proba


# ---------------------------------------------------------------------------
# 3. Hyperparameter tuning
# ---------------------------------------------------------------------------

def random_search(fit_predict, param_distributions, df, y, folds, n_iter=10, random_state=42):
    """Random search: n_iter random configurations, each scored by macro-F1 in CV.

    Works like RandomizedSearchCV from sklearn, but accepts any fit_predict function
    (including the PyTorch network), so all models are tuned the same way.
    Returns a table of configurations sorted from best to worst. The 'params' column keeps
    the original parameter dictionary (in a pandas table, e.g. max_depth=30 would become 30.0).
    """
    rows = []
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
    for i, params in enumerate(sampler):
        fold_table, _ = cross_validate_model(fit_predict, df, y, folds, params)
        row = dict(params)
        row["params"] = params
        row["macro_f1_mean"] = fold_table["macro_f1"].mean()
        row["macro_f1_std"] = fold_table["macro_f1"].std()
        row["qwk_mean"] = fold_table["qwk"].mean()
        row["seconds"] = fold_table["seconds"].sum()
        rows.append(row)
        print(f"  [{i + 1}/{n_iter}] macro-F1 {row['macro_f1_mean']:.4f}  {params}")
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def save_search(name, table):
    """Saves the tuning results: a table (CSV, to look at) and the parameters (JSON, to load)."""
    csv_path = os.path.join(RESULTS_DIR, f"search_{name}.csv")
    table.drop(columns="params").to_csv(csv_path, index=False)
    configs = list(table["params"])  # from best to worst
    # default=float turns numpy numbers into plain numbers that JSON can store
    with open(os.path.join(RESULTS_DIR, f"search_{name}.json"), "w") as f:
        json.dump(configs, f, indent=2, default=float)


def load_configs(name):
    """A model's configurations from tuning, best first. [{}] for a model without tuning."""
    path = os.path.join(RESULTS_DIR, f"search_{name}.json")
    if not os.path.exists(path):
        return [{}]
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------

def compare_models(df, y, folds, configs):
    """Several models on the same folds.

    configs: a dictionary {name: (fit_predict function, parameters)}.
    Returns:
    - per_fold: the metrics of every model in every fold (column 'model'),
    - by_source: the metrics of every model separately for each data source,
    - oof: a dictionary {name: out-of-fold probabilities}.
    """
    per_fold, by_source, oof = [], [], {}
    for name, (fit_predict, params) in configs.items():
        fold_table, proba = cross_validate_model(fit_predict, df, y, folds, params)
        fold_table.insert(0, "model", name)
        per_fold.append(fold_table)

        source_table = metrics_by_group(y, proba.argmax(axis=1), df["Source"].to_numpy())
        source_table = source_table.reset_index(names="Source")
        source_table["model"] = name
        by_source.append(source_table)

        oof[name] = proba
        print(f"{name:18s} macro-F1 {fold_table['macro_f1'].mean():.4f} "
              f"± {fold_table['macro_f1'].std():.4f}")
    return pd.concat(per_fold, ignore_index=True), pd.concat(by_source, ignore_index=True), oof


def summarize(per_fold):
    """Mean and standard deviation of every metric for every model (one row per model)."""
    metrics = [c for c in per_fold.columns if c not in ("model", "fold", "seconds")]
    grouped = per_fold.groupby("model", sort=False)
    table = grouped[metrics].mean().join(grouped[metrics].std().add_suffix("_std"))
    table["seconds"] = grouped["seconds"].sum()
    return table.reset_index()


# ---------------------------------------------------------------------------
# 5. Evaluation on the test sets
# ---------------------------------------------------------------------------

def metrics_by_group(y_true, y_pred, groups):
    """Metrics separately for each value of a group (e.g. each source or year)."""
    rows = {}
    for value in sorted(pd.unique(groups)):
        mask = np.asarray(groups == value)
        rows[value] = compute_metrics(y_true[mask], y_pred[mask])
        rows[value]["n"] = int(mask.sum())
    return pd.DataFrame(rows).T


def bootstrap_ci(y_true, y_pred, n_iterations=1000, random_state=42):
    """95% bootstrap confidence interval for every metric.

    We draw the test rows with replacement 1000 times and compute the metrics;
    the interval is the 2.5th and 97.5th percentile of the values obtained.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        samples.append(compute_metrics(y_true[idx], y_pred[idx]))
    samples = pd.DataFrame(samples)
    return pd.DataFrame({
        "value": pd.Series(compute_metrics(y_true, y_pred)),
        "ci_low": samples.quantile(0.025),
        "ci_high": samples.quantile(0.975),
    })


def confusion_table(y_true, y_pred, class_names=CLASS_NAMES):
    """Confusion matrix (rows = true class, columns = predicted class)."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
