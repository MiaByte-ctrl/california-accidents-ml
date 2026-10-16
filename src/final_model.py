"""Final model (notebook 03).

What is here:
1. choose_lightgbm_config() — the 3 best LightGBM configurations from the sample,
   checked again on the full training set.
2. Saving and loading the parameters of the final model.
3. evaluate_on_test() — a single evaluation on a test set.
4. four_class_reference() — reference result for the original 4 classes.
"""
import json
import os

import pandas as pd

from src.data import CLASS_NAMES_4, PROJECT_DIR, RESULTS_DIR
from src.evaluation import (bootstrap_ci, compare_models, compute_metrics, confusion_table,
                            load_configs, metrics_by_group, summarize)
from src.models import fit_lightgbm, fit_predict_lightgbm

FINAL_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "lightgbm_final.joblib")
FINAL_CONFIG_PATH = os.path.join(RESULTS_DIR, "final_config.json")


# ---------------------------------------------------------------------------
# 1. Choosing the configuration on the full training set
# ---------------------------------------------------------------------------

def choose_lightgbm_config(train, y, folds, n_configs=3):
    """CV on the full train set for the n best LightGBM configurations from the sample tuning.

    Returns (table in the same format as the tuning results: parameters + macro-F1 mean and SD,
    parameters of the best configuration, its per-fold metrics, its OOF predictions).
    """
    candidates = load_configs("lightgbm")[:n_configs]
    configs = {f"config_{i + 1}": (fit_predict_lightgbm, params)
               for i, params in enumerate(candidates)}
    per_fold, _, oof = compare_models(train, y, folds, configs)

    summary = summarize(per_fold).set_index("model")
    rows = []
    for name, (_, params) in configs.items():
        row = dict(params)
        row["macro_f1_mean"] = summary.loc[name, "macro_f1"]
        row["macro_f1_std"] = summary.loc[name, "macro_f1_std"]
        row["qwk_mean"] = summary.loc[name, "qwk"]
        row["seconds"] = summary.loc[name, "seconds"]
        rows.append(row)
    table = pd.DataFrame(rows)

    best = summary["macro_f1"].idxmax()
    best_per_fold = per_fold[per_fold["model"] == best].assign(model="lightgbm")
    return table, configs[best][1], best_per_fold, oof[best]


# ---------------------------------------------------------------------------
# 2. Parameters of the final model
# ---------------------------------------------------------------------------

def save_final_params(params):
    with open(FINAL_CONFIG_PATH, "w") as f:
        json.dump({"params": params}, f, indent=2)


def load_final_params():
    with open(FINAL_CONFIG_PATH) as f:
        return json.load(f)["params"]


# ---------------------------------------------------------------------------
# 3. Evaluation on a test set
# ---------------------------------------------------------------------------

def evaluate_on_test(model, test, y_test):
    """Single evaluation of the final model on one test set.

    Returns metrics with bootstrap intervals, metrics per source and per year, and the
    confusion matrix.

    The model picks the class with the highest probability (model.predict).
    """
    y_pred = model.predict(test)
    return {
        "metrics": bootstrap_ci(y_test, y_pred),
        "by_source": metrics_by_group(y_test, y_pred, test["Source"].to_numpy()),
        "by_year": metrics_by_group(y_test, y_pred, test["Year"].to_numpy()),
        "confusion": confusion_table(y_test, y_pred),
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# 4. Reference result: 4 classes
# ---------------------------------------------------------------------------

def four_class_reference(train, tests, params):
    """The same configuration trained on the original 4 classes.

    Only for comparison with the old version of the project, not for choosing the model.
    tests: a dictionary {name: DataFrame}. Returns a metrics table (one row per test set).
    """
    model = fit_lightgbm(train, train["Severity"].to_numpy() - 1, params)
    rows = {}
    for name, test in tests.items():
        rows[name] = compute_metrics(test["Severity"].to_numpy() - 1, model.predict(test),
                                     CLASS_NAMES_4)
    return pd.DataFrame(rows).T
