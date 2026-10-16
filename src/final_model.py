"""Model finalny (notebook 03).

Co tu jest:
1. choose_lightgbm_config() — 3 najlepsze konfiguracje LightGBM z próbki sprawdzone na pełnym treningu.
2. Zapis i odczyt parametrów modelu finalnego.
3. evaluate_on_test() — jednorazowa ocena na zbiorze testowym.
4. four_class_reference() — wynik pomocniczy dla oryginalnych 4 klas.
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
# 1. Wybór konfiguracji na pełnym zbiorze treningowym
# ---------------------------------------------------------------------------

def choose_lightgbm_config(train, y, folds, n_configs=3):
    """CV na pełnym train dla n najlepszych konfiguracji LightGBM ze strojenia na próbce.

    Zwraca (tabela średnia ± SD, parametry najlepszej, jej metryki per fold, jej OOF).
    """
    candidates = load_configs("lightgbm")[:n_configs]
    configs = {f"config_{i + 1}": (fit_predict_lightgbm, params)
               for i, params in enumerate(candidates)}
    per_fold, _, oof = compare_models(train, y, folds, configs)

    table = summarize(per_fold)
    best = table.loc[table["macro_f1"].idxmax(), "model"]
    best_params = configs[best][1]
    best_per_fold = per_fold[per_fold["model"] == best].assign(model="lightgbm")
    return table, best_params, best_per_fold, oof[best]


# ---------------------------------------------------------------------------
# 2. Parametry modelu finalnego
# ---------------------------------------------------------------------------

def save_final_params(params):
    with open(FINAL_CONFIG_PATH, "w") as f:
        json.dump({"params": params}, f, indent=2)


def load_final_params():
    with open(FINAL_CONFIG_PATH) as f:
        return json.load(f)["params"]


# ---------------------------------------------------------------------------
# 3. Ocena na teście
# ---------------------------------------------------------------------------

def evaluate_on_test(model, test, y_test):
    """Jednorazowa ocena: metryki z przedziałem bootstrap, per źródło, per rok, macierz pomyłek.

    Model wybiera klasę o największym prawdopodobieństwie (model.predict).
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
# 4. Wynik pomocniczy: 4 klasy
# ---------------------------------------------------------------------------

def four_class_reference(train, tests, params):
    """Ta sama konfiguracja wytrenowana na oryginalnych 4 klasach.

    Tylko do porównania ze starą wersją projektu, nie do wyboru modelu.
    tests: słownik {nazwa: DataFrame}. Zwraca tabelę metryk (wiersz = zbiór testowy).
    """
    model = fit_lightgbm(train, train["Severity"].to_numpy() - 1, params)
    rows = {}
    for name, test in tests.items():
        rows[name] = compute_metrics(test["Severity"].to_numpy() - 1, model.predict(test),
                                     CLASS_NAMES_4)
    return pd.DataFrame(rows).T
