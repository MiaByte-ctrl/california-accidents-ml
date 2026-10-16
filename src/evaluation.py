"""Ocena modeli (notebooki 02–03).

Co tu jest:
1. compute_metrics() — wszystkie metryki dla jednego zestawu predykcji.
2. cross_validate() — jeden model na wszystkich foldach.
3. random_search() + zapis i odczyt wyników strojenia.
4. compare_models() — kilka modeli na tych samych foldach; summarize() — średnia ± SD.
5. Metryki osobno dla grup (np. źródeł), bootstrap i macierz pomyłek — do oceny na testach.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import ParameterSampler

from src.data import CLASS_NAMES, RESULTS_DIR

# ---------------------------------------------------------------------------
# 1. Metryki
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """macro-F1 (główna metryka), balanced accuracy, QWK i recall każdej klasy."""
    labels = list(range(len(class_names)))
    result = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, value in zip(class_names, recalls):
        result[f"recall_{name}"] = value
    return result


# ---------------------------------------------------------------------------
# 2. Walidacja krzyżowa
# ---------------------------------------------------------------------------

def cross_validate(fit_predict, df, y, folds, params=None):
    """Uczy i ocenia model na każdym foldzie.

    fit_predict to jedna z funkcji z src/models.py. Preprocessing dzieje się
    w jej środku, więc widzi tylko dane treningowe danego foldu.

    Zwraca (tabela metryk: wiersz = fold, prawdopodobieństwa dla wszystkich wierszy df).
    Prawdopodobieństwo dla wiersza pochodzi z modelu, który tego wiersza nie widział
    (tzw. predykcje out-of-fold).
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
# 3. Strojenie hiperparametrów
# ---------------------------------------------------------------------------

def random_search(fit_predict, param_distributions, df, y, folds, n_iter=10, random_state=42):
    """Losowe przeszukiwanie: n_iter losowych konfiguracji, każda oceniona w CV po macro-F1.

    Działa jak RandomizedSearchCV ze sklearn, ale przyjmuje dowolną funkcję fit_predict
    (także sieć w PyTorch), więc wszystkie modele stroimy tak samo.
    Zwraca tabelę konfiguracji posortowaną od najlepszej. Kolumna 'params' przechowuje
    oryginalny słownik parametrów (w tabeli pandas zamieniłby np. max_depth=30 na 30.0).
    """
    rows = []
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
    for i, params in enumerate(sampler):
        fold_table, _ = cross_validate(fit_predict, df, y, folds, params)
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
    """Zapisuje wyniki strojenia: tabelę (CSV, do oglądania) i parametry (JSON, do wczytania)."""
    table.drop(columns="params").to_csv(os.path.join(RESULTS_DIR, f"search_{name}.csv"), index=False)
    configs = list(table["params"])  # od najlepszej do najgorszej
    # default=float zamienia liczby numpy na zwykłe liczby, które JSON potrafi zapisać
    with open(os.path.join(RESULTS_DIR, f"search_{name}.json"), "w") as f:
        json.dump(configs, f, indent=2, default=float)


def load_configs(name):
    """Konfiguracje modelu ze strojenia, od najlepszej. [{}] dla modelu bez strojenia."""
    path = os.path.join(RESULTS_DIR, f"search_{name}.json")
    if not os.path.exists(path):
        return [{}]
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4. Porównanie modeli
# ---------------------------------------------------------------------------

def compare_models(df, y, folds, configs):
    """Kilka modeli na tych samych foldach.

    configs: słownik {nazwa: (funkcja fit_predict, parametry)}.
    Zwraca:
    - per_fold: metryki każdego modelu w każdym foldzie (kolumna 'model'),
    - by_source: metryki każdego modelu osobno dla każdego źródła danych,
    - oof: słownik {nazwa: prawdopodobieństwa out-of-fold}.
    """
    per_fold, by_source, oof = [], [], {}
    for name, (fit_predict, params) in configs.items():
        fold_table, proba = cross_validate(fit_predict, df, y, folds, params)
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
    """Średnia i odchylenie standardowe każdej metryki dla każdego modelu (jeden wiersz na model)."""
    metrics = [c for c in per_fold.columns if c not in ("model", "fold", "seconds")]
    grouped = per_fold.groupby("model", sort=False)
    table = grouped[metrics].mean().join(grouped[metrics].std().add_suffix("_std"))
    table["seconds"] = grouped["seconds"].sum()
    return table.reset_index()


# ---------------------------------------------------------------------------
# 5. Ocena na testach
# ---------------------------------------------------------------------------

def metrics_by_group(y_true, y_pred, groups):
    """Metryki osobno dla każdej wartości grupy (np. każdego źródła albo roku)."""
    rows = {}
    for value in sorted(pd.unique(groups)):
        mask = np.asarray(groups == value)
        rows[value] = compute_metrics(y_true[mask], y_pred[mask])
        rows[value]["n"] = int(mask.sum())
    return pd.DataFrame(rows).T


def bootstrap_ci(y_true, y_pred, n_iterations=1000, random_state=42):
    """95% przedział ufności bootstrap dla każdej metryki.

    1000 razy losujemy wiersze testu ze zwracaniem i liczymy metryki;
    przedział to 2,5. i 97,5. percentyl otrzymanych wartości.
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
    """Macierz pomyłek (wiersze = prawdziwa klasa, kolumny = przewidziana)."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
