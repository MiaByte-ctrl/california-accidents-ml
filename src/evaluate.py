"""Metryki, walidacja krzyżowa i strojenie (jednakowe dla wszystkich modeli)."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import ParameterSampler

# Zadanie główne: 3 uporządkowane klasy
CLASS_NAMES = ["Sev1-2", "Sev3", "Sev4"]
CLASS_NAMES_4 = ["Sev1", "Sev2", "Sev3", "Sev4"]


def to_3_classes(severity):
    """Severity 1–4 -> 0 (1 i 2), 1 (3), 2 (4)."""
    return np.select([severity <= 2, severity == 3], [0, 1], default=2)


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Słownik metryk: macro-F1 (główna), balanced accuracy, QWK, recall per klasa."""
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


def cross_validate(fit_predict, df, y, folds, params=None):
    """Uruchamia model na każdym foldzie.

    fit_predict(df_train, y_train, df_val, params) musi zwrócić macierz
    prawdopodobieństw (n_val x liczba klas). Cały preprocessing dzieje się
    w środku fit_predict, więc widzi tylko dane treningowe danego foldu.

    Zwraca (tabela metryk per fold, prawdopodobieństwa out-of-fold).
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


def summarize(fold_table):
    """Średnia ± odchylenie standardowe metryk z foldów (jedna linijka tekstu na metrykę)."""
    metric_cols = [c for c in fold_table.columns if c not in ("fold", "seconds")]
    return pd.DataFrame({
        "mean": fold_table[metric_cols].mean(),
        "std": fold_table[metric_cols].std(),
    })


def random_search(fit_predict, param_distributions, df, y, folds, n_iter=8, random_state=42):
    """Proste losowe przeszukiwanie hiperparametrów po macro-F1 z CV.

    Odpowiednik RandomizedSearchCV, ale działa z każdą funkcją fit_predict
    (także z siecią w PyTorch), więc wszystkie modele stroimy tak samo.
    """
    rows = []
    sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
    for i, params in enumerate(sampler):
        fold_table, _ = cross_validate(fit_predict, df, y, folds, params)
        row = dict(params)
        row["macro_f1_mean"] = fold_table["macro_f1"].mean()
        row["macro_f1_std"] = fold_table["macro_f1"].std()
        row["qwk_mean"] = fold_table["qwk"].mean()
        row["seconds"] = fold_table["seconds"].sum()
        rows.append(row)
        print(f"  [{i + 1}/{n_iter}] macro-F1 {row['macro_f1_mean']:.4f}  ({row['seconds']:.0f}s)  {params}")
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def metrics_by_group(y_true, y_pred, groups):
    """Metryki osobno dla każdej wartości grupy (np. Source)."""
    rows = {}
    for value in sorted(pd.unique(groups)):
        mask = np.asarray(groups == value)
        rows[value] = compute_metrics(y_true[mask], y_pred[mask])
        rows[value]["n"] = int(mask.sum())
    return pd.DataFrame(rows).T


def bootstrap_ci(y_true, y_pred, n_iterations=1000, random_state=42):
    """95% przedział bootstrap dla każdej metryki (do oceny na zbiorze testowym)."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        samples.append(compute_metrics(y_true[idx], y_pred[idx]))
    samples = pd.DataFrame(samples)
    point = pd.Series(compute_metrics(y_true, y_pred))
    return pd.DataFrame({
        "value": point,
        "ci_low": samples.quantile(0.025),
        "ci_high": samples.quantile(0.975),
    })


def confusion_table(y_true, y_pred, class_names=CLASS_NAMES):
    """Macierz pomyłek jako DataFrame (wiersze = prawda, kolumny = predykcja)."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=class_names, columns=class_names)
