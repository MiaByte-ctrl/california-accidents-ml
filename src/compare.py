"""Porównanie najlepszych konfiguracji modeli na tych samych foldach CV."""
import os

import pandas as pd

from src.evaluate import confusion_table, cross_validate, metrics_by_group, summarize
from src.experiment import RESULTS_DIR
from src.models import MODELS

MODEL_ORDER = ["majority", "logreg", "random_forest", "neural_net", "lightgbm", "lightgbm_ordinal"]


def best_params(name):
    """Najlepsze parametry z pliku search_<name>.csv (pusty słownik, gdy brak strojenia)."""
    path = os.path.join(RESULTS_DIR, f"search_{name}.csv")
    if not os.path.exists(path):
        return {}
    best = pd.read_csv(path, dtype=str).iloc[0]
    params = {}
    for key, text in best.items():
        if key in ("macro_f1_mean", "macro_f1_std", "qwk_mean", "seconds"):
            continue
        params[key] = parse_value(key, text)
    return params


# Parametry, które modele przyjmują wyłącznie jako liczby całkowite
INTEGER_PARAMS = ["n_estimators", "num_leaves", "min_child_samples", "subsample_freq",
                  "max_depth", "min_samples_leaf", "hidden", "batch_size"]


def parse_value(key, text):
    """Zamienia tekst z CSV z powrotem na None / int / float / str."""
    if pd.isna(text) or text == "None":
        return None
    try:
        number = float(text)
    except ValueError:
        return text  # np. max_features="sqrt"
    if key in INTEGER_PARAMS:
        return int(number)
    return number


def compare_models(sample, y, folds, names=MODEL_ORDER):
    """Zwraca (tabela mean ± std, metryki per źródło, macierze pomyłek, predykcje OOF)."""
    summary_rows, by_source, confusions, oof = [], {}, {}, {}
    for name in names:
        params = best_params(name)
        fold_table, proba = cross_validate(MODELS[name], sample, y, folds, params)
        y_pred = proba.argmax(axis=1)
        stats = summarize(fold_table)
        row = {"model": name}
        for metric in stats.index:
            row[metric] = stats.loc[metric, "mean"]
            row[metric + "_std"] = stats.loc[metric, "std"]
        row["seconds"] = fold_table["seconds"].sum()
        summary_rows.append(row)
        by_source[name] = metrics_by_group(y, y_pred, sample["Source"].to_numpy())
        confusions[name] = confusion_table(y, y_pred)
        oof[name] = proba
        print(f"{name:18s} macro-F1 {row['macro_f1']:.4f} ± {row['macro_f1_std']:.4f}", flush=True)
    return pd.DataFrame(summary_rows), by_source, confusions, oof


def source_ablation(sample, y, folds, name):
    """Ten sam model bez informacji o źródle (Source ustawione na stałą)."""
    no_source = sample.assign(Source="unknown")
    fold_table, _ = cross_validate(MODELS[name], no_source, y, folds, best_params(name))
    return summarize(fold_table)
