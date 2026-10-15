"""Model finalny: CV na pełnym treningu, strojenie progów, trening i jednorazowa ocena na testach."""
import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.compare import parse_value
from src.evaluate import (bootstrap_ci, compute_metrics, confusion_table, cross_validate,
                          metrics_by_group, summarize, to_3_classes)
from src.experiment import RESULTS_DIR, load_all
from src.models import fit_predict_lightgbm
from src.split import cv_folds
from src.data import PROJECT_DIR

FINAL_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "lightgbm_final.joblib")


def top_configs(n=3):
    """n najlepszych konfiguracji LightGBM z przeszukiwania na próbce."""
    search = pd.read_csv(os.path.join(RESULTS_DIR, "search_lightgbm.csv"), dtype=str)
    configs = []
    for _, row in search.head(n).iterrows():
        params = {}
        for key, text in row.items():
            if key not in ("macro_f1_mean", "macro_f1_std", "qwk_mean", "seconds"):
                params[key] = parse_value(key, text)
        configs.append(params)
    return configs


def full_train_cv(train, y, configs, n_splits=5):
    """5-krotna grupowa CV na całym zbiorze train dla każdej konfiguracji.

    Zwraca (tabela wyników, prawdopodobieństwa OOF najlepszej konfiguracji, najlepsze parametry).
    """
    folds = cv_folds(train, n_splits=n_splits)
    rows, best = [], None
    for i, params in enumerate(configs):
        fold_table, oof = cross_validate(fit_predict_lightgbm, train, y, folds, params)
        stats = summarize(fold_table)
        row = dict(params)
        row["macro_f1_mean"] = stats.loc["macro_f1", "mean"]
        row["macro_f1_std"] = stats.loc["macro_f1", "std"]
        row["qwk_mean"] = stats.loc["qwk", "mean"]
        row["seconds"] = fold_table["seconds"].sum()
        rows.append(row)
        print(f"  config {i + 1}: macro-F1 {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} "
              f"({row['seconds']:.0f}s)", flush=True)
        if best is None or row["macro_f1_mean"] > best[0]:
            best = (row["macro_f1_mean"], oof, params, folds)
    return pd.DataFrame(rows), best[1], best[2], best[3]


def apply_scaling(proba, scaling):
    """Mnoży prawdopodobieństwa klas przez współczynniki i wybiera największe."""
    return (proba * np.asarray(scaling)).argmax(axis=1)


def tune_class_scaling(oof_proba, y, folds, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)):
    """Dobiera mnożniki dla Sev3 i Sev4 (Sev1-2 = 1), maksymalizując macro-F1 na OOF z treningu.

    Żeby oszacować, czy to strojenie nie jest tylko dopasowaniem do szumu, robimy też
    uczciwy test: mnożniki dobrane na 4 foldach oceniamy na piątym.
    Zwraca (najlepsze mnożniki, tabela kontrolna per fold).
    """
    def best_on(mask):
        best_score, best_scaling = -1, (1.0, 1.0, 1.0)
        for s3, s4 in itertools.product(grid, grid):
            scaling = (1.0, s3, s4)
            score = f1_score(y[mask], apply_scaling(oof_proba[mask], scaling), average="macro")
            if score > best_score:
                best_score, best_scaling = score, scaling
        return best_scaling

    all_rows = np.ones(len(y), dtype=bool)
    final_scaling = best_on(all_rows)

    check = []
    for fold, (_, val_idx) in enumerate(folds):
        held_out = np.zeros(len(y), dtype=bool)
        held_out[val_idx] = True
        scaling = best_on(~held_out)
        check.append({
            "fold": fold,
            "scaling": scaling,
            "macro_f1_argmax": f1_score(y[held_out], oof_proba[held_out].argmax(axis=1), average="macro"),
            "macro_f1_scaled": f1_score(y[held_out], apply_scaling(oof_proba[held_out], scaling), average="macro"),
        })
    return final_scaling, pd.DataFrame(check)


def fit_final_model(train, y, params):
    """Trenuje finalny Pipeline (preprocessing + LightGBM) na całym zbiorze train."""
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import make_pipeline

    from src.features import build_preprocessor
    from src.models import SEED, class_weights_for, split_params

    power, model_params = split_params(params)
    model = make_pipeline(
        build_preprocessor(scale_numeric=False),
        LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **model_params),
    )
    model.fit(train, y, lgbmclassifier__sample_weight=class_weights_for(y, power))
    return model


def evaluate_on_test(model, scaling, test, y_test):
    """Jednorazowa ocena na teście: metryki z bootstrapem, per źródło, macierz pomyłek."""
    y_pred = apply_scaling(model.predict_proba(test), scaling)
    return {
        "metrics": bootstrap_ci(y_test, y_pred),
        "by_source": metrics_by_group(y_test, y_pred, test["Source"].to_numpy()),
        "confusion": confusion_table(y_test, y_pred),
        "y_pred": y_pred,
    }
