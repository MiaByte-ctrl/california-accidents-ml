"""Modele i ich hiperparametry (notebooki 02–03).

Co tu jest:
1. Wagi klas — wspólne dla wszystkich modeli.
2. Modele. Każdy ma tę samą postać:
       fit_predict_<model>(df_train, y_train, df_val, params) -> prawdopodobieństwa klas
   Preprocessing jest dopasowywany w środku, tylko na df_train.
3. MODELS — słownik nazwa -> funkcja.
4. SEARCH_SPACES — wartości hiperparametrów, które losujemy przy strojeniu.
"""
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import build_preprocessor

N_CLASSES = 3
SEED = 42

# ---------------------------------------------------------------------------
# 1. Wagi klas
# ---------------------------------------------------------------------------


def class_weights_for(y, power):
    """Waga każdego wiersza: zbalansowana waga jego klasy podniesiona do potęgi `power`.

    power = 1 -> pełne zbalansowanie (rzadkie klasy liczą się tyle co częste),
    power = 0 -> brak ważenia, wartości pośrednie -> ważenie częściowe.
    """
    return compute_sample_weight("balanced", y) ** power


def split_params(params):
    """Oddziela weight_power (nasz parametr) od parametrów samego modelu."""
    params = dict(params)
    power = params.pop("weight_power", 1.0)
    return power, params


def fit_with_weights(estimator, df_train, y_train, power, scale_numeric):
    """Dopasowuje preprocessing i model na danych treningowych, z wagami klas.

    Zwraca Pipeline (preprocessing + model), którego można używać do predykcji
    na nowych danych i zapisać do pliku.
    """
    preprocessor = build_preprocessor(scale_numeric=scale_numeric)
    X_train = preprocessor.fit_transform(df_train)
    estimator.fit(X_train, y_train, sample_weight=class_weights_for(y_train, power))
    return make_pipeline(preprocessor, estimator)


# ---------------------------------------------------------------------------
# 2. Modele
# ---------------------------------------------------------------------------

def fit_predict_majority(df_train, y_train, df_val, params):
    """Punkt odniesienia: zawsze klasa najczęstsza."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(df_train, y_train)
    return model.predict_proba(df_val)


def fit_predict_logreg(df_train, y_train, df_val, params):
    """Punkt odniesienia: regresja logistyczna z wagami klas."""
    power, params = split_params(params)
    estimator = LogisticRegression(max_iter=2000, random_state=SEED, **params)
    model = fit_with_weights(estimator, df_train, y_train, power, scale_numeric=True)
    return model.predict_proba(df_val)


def fit_predict_random_forest(df_train, y_train, df_val, params):
    power, params = split_params(params)
    estimator = RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)
    # Drzewa nie potrzebują skalowania cech
    model = fit_with_weights(estimator, df_train, y_train, power, scale_numeric=False)
    return model.predict_proba(df_val)


def fit_lightgbm(df_train, y_train, params):
    """Dopasowany LightGBM (Pipeline). Używany też do modelu finalnego."""
    power, params = split_params(params)
    estimator = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    return fit_with_weights(estimator, df_train, y_train, power, scale_numeric=False)


def fit_predict_lightgbm(df_train, y_train, df_val, params):
    return fit_lightgbm(df_train, y_train, params).predict_proba(df_val)


def fit_predict_lightgbm_ordinal(df_train, y_train, df_val, params):
    """Podejście porządkowe (Frank & Hall): dwa klasyfikatory „tak/nie”.

    Model A uczy się P(y > 0), czyli „Sev3 lub Sev4”; model B uczy się P(y > 1), czyli „Sev4”.
    Prawdopodobieństwa trzech klas składamy z różnic tych dwóch wartości.
    """
    power, params = split_params(params)
    preprocessor = build_preprocessor(scale_numeric=False)
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)

    above = []
    for threshold in [0, 1]:
        target = (y_train > threshold).astype(int)
        model = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params)
        model.fit(X_train, target, sample_weight=class_weights_for(target, power))
        above.append(model.predict_proba(X_val)[:, 1])

    p_above_0 = above[0]
    p_above_1 = np.minimum(above[1], p_above_0)  # P(y>1) nie może być większe niż P(y>0)
    return np.column_stack([1 - p_above_0, p_above_0 - p_above_1, p_above_1])


# --- Sieć neuronowa (PyTorch) ---

NN_DEFAULTS = {
    "weight_power": 1.0,
    "hidden": 128,         # liczba neuronów w pierwszej warstwie ukrytej
    "dropout": 0.3,
    "lr": 1e-3,            # learning rate
    "weight_decay": 0.0,   # kara L2 na wagi
    "batch_size": 1024,
    "max_epochs": 30,
}
PATIENCE = 3  # ile epok bez poprawy czekamy przed zatrzymaniem


def build_network(n_inputs, hidden, dropout):
    """Sieć: dwie warstwy ukryte (hidden i hidden/2 neuronów) i warstwa wyjściowa na 3 klasy."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_inputs, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, N_CLASSES),
    )


def train_with_early_stopping(network, loss_fn, X_fit, y_fit, X_stop, y_stop, p):
    """Uczy sieć epokami; kończy, gdy strata na danych X_stop nie spada przez PATIENCE epok.

    Na końcu przywraca wagi z najlepszej epoki.
    """
    import torch

    optimizer = torch.optim.Adam(network.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    generator = torch.Generator().manual_seed(SEED)  # stała kolejność wierszy w epokach
    best_loss, best_state, patience_left = np.inf, None, PATIENCE

    for epoch in range(p["max_epochs"]):
        # Jedna epoka: przejście przez dane treningowe paczkami po batch_size wierszy
        network.train()
        order = torch.randperm(len(X_fit), generator=generator)
        for start in range(0, len(order), p["batch_size"]):
            batch = order[start:start + p["batch_size"]]
            if len(batch) < 2:  # BatchNorm potrzebuje co najmniej 2 wierszy
                continue
            optimizer.zero_grad()
            loss_fn(network(X_fit[batch]), y_fit[batch]).backward()
            optimizer.step()

        # Sprawdzenie straty na danych do early stopping
        network.eval()
        with torch.no_grad():
            stop_loss = loss_fn(network(X_stop), y_stop).item()
        if stop_loss < best_loss:
            best_loss, patience_left = stop_loss, PATIENCE
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    network.load_state_dict(best_state)
    return network


def fit_predict_neural_net(df_train, y_train, df_val, params):
    """Sieć MLP z ważoną funkcją straty i early stopping.

    Do early stopping używamy osobnych 10% zdarzeń z df_train — nigdy df_val,
    bo df_val służy do oceny modelu.
    """
    import torch  # ładowany tylko tutaj, bo jest duży i potrzebny tylko sieci

    torch.manual_seed(SEED)
    p = {**NN_DEFAULTS, **params}  # parametry domyślne nadpisane podanymi

    preprocessor = build_preprocessor(scale_numeric=True)
    X_all = preprocessor.fit_transform(df_train).astype(np.float32)
    X_val = preprocessor.transform(df_val).astype(np.float32)

    # Wydzielenie 10% zdarzeń do early stopping
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    fit_idx, stop_idx = next(inner.split(X_all, y_train, groups=df_train["event_id"]))

    # Te same wagi klas co w innych modelach: zbalansowane ** weight_power
    counts = np.bincount(y_train[fit_idx], minlength=N_CLASSES)
    class_weights = (len(fit_idx) / (N_CLASSES * np.maximum(counts, 1))) ** p["weight_power"]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    network = build_network(X_all.shape[1], p["hidden"], p["dropout"])
    network = train_with_early_stopping(
        network, loss_fn,
        torch.tensor(X_all[fit_idx]), torch.tensor(y_train[fit_idx], dtype=torch.long),
        torch.tensor(X_all[stop_idx]), torch.tensor(y_train[stop_idx], dtype=torch.long),
        p,
    )

    network.eval()
    with torch.no_grad():
        logits = network(torch.tensor(X_val))
    return torch.softmax(logits, dim=1).numpy()


# ---------------------------------------------------------------------------
# 3. Wszystkie modele
# ---------------------------------------------------------------------------

MODELS = {
    "majority": fit_predict_majority,
    "logreg": fit_predict_logreg,
    "random_forest": fit_predict_random_forest,
    "neural_net": fit_predict_neural_net,
    "lightgbm": fit_predict_lightgbm,
    "lightgbm_ordinal": fit_predict_lightgbm_ordinal,
}

# ---------------------------------------------------------------------------
# 4. Hiperparametry do strojenia (po 10 losowań na model)
# ---------------------------------------------------------------------------

# Siła ważenia klas: ta sama siatka dla każdego modelu, żeby wszystkie traktować jednakowo
WEIGHT_POWERS = [0.0, 0.25, 0.5, 0.75, 1.0]

LIGHTGBM_SPACE = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.8, 1.0],
    "subsample_freq": [1],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
    "weight_power": WEIGHT_POWERS,
}

SEARCH_SPACES = {
    "logreg": {
        "C": loguniform(1e-3, 10),  # losowane w skali logarytmicznej (0,001 … 10)
        "weight_power": WEIGHT_POWERS,
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3],
        "weight_power": WEIGHT_POWERS,
    },
    "neural_net": {
        "hidden": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-3, 3e-3],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [512, 1024],
        "weight_power": WEIGHT_POWERS,
    },
    "lightgbm": LIGHTGBM_SPACE,
    "lightgbm_ordinal": LIGHTGBM_SPACE,
}
