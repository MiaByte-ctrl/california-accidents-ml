"""Modele. Każdy ma tę samą postać: fit_predict(df_train, y_train, df_val, params) -> prawdopodobieństwa.

Wspólne zasady (CLAUDE.md):
- preprocessing dopasowany tylko na df_train (build_preprocessor w środku funkcji),
- każdy model dostaje wagi klas policzone na y_train tą samą funkcją class_weights_for(),
  a siłę ważenia (weight_power) stroimy w CV tak samo dla wszystkich modeli,
- random_state = 42.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.features import build_preprocessor

N_CLASSES = 3
SEED = 42


def class_weights_for(y, power):
    """Wagi wierszy: zbalansowane wagi klas podniesione do potęgi `power`.

    power = 1 -> pełne zbalansowanie (jak class_weight="balanced"),
    power = 0 -> brak ważenia, wartości pośrednie -> ważenie częściowe.
    """
    return compute_sample_weight("balanced", y) ** power


def split_params(params):
    """Oddziela weight_power od parametrów przekazywanych do samego modelu."""
    params = dict(params)
    power = params.pop("weight_power", 1.0)
    return power, params


def fit_predict_majority(df_train, y_train, df_val, params):
    """Punkt odniesienia: zawsze klasa najczęstsza."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(df_train, y_train)
    return model.predict_proba(df_val)


def fit_predict_logreg(df_train, y_train, df_val, params):
    """Punkt odniesienia: regresja logistyczna z wagami klas."""
    power, params = split_params(params)
    model = make_pipeline(
        build_preprocessor(scale_numeric=True),
        LogisticRegression(max_iter=2000, random_state=SEED, **params),
    )
    model.fit(df_train, y_train,
              logisticregression__sample_weight=class_weights_for(y_train, power))
    return model.predict_proba(df_val)


def fit_predict_random_forest(df_train, y_train, df_val, params):
    power, params = split_params(params)
    model = make_pipeline(
        build_preprocessor(scale_numeric=False),  # drzewa nie potrzebują skalowania
        RandomForestClassifier(n_jobs=-1, random_state=SEED, **params),
    )
    model.fit(df_train, y_train,
              randomforestclassifier__sample_weight=class_weights_for(y_train, power))
    return model.predict_proba(df_val)


def fit_predict_lightgbm(df_train, y_train, df_val, params):
    power, params = split_params(params)
    model = make_pipeline(
        build_preprocessor(scale_numeric=False),
        LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params),
    )
    model.fit(df_train, y_train,
              lgbmclassifier__sample_weight=class_weights_for(y_train, power))
    return model.predict_proba(df_val)


def fit_predict_lightgbm_ordinal(df_train, y_train, df_val, params):
    """Podejście porządkowe (Frank & Hall): dwa klasyfikatory binarne.

    Model A uczy się P(y > 0), czyli „Sev3 lub Sev4”; model B uczy się P(y > 1), czyli „Sev4”.
    Prawdopodobieństwa klas składamy z różnic tych dwóch wartości.
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
    proba = np.column_stack([1 - p_above_0, p_above_0 - p_above_1, p_above_1])
    return proba


def fit_predict_neural_net(df_train, y_train, df_val, params):
    """Sieć MLP w PyTorch (jak w notebooku 03), z ważoną funkcją straty i early stopping.

    Early stopping liczony na wewnętrznym wycinku treningu (10% zdarzeń),
    nigdy na foldzie walidacyjnym.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    power = params.get("weight_power", 1.0)
    hidden = params.get("hidden", 128)
    dropout = params.get("dropout", 0.3)
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 0.0)
    batch_size = params.get("batch_size", 1024)
    max_epochs = params.get("max_epochs", 30)

    preprocessor = build_preprocessor(scale_numeric=True)
    X_all = preprocessor.fit_transform(df_train).astype(np.float32)
    X_val = preprocessor.transform(df_val).astype(np.float32)

    # Wewnętrzny podział do early stopping (grupy = zdarzenia)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    fit_idx, stop_idx = next(inner.split(X_all, y_train, groups=df_train["event_id"]))

    X_fit = torch.tensor(X_all[fit_idx])
    y_fit = torch.tensor(y_train[fit_idx], dtype=torch.long)
    X_stop = torch.tensor(X_all[stop_idx])
    y_stop = torch.tensor(y_train[stop_idx], dtype=torch.long)

    # Te same wagi klas co w pozostałych modelach: zbalansowane ** weight_power
    counts = np.bincount(y_train[fit_idx], minlength=N_CLASSES)
    class_weights = (len(fit_idx) / (N_CLASSES * np.maximum(counts, 1))) ** power
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    model = nn.Sequential(
        nn.Linear(X_all.shape[1], hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, N_CLASSES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator().manual_seed(SEED)
    best_loss, best_state, patience_left = np.inf, None, 3
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(X_fit), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 2:  # BatchNorm potrzebuje co najmniej 2 wierszy
                continue
            optimizer.zero_grad()
            loss_fn(model(X_fit[batch]), y_fit[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            stop_loss = loss_fn(model(X_stop), y_stop).item()
        if stop_loss < best_loss:
            best_loss, patience_left = stop_loss, 3
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val))
    return torch.softmax(logits, dim=1).numpy()


MODELS = {
    "majority": fit_predict_majority,
    "logreg": fit_predict_logreg,
    "random_forest": fit_predict_random_forest,
    "neural_net": fit_predict_neural_net,
    "lightgbm": fit_predict_lightgbm,
    "lightgbm_ordinal": fit_predict_lightgbm_ordinal,
}
