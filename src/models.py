"""Models and their hyperparameters (notebooks 02–03).

What is here:
1. Class weights — shared by all models.
2. Models. Each has the same form:
       fit_predict_<model>(df_train, y_train, df_val, params) -> class probabilities
   Preprocessing is fitted inside, on df_train only.
3. MODELS — a dictionary name -> function.
4. SEARCH_SPACES — the hyperparameter values drawn during tuning.
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
# 1. Class weights
# ---------------------------------------------------------------------------


def class_weights_for(y, power):
    """Weight of each row: the balanced weight of its class raised to the power `power`.

    power = 1 -> fully balanced (rare classes count as much as common ones),
    power = 0 -> no weighting, values in between -> partial weighting.
    """
    return compute_sample_weight("balanced", y) ** power


def split_params(params):
    """Separates weight_power (our own parameter) from the parameters of the model itself."""
    params = dict(params)
    power = params.pop("weight_power", 1.0)
    return power, params


def fit_with_weights(estimator, df_train, y_train, power, scale_numeric):
    """Fits the preprocessing and the model on the training data, with class weights.

    Returns a Pipeline (preprocessing + model) that can be used to predict on new data
    and saved to a file.
    """
    preprocessor = build_preprocessor(scale_numeric=scale_numeric)
    X_train = preprocessor.fit_transform(df_train)
    estimator.fit(X_train, y_train, sample_weight=class_weights_for(y_train, power))
    return make_pipeline(preprocessor, estimator)


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------

def fit_predict_majority(df_train, y_train, df_val, params):
    """Baseline: always the most frequent class."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(df_train, y_train)
    return model.predict_proba(df_val)


def fit_predict_logreg(df_train, y_train, df_val, params):
    """Baseline: logistic regression with class weights."""
    power, params = split_params(params)
    estimator = LogisticRegression(max_iter=2000, random_state=SEED, **params)
    model = fit_with_weights(estimator, df_train, y_train, power, scale_numeric=True)
    return model.predict_proba(df_val)


def fit_predict_random_forest(df_train, y_train, df_val, params):
    power, params = split_params(params)
    estimator = RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)
    # Trees do not need feature scaling
    model = fit_with_weights(estimator, df_train, y_train, power, scale_numeric=False)
    return model.predict_proba(df_val)


def fit_lightgbm(df_train, y_train, params):
    """A fitted LightGBM (Pipeline). Also used for the final model."""
    power, params = split_params(params)
    estimator = LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    return fit_with_weights(estimator, df_train, y_train, power, scale_numeric=False)


def fit_predict_lightgbm(df_train, y_train, df_val, params):
    return fit_lightgbm(df_train, y_train, params).predict_proba(df_val)


def fit_predict_lightgbm_ordinal(df_train, y_train, df_val, params):
    """Ordinal approach (Frank & Hall): two yes/no classifiers.

    Model A learns P(y > 0), i.e. "Sev3 or Sev4"; model B learns P(y > 1), i.e. "Sev4".
    The probabilities of the three classes are built from the differences of these two values.
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
    p_above_1 = np.minimum(above[1], p_above_0)  # P(y>1) cannot be larger than P(y>0)
    return np.column_stack([1 - p_above_0, p_above_0 - p_above_1, p_above_1])


# --- Neural network (PyTorch) ---

NN_DEFAULTS = {
    "weight_power": 1.0,
    "hidden": 128,         # number of neurons in the first hidden layer
    "dropout": 0.3,
    "lr": 1e-3,            # learning rate
    "weight_decay": 0.0,   # L2 penalty on the weights
    "batch_size": 1024,
    "max_epochs": 30,
}
PATIENCE = 3  # how many epochs without improvement we wait before stopping


def build_network(n_inputs, hidden, dropout):
    """Network: two hidden layers (hidden and hidden/2 neurons) and an output layer for 3 classes."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_inputs, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, N_CLASSES),
    )


def train_with_early_stopping(network, loss_fn, X_fit, y_fit, X_stop, y_stop, p):
    """Trains the network epoch by epoch; stops when the loss on X_stop has not fallen for PATIENCE epochs.

    At the end it restores the weights from the best epoch.
    """
    import torch

    optimizer = torch.optim.Adam(network.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    generator = torch.Generator().manual_seed(SEED)  # fixed order of rows in the epochs
    best_loss, best_state, patience_left = np.inf, None, PATIENCE

    for epoch in range(p["max_epochs"]):
        # One epoch: a pass over the training data in batches of batch_size rows
        network.train()
        order = torch.randperm(len(X_fit), generator=generator)
        for start in range(0, len(order), p["batch_size"]):
            batch = order[start:start + p["batch_size"]]
            if len(batch) < 2:  # BatchNorm needs at least 2 rows
                continue
            optimizer.zero_grad()
            loss_fn(network(X_fit[batch]), y_fit[batch]).backward()
            optimizer.step()

        # Check the loss on the early-stopping data
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
    """MLP network with a weighted loss and early stopping.

    Early stopping uses a separate 10% of the events from df_train — never df_val,
    because df_val is used to evaluate the model.
    """
    import torch  # only loaded here, because it is large and only the network needs it

    torch.manual_seed(SEED)
    p = {**NN_DEFAULTS, **params}  # default parameters overridden by the ones passed in

    preprocessor = build_preprocessor(scale_numeric=True)
    X_all = preprocessor.fit_transform(df_train).astype(np.float32)
    X_val = preprocessor.transform(df_val).astype(np.float32)

    # Set aside 10% of the events for early stopping
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    fit_idx, stop_idx = next(inner.split(X_all, y_train, groups=df_train["event_id"]))

    # The same class weights as in the other models: balanced ** weight_power
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
# 3. All models
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
# 4. Hyperparameters to tune (10 draws per model)
# ---------------------------------------------------------------------------

# Strength of class weighting: the same grid for every model, so all are treated equally
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
        "C": loguniform(1e-3, 10),  # drawn on a log scale (0.001 … 10)
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
