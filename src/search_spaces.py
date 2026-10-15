"""Przestrzenie hiperparametrów do losowego przeszukiwania (po 10 losowań na model).

weight_power (siła ważenia klas) ma tę samą siatkę we wszystkich modelach,
żeby każdy model miał tak samo staranne traktowanie niezbalansowania.
"""
from scipy.stats import loguniform

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
        "C": loguniform(1e-3, 10),
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
