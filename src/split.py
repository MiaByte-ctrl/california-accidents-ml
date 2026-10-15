"""Podział danych.

- test_out: zdarzenia od 2022-01-01 (test poza czasem treningu, mierzy dryf),
- test_in:  losowe ~15% zdarzeń z lat 2016–2021,
- train:    reszta 2016–2021; strojenie i wybór modelu tylko przez CV na train.

Losujemy całe zdarzenia (event_id), więc duplikaty tego samego wypadku
nigdy nie trafiają do dwóch różnych części.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import PROJECT_DIR

SPLIT_PATH = os.path.join(PROJECT_DIR, "data", "processed", "split_v2.csv")
OUT_OF_TIME_START = pd.Timestamp("2022-01-01")
TEST_IN_FRACTION = 0.15


def assign_split(df, random_state=42):
    """Zwraca Series z etykietą części ('train', 'test_in', 'test_out') dla każdego wiersza."""
    split = pd.Series("train", index=df.index)
    split[df["Start_Time"] >= OUT_OF_TIME_START] = "test_out"

    # Losujemy zdarzenia z okresu 2016–2021 do test_in, osobno w każdej
    # warstwie Severity × Source, żeby proporcje się zgadzały.
    rng = np.random.default_rng(random_state)
    in_time = df[split == "train"]
    events = in_time[["event_id", "Severity", "Source"]].drop_duplicates("event_id")
    test_events = []
    for _, group in events.groupby(["Severity", "Source"]):
        n = int(round(len(group) * TEST_IN_FRACTION))
        test_events.append(rng.choice(group["event_id"].to_numpy(), size=n, replace=False))
    test_events = np.concatenate(test_events)

    split[(split == "train") & df["event_id"].isin(test_events)] = "test_in"
    return split


def save_split(df, split):
    """Zapisuje przypisanie ID -> część, żeby każdy notebook używał tego samego podziału."""
    pd.DataFrame({"ID": df["ID"], "split": split}).to_csv(SPLIT_PATH, index=False)


def load_split(df):
    """Wczytuje zapisany podział i dopasowuje go do wierszy df po ID."""
    saved = pd.read_csv(SPLIT_PATH)
    return df["ID"].map(saved.set_index("ID")["split"])


def cv_folds(df, n_splits=5, random_state=42):
    """Foldy CV dla zbioru treningowego: grupy = zdarzenia, stratyfikacja po Severity × Source."""
    strata = df["Severity"].astype(str) + "_" + df["Source"].astype(str)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(df, strata, groups=df["event_id"]))
