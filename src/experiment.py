"""Przygotowanie danych do eksperymentów (próbka 10% z części train + foldy CV)."""
import os

from src.data import PROJECT_DIR, load_clean, stratified_sample
from src.evaluate import to_3_classes
from src.features import add_features
from src.split import SPLIT_PATH, assign_split, cv_folds, load_split, save_split

RESULTS_DIR = os.path.join(PROJECT_DIR, "reports", "results")


def load_all():
    """Oczyszczone dane z cechami i kolumną 'split'.

    Przy pierwszym uruchomieniu tworzy podział i zapisuje go do pliku; później
    zawsze go wczytuje, żeby wszystkie notebooki używały tego samego podziału.
    """
    df = add_features(load_clean())
    if not os.path.exists(SPLIT_PATH):
        save_split(df, assign_split(df))
    df["split"] = load_split(df)
    return df


def prototype_data(n_splits=5):
    """Próbka 10% zdarzeń z części train, etykiety 3-klasowe i grupowe foldy CV."""
    df = load_all()
    train = df[df["split"] == "train"]
    sample = stratified_sample(train, frac=0.1, random_state=42)
    y = to_3_classes(sample["Severity"].to_numpy())
    folds = cv_folds(sample, n_splits=n_splits)
    return sample, y, folds
