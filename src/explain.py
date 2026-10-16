"""Interpretacja modelu finalnego przez SHAP, z sumowaniem do cech oryginalnych (notebook 04)."""
import numpy as np
import pandas as pd
import shap

from src.data import CLASS_NAMES
from src.preprocessing import original_feature


def shap_values_by_feature(pipeline, df_sample):
    """SHAP dla Pipeline (preprocessing + LightGBM), zsumowane do cech oryginalnych.

    Wartości SHAP są addytywne, więc wartość cechy 'County' to suma wartości
    wszystkich jej kolumn one-hot (podobnie Hour_sin + Hour_cos = Hour).

    Zwraca słownik {nazwa klasy: DataFrame (wiersze x cechy oryginalne)}.
    """
    preprocessor = pipeline[0]
    model = pipeline[-1]
    X = preprocessor.transform(df_sample)
    encoded_names = preprocessor.get_feature_names_out()
    groups = [original_feature(name) for name in encoded_names]

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))  # (wiersze, kolumny, klasy)

    result = {}
    for k, class_name in enumerate(CLASS_NAMES):
        per_column = pd.DataFrame(values[:, :, k], columns=encoded_names)
        result[class_name] = per_column.T.groupby(groups).sum().T
    return result


def importance_table(shap_by_class):
    """Średnia |SHAP| każdej cechy oryginalnej, osobno dla klas, posortowana po średniej."""
    table = pd.DataFrame({name: values.abs().mean() for name, values in shap_by_class.items()})
    table["mean_all_classes"] = table.mean(axis=1)
    return table.sort_values("mean_all_classes", ascending=False)
