"""Interpreting the final model with SHAP, summed back to the original features (notebook 04)."""
import numpy as np
import pandas as pd
import shap

from src.data import CLASS_NAMES
from src.preprocessing import original_feature


def shap_values_by_feature(pipeline, df_sample):
    """SHAP for a Pipeline (preprocessing + LightGBM), summed back to the original features.

    SHAP values are additive, so the value of the feature 'County' is the sum of the values
    of all its one-hot columns (likewise Hour_sin + Hour_cos = Hour).

    Returns a dictionary {class name: DataFrame (rows x original features)}.
    """
    preprocessor = pipeline[0]
    model = pipeline[-1]
    X = preprocessor.transform(df_sample)
    encoded_names = preprocessor.get_feature_names_out()
    groups = [original_feature(name) for name in encoded_names]

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))  # (rows, columns, classes)

    result = {}
    for k, class_name in enumerate(CLASS_NAMES):
        per_column = pd.DataFrame(values[:, :, k], columns=encoded_names)
        result[class_name] = per_column.T.groupby(groups).sum().T
    return result


def importance_table(shap_by_class):
    """Mean |SHAP| of each original feature, separately per class, sorted by the overall mean."""
    table = pd.DataFrame({name: values.abs().mean() for name, values in shap_by_class.items()})
    table["mean_all_classes"] = table.mean(axis=1)
    return table.sort_values("mean_all_classes", ascending=False)
