"""Preprocessing: what the model sees and how the features are prepared (notebooks 02–04).

What is here:
1. The lists of features that go into the model.
2. build_preprocessor() — filling in missing values, scaling and encoding categories.
   It returns an object that is fitted ONLY on the training data.
3. original_feature() — reads the original feature from an encoded column name (for SHAP).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Model features
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Start_Lat", "Start_Lng",
]
BOOL_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]
CYCLIC_COLS = {"Hour": 24, "DayOfWeek": 7, "Month": 12}  # feature: length of the cycle
CATEGORICAL_COLS = [
    "Source",  # data provider — the labels depend heavily on it
    "County", "road_type", "Weather_Aggregated", "Wind_Direction_Mapped",
    "Sunrise_Sunset",
]
FEATURES = NUMERIC_COLS + BOOL_COLS + list(CYCLIC_COLS) + CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def cyclic_encode(X):
    """Turns Hour/DayOfWeek/Month into sin/cos pairs, so that 23:00 is close to 0:00."""
    X = np.asarray(X, dtype=float)
    periods = np.array(list(CYCLIC_COLS.values()), dtype=float)
    angles = 2 * np.pi * X / periods
    return np.hstack([np.sin(angles), np.cos(angles)])


def cyclic_feature_names(transformer, input_features):
    """Names of the columns returned by cyclic_encode (all the sin columns first, then cos)."""
    return [f"{col}_sin" for col in CYCLIC_COLS] + [f"{col}_cos" for col in CYCLIC_COLS]


def build_preprocessor(scale_numeric=True):
    """Feature preparation, fitted on the training data and then applied to test data.

    - numbers: missing values filled with the median, then (optionally) standardised,
    - 0/1 flags: missing values filled with the most frequent value,
    - hour, day of week, month: sin/cos,
    - categories: a missing value becomes its own "missing" category, then one-hot encoding.

    scale_numeric=False for trees, which do not need scaling.
    """
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale_numeric:
        numeric_steps.append(StandardScaler())

    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        # min_frequency merges categories with fewer than 100 rows (e.g. small counties) into one
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                      sparse_output=False),
    )

    return ColumnTransformer([
        ("num", make_pipeline(*numeric_steps), NUMERIC_COLS),
        ("bool", SimpleImputer(strategy="most_frequent"), BOOL_COLS),
        ("cyc", FunctionTransformer(cyclic_encode, feature_names_out=cyclic_feature_names),
         list(CYCLIC_COLS)),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


# ---------------------------------------------------------------------------
# 3. Encoded column names -> original features
# ---------------------------------------------------------------------------

def original_feature(encoded_name):
    """E.g. 'cat__County_Orange' -> 'County', 'cyc__Hour_sin' -> 'Hour'."""
    name = encoded_name.split("__", 1)[1]  # removes the step prefix, e.g. "cat__"
    for col in list(CYCLIC_COLS) + CATEGORICAL_COLS:
        if name.startswith(col + "_"):
            return col
    return name
