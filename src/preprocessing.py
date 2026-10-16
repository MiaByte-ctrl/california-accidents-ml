"""Preprocessing: co widzi model i jak przygotowujemy cechy (notebooki 02–04).

Co tu jest:
1. Listy cech, które trafiają do modelu.
2. build_preprocessor() — uzupełnianie braków, skalowanie i kodowanie kategorii.
   Zwraca obiekt, który dopasowujemy WYŁĄCZNIE na danych treningowych.
3. original_feature() — z nazwy kolumny po kodowaniu odczytuje cechę oryginalną (dla SHAP).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Cechy modelu
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Start_Lat", "Start_Lng",
]
BOOL_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]
CYCLIC_COLS = {"Hour": 24, "DayOfWeek": 7, "Month": 12}  # cecha: długość cyklu
CATEGORICAL_COLS = [
    "Source",  # dostawca danych — etykiety mocno od niego zależą
    "County", "road_type", "Weather_Aggregated", "Wind_Direction_Mapped",
    "Sunrise_Sunset",
]
FEATURES = NUMERIC_COLS + BOOL_COLS + list(CYCLIC_COLS) + CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def cyclic_encode(X):
    """Zamienia Hour/DayOfWeek/Month na pary sin/cos, żeby 23:00 było blisko 0:00."""
    X = np.asarray(X, dtype=float)
    periods = np.array(list(CYCLIC_COLS.values()), dtype=float)
    angles = 2 * np.pi * X / periods
    return np.hstack([np.sin(angles), np.cos(angles)])


def cyclic_feature_names(transformer, input_features):
    """Nazwy kolumn zwracanych przez cyclic_encode (najpierw wszystkie sin, potem cos)."""
    return [f"{col}_sin" for col in CYCLIC_COLS] + [f"{col}_cos" for col in CYCLIC_COLS]


def build_preprocessor(scale_numeric=True):
    """Przygotowanie cech, dopasowywane na treningu i stosowane do danych testowych.

    - liczby: braki uzupełniane medianą, potem (opcjonalnie) standaryzacja,
    - flagi 0/1: braki uzupełniane najczęstszą wartością,
    - godzina, dzień tygodnia, miesiąc: sin/cos,
    - kategorie: brak jako osobna kategoria "missing", potem kodowanie one-hot.

    scale_numeric=False dla drzew, które skalowania nie potrzebują.
    """
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale_numeric:
        numeric_steps.append(StandardScaler())

    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        # min_frequency łączy kategorie rzadsze niż 100 wierszy (np. małe hrabstwa) w jedną
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
# 3. Nazwy kolumn po kodowaniu -> cechy oryginalne
# ---------------------------------------------------------------------------

def original_feature(encoded_name):
    """Np. 'cat__County_Orange' -> 'County', 'cyc__Hour_sin' -> 'Hour'."""
    name = encoded_name.split("__", 1)[1]  # usuwa prefiks kroku, np. "cat__"
    for col in list(CYCLIC_COLS) + CATEGORICAL_COLS:
        if name.startswith(col + "_"):
            return col
    return name
