"""Cechy i preprocessing.

Dwie części:
1. add_features(): przekształcenia wiersz po wierszu (bez statystyk ze zbioru),
   więc można je zrobić przed podziałem.
2. build_preprocessor(): imputacja, skalowanie, kodowanie. Zwraca ColumnTransformer,
   który wkładamy do Pipeline i dopasowujemy WYŁĄCZNIE na treningu.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Speed(mph)", "Start_Lat", "Start_Lng",
]
BOOL_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]
CYCLIC_COLS = {"Hour": 24, "DayOfWeek": 7, "Month": 12}
CATEGORICAL_COLS = [
    "Source",
    "County", "road_type", "Weather_Aggregated", "Wind_Direction_Mapped",
    "Sunrise_Sunset",
]
FEATURES = NUMERIC_COLS + BOOL_COLS + list(CYCLIC_COLS) + CATEGORICAL_COLS

# Grupy pogody w kolejności PRIORYTETU: pierwsze dopasowanie wygrywa.
# (W starym notebooku 01 późniejsze reguły nadpisywały wcześniejsze.)
WEATHER_GROUPS = [
    ("Storm", "thunder|t-storm|storm|hail|tornado|squall"),
    ("Snow_Ice", "snow|wintry|freezing|ice|sleet"),
    ("Rain", "rain|drizzle|shower"),
    ("Fog_Haze", "fog|haze|mist|smoke|dust|sand"),
    ("Cloudy", "cloud|overcast"),
    ("Clear_or_Fair", "clear|fair"),
]

WIND_DIRECTION_MAP = {
    "Calm": "Calm", "CALM": "Calm",
    "Variable": "Variable", "VAR": "Variable",
    "North": "N", "N": "N", "NNE": "N", "NNW": "N",
    "East": "E", "E": "E", "ENE": "E", "ESE": "E",
    "South": "S", "S": "S", "SSE": "S", "SSW": "S",
    "West": "W", "W": "W", "WSW": "W", "WNW": "W",
    "NE": "NE", "SE": "SE", "SW": "SW", "NW": "NW",
}


def aggregate_weather(weather):
    """Mapuje szczegółowy opis pogody na kilka grup (pierwsze dopasowanie wygrywa)."""
    text = weather.fillna("").str.lower()
    conditions = [text.str.contains(pattern) for _, pattern in WEATHER_GROUPS]
    names = [name for name, _ in WEATHER_GROUPS]
    result = pd.Series(np.select(conditions, names, default="Other"), index=weather.index)
    # Brak opisu zostawiamy jako brak (NaN); uzupełni go imputer w Pipeline
    result[weather.isna()] = np.nan
    return result


def classify_road(street):
    """Typ drogi z nazwy ulicy (pierwsze dopasowanie wygrywa)."""
    upper = street.fillna("").str.upper()
    conditions = [
        upper.str.match(r"^I-\d+"),
        upper.str.match(r"^US-\d+"),
        upper.str.match(r"^CA-\d+"),
        upper.str.contains("HWY|FWY|EXPY|HIGHWAY|FREEWAY|EXPRESSWAY"),
    ]
    names = ["Interstate", "US_Highway", "State_Route", "Major_Arterial"]
    return np.select(conditions, names, default="Local_Road")


def add_features(df):
    """Cechy wyliczane wiersz po wierszu. Nie używa statystyk z danych."""
    df = df.copy()
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    # Year służy tylko do podziału i analizy dryfu, nie jest cechą modelu
    df["Year"] = df["Start_Time"].dt.year

    df["Weather_Aggregated"] = aggregate_weather(df["Weather_Condition"])
    df["Wind_Direction_Mapped"] = df["Wind_Direction"].map(WIND_DIRECTION_MAP)
    df["road_type"] = classify_road(df["Street"])

    for col in BOOL_COLS:
        df[col] = df[col].astype(float)
    return df


def cyclic_encode(X):
    """Zamienia Hour/DayOfWeek/Month na pary sin/cos (23:00 jest blisko 0:00)."""
    X = np.asarray(X, dtype=float)
    periods = np.array(list(CYCLIC_COLS.values()), dtype=float)
    angles = 2 * np.pi * X / periods
    return np.hstack([np.sin(angles), np.cos(angles)])


def cyclic_feature_names(transformer, input_features):
    """Nazwy kolumn zwracanych przez cyclic_encode (najpierw wszystkie sin, potem cos)."""
    return [f"{col}_sin" for col in CYCLIC_COLS] + [f"{col}_cos" for col in CYCLIC_COLS]


def original_feature(encoded_name):
    """Nazwa cechy oryginalnej dla kolumny po preprocessingu, np. 'cat__County_Orange' -> 'County'."""
    name = encoded_name.split("__", 1)[1]
    for col in list(CYCLIC_COLS) + CATEGORICAL_COLS:
        if name.startswith(col + "_"):
            return col
    return name


def build_preprocessor(scale_numeric=True):
    """ColumnTransformer: imputacja, skalowanie i kodowanie dopasowywane na treningu.

    scale_numeric=False przydaje się dla drzew, które skalowania nie potrzebują.
    """
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale_numeric:
        numeric_steps.append(StandardScaler())

    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        # min_frequency łączy bardzo rzadkie kategorie (np. małe hrabstwa) w jedną
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
