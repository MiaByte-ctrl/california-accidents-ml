"""Dane: wczytanie, czyszczenie, cechy i podział (notebook 01).

Co tu jest, w kolejności użycia:
1. Ścieżki do plików i definicja zadania (3 klasy).
2. Wczytanie i czyszczenie danych z Kalifornii.
3. Cechy liczone wiersz po wierszu (czas, pogoda, wiatr, typ drogi).
4. Podział na train / test_in / test_out, próbka 10% i foldy walidacji krzyżowej.
5. load_all() — wszystko powyższe jednym wywołaniem.

Zasada: w tym pliku NIE liczymy żadnych statystyk ze zbioru (median, kwantyli, mód).
Wszystko, co zależy od wartości w danych, dzieje się w Pipeline (src/preprocessing.py),
dopasowywanym tylko na zbiorze treningowym.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import BOOL_COLS

# ---------------------------------------------------------------------------
# 1. Ścieżki i definicja zadania
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_DIR, "data", "raw", "US_Accidents_March23.csv")
CA_CSV_PATH = os.path.join(PROJECT_DIR, "data", "processed", "CA.csv")
CLEAN_PATH = os.path.join(PROJECT_DIR, "data", "processed", "ca_clean_v2.pkl")
SPLIT_PATH = os.path.join(PROJECT_DIR, "data", "processed", "split_v2.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "reports", "results")

# Przewidujemy 3 uporządkowane klasy (Severity 1 i 2 połączone)
CLASS_NAMES = ["Sev1-2", "Sev3", "Sev4"]
CLASS_NAMES_4 = ["Sev1", "Sev2", "Sev3", "Sev4"]  # tylko do wyniku pomocniczego


def to_3_classes(severity):
    """Severity 1–4 -> 0 (1 i 2), 1 (3), 2 (4)."""
    return np.select([severity <= 2, severity == 3], [0, 1], default=2)


# ---------------------------------------------------------------------------
# 2. Wczytanie i czyszczenie
# ---------------------------------------------------------------------------

# Kolumny wykluczone, bo zdradzają wynik (opisują skutek wypadku albo są znane po nim)
LEAKY_COLUMNS = [
    "End_Time",       # znany dopiero po zakończeniu zdarzenia
    "End_Lat",        # zasięg skutków; brak dla Source2/3, więc zdradza źródło
    "End_Lng",
    "Distance(mi)",   # długość odcinka objętego skutkami = skutek opóźnienia
    "Description",    # tekst opisuje skutek ("road closed", "stationary traffic")
]

# Kolumny bez informacji dla modelu: stałe w Kalifornii albo prawie całkowicie puste
USELESS_COLUMNS = [
    "State", "Country", "Timezone", "Turning_Loop",
    "Number", "Wind_Chill(F)", "Precipitation(in)",
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

# Stałe progi fizyczne (nie liczone z danych, więc nie powodują wycieku)
CLIPPING_RULES = {
    "Temperature(F)": (-20, 120),
    "Humidity(%)": (0, 100),
    "Visibility(mi)": (0, 10),
    "Wind_Speed(mph)": (0, 100),
    "Pressure(in)": (20, 32),  # luźny zakres, bo stacje w górach mają niższe ciśnienie
}


def load_california():
    """Zwraca surowe rekordy z Kalifornii.

    Jeśli istnieje data/processed/CA.csv (gotowy wycinek), czytamy go. W przeciwnym razie
    filtrujemy plik surowy kawałkami, żeby nie trzymać w pamięci całych USA.
    """
    if os.path.exists(CA_CSV_PATH):
        return pd.read_csv(CA_CSV_PATH)

    parts = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=500_000):
        parts.append(chunk[chunk["State"] == "CA"])
    return pd.concat(parts, ignore_index=True)


def clean(df):
    """Czyszczenie wiersz po wierszu: typy, usunięcie kolumn, stałe progi, numer zdarzenia."""
    df = df.copy()

    # Czas startu jest potrzebny do cech i do podziału; wiersze bez niego usuwamy
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Numer zdarzenia: ten sam czas i miejsce = ten sam wypadek zapisany kilka razy.
    # Wiersze z tym samym numerem muszą trafić do tej samej części podziału.
    # dropna=False, żeby wiersze z brakiem w kluczu też dostały numer.
    df["event_id"] = df.groupby(["Start_Time", "Start_Lat", "Start_Lng"], dropna=False).ngroup()

    df = df.drop(columns=LEAKY_COLUMNS + USELESS_COLUMNS, errors="ignore")

    for col, (low, high) in CLIPPING_RULES.items():
        df[col] = df[col].clip(lower=low, upper=high)

    return df.reset_index(drop=True)


def load_clean(use_cache=True):
    """Wczytuje oczyszczone dane z pliku pośredniego albo tworzy je od nowa."""
    if use_cache and os.path.exists(CLEAN_PATH):
        return pd.read_pickle(CLEAN_PATH)
    df = clean(load_california())
    df.to_pickle(CLEAN_PATH)
    return df


# ---------------------------------------------------------------------------
# 3. Cechy wiersz po wierszu
# ---------------------------------------------------------------------------

# Grupy pogody w kolejności PRIORYTETU: pierwsze dopasowanie wygrywa
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
    # np.select bierze pierwszy spełniony warunek z listy
    result = pd.Series(np.select(conditions, names, default="Other"), index=weather.index)
    # Brak opisu zostawiamy jako brak; uzupełni go imputer w Pipeline
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
    """Dodaje cechy liczone wiersz po wierszu (bez statystyk z danych)."""
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


# ---------------------------------------------------------------------------
# 4. Podział, próbka i foldy
# ---------------------------------------------------------------------------

OUT_OF_TIME_START = pd.Timestamp("2022-01-01")
TEST_IN_FRACTION = 0.15


def sample_events(df, fraction, random_state=42):
    """Losuje część zdarzeń (event_id), osobno w każdej warstwie Severity × Source.

    Losujemy całe zdarzenia, żeby duplikaty jednego wypadku nie rozjechały się
    między częściami. Losowanie w warstwach zachowuje proporcje klas i źródeł.
    Zwraca tablicę wylosowanych numerów zdarzeń.
    """
    rng = np.random.default_rng(random_state)
    events = df[["event_id", "Severity", "Source"]].drop_duplicates("event_id")
    chosen = []
    for _, group in events.groupby(["Severity", "Source"]):
        n = int(round(len(group) * fraction))
        chosen.append(rng.choice(group["event_id"].to_numpy(), size=n, replace=False))
    return np.concatenate(chosen)


def assign_split(df, random_state=42):
    """Etykieta części dla każdego wiersza: 'train', 'test_in' albo 'test_out'.

    - test_out: zdarzenia od 2022-01-01 (sprawdza, jak model radzi sobie w przyszłości),
    - test_in:  losowe 15% zdarzeń z lat 2016–2021,
    - train:    reszta lat 2016–2021.
    """
    split = pd.Series("train", index=df.index)
    split[df["Start_Time"] >= OUT_OF_TIME_START] = "test_out"

    test_events = sample_events(df[split == "train"], TEST_IN_FRACTION, random_state)
    split[(split == "train") & df["event_id"].isin(test_events)] = "test_in"
    return split


def save_split(df, split):
    """Zapisuje przypisanie ID -> część, żeby każdy notebook używał tego samego podziału."""
    pd.DataFrame({"ID": df["ID"], "split": split}).to_csv(SPLIT_PATH, index=False)


def load_split(df):
    """Wczytuje zapisany podział i dopasowuje go do wierszy df po ID."""
    saved = pd.read_csv(SPLIT_PATH)
    return df["ID"].map(saved.set_index("ID")["split"])


def stratified_sample(df, frac=0.1, random_state=42):
    """Próbka ~10% zdarzeń do szybkiego porównywania modeli."""
    chosen = sample_events(df, frac, random_state)
    return df[df["event_id"].isin(chosen)].reset_index(drop=True)


def cv_folds(df, n_splits=5, random_state=42):
    """Foldy walidacji krzyżowej: zdarzenie zawsze w jednym foldzie, proporcje Severity × Source zachowane."""
    strata = df["Severity"].astype(str) + "_" + df["Source"].astype(str)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(df, strata, groups=df["event_id"]))


# ---------------------------------------------------------------------------
# 5. Wszystko razem
# ---------------------------------------------------------------------------

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
    """Próbka 10% zdarzeń z części train, etykiety 3-klasowe i foldy CV."""
    df = load_all()
    train = df[df["split"] == "train"]
    sample = stratified_sample(train, frac=0.1, random_state=42)
    y = to_3_classes(sample["Severity"].to_numpy())
    folds = cv_folds(sample, n_splits=n_splits)
    return sample, y, folds
