"""Wczytanie i deterministyczne czyszczenie danych dla Kalifornii.

Zasada: w tym pliku NIE liczymy żadnych statystyk ze zbioru (median, kwantyli, mód).
Wszystko, co zależy od danych, dzieje się później w sklearn.Pipeline, dopasowywanym
tylko na zbiorze treningowym.
"""
import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_DIR, "data", "raw", "US_Accidents_March23.csv")
CA_CSV_PATH = os.path.join(PROJECT_DIR, "data", "processed", "CA.csv")
CLEAN_PATH = os.path.join(PROJECT_DIR, "data", "processed", "ca_clean_v2.pkl")

# Kolumny wykluczone z powodu wycieku.
LEAKY_COLUMNS = [
    "End_Time",       # znany dopiero po zakończeniu zdarzenia
    "End_Lat",        # zasięg skutków; brak dla Source2/3, więc zdradza źródło
    "End_Lng",
    "Distance(mi)",   # długość odcinka objętego skutkami = skutek opóźnienia
    "Description",    # tekst opisuje skutek ("road closed", "stationary traffic")
]

# Kolumny bez informacji dla modelu: stałe w Kalifornii, identyfikatory
# albo prawie całkowicie puste.
USELESS_COLUMNS = [
    "State", "Country", "Timezone", "Turning_Loop",
    "Number", "Wind_Chill(F)", "Precipitation(in)",
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

# Stałe progi fizyczne (nie liczone z danych, więc nie powodują wycieku).
CLIPPING_RULES = {
    "Temperature(F)": (-20, 120),
    "Humidity(%)": (0, 100),
    "Visibility(mi)": (0, 10),
    "Wind_Speed(mph)": (0, 100),
    # Ciśnienie: luźny zakres, bo stacje górskie mają niższe ciśnienie
    "Pressure(in)": (20, 32),
}


def load_california():
    """Zwraca surowe rekordy z Kalifornii.

    Jeśli istnieje data/processed/CA.csv (wycinek Kalifornii zrobiony wcześniej),
    czytamy go. W przeciwnym razie filtrujemy plik surowy kawałkami, żeby nie
    trzymać w pamięci całych USA.
    """
    if os.path.exists(CA_CSV_PATH):
        return pd.read_csv(CA_CSV_PATH)

    parts = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=500_000):
        parts.append(chunk[chunk["State"] == "CA"])
    return pd.concat(parts, ignore_index=True)


def clean(df):
    """Deterministyczne czyszczenie: typy, wycieki, stałe progi, klucz zdarzenia."""
    df = df.copy()

    # Czas startu jest potrzebny do cech i do podziału czasowego
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Klucz zdarzenia: ten sam czas i miejsce = ten sam wypadek.
    # Wiersze z tym samym kluczem muszą trafić do tej samej części podziału.
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


def stratified_sample(df, frac=0.1, random_state=42):
    """Losowa próbka ~10% stratyfikowana po Severity i Source (do prototypowania).

    Losujemy całe zdarzenia (event_id), żeby duplikaty nie rozjechały się
    między częściami w późniejszej CV.
    """
    rng = np.random.default_rng(random_state)
    events = df[["event_id", "Severity", "Source"]].drop_duplicates("event_id")
    chosen = []
    for _, group in events.groupby(["Severity", "Source"]):
        n = int(round(len(group) * frac))
        chosen.append(rng.choice(group["event_id"].to_numpy(), size=n, replace=False))
    chosen = np.concatenate(chosen)
    return df[df["event_id"].isin(chosen)].reset_index(drop=True)
