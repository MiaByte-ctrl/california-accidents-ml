"""Data: loading, cleaning, features and the train/test split (notebook 01).

What is here, in the order it is used:
1. File paths and the task definition (3 classes).
2. Loading and cleaning the California data.
3. Row-by-row features (time, weather, wind, road type).
4. The train / test_in / test_out split, the 10% sample and the cross-validation folds.
5. load_all() — everything above in one call.

Rule: this file does NOT compute any statistics from the data (medians, quantiles, modes).
Everything that depends on values in the data happens in the Pipeline (src/preprocessing.py),
which is fitted on the training data only.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import BOOL_COLS

# ---------------------------------------------------------------------------
# 1. Paths and task definition
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_DIR, "data", "raw", "US_Accidents_March23.csv")
CA_CSV_PATH = os.path.join(PROJECT_DIR, "data", "processed", "CA.csv")
CLEAN_PATH = os.path.join(PROJECT_DIR, "data", "processed", "ca_clean_v2.pkl")
SPLIT_PATH = os.path.join(PROJECT_DIR, "data", "processed", "split_v2.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "reports", "results")

# We predict 3 ordered classes (Severity 1 and 2 merged)
CLASS_NAMES = ["Sev1-2", "Sev3", "Sev4"]
CLASS_NAMES_4 = ["Sev1", "Sev2", "Sev3", "Sev4"]  # only for the reference result


def to_3_classes(severity):
    """Severity 1–4 -> 0 (1 and 2), 1 (3), 2 (4)."""
    return np.select([severity <= 2, severity == 3], [0, 1], default=2)


# ---------------------------------------------------------------------------
# 2. Loading and cleaning
# ---------------------------------------------------------------------------

# Columns excluded because they give away the answer (they describe the accident's
# consequences or are only known after it)
LEAKY_COLUMNS = [
    "End_Time",       # only known once the accident is over
    "End_Lat",        # extent of the impact; missing for Source2/3, so it reveals the source
    "End_Lng",
    "Distance(mi)",   # length of road affected = a consequence of the delay
    "Description",    # the text describes the outcome ("road closed", "stationary traffic")
]

# Columns with no information for the model: constant in California or almost always empty
USELESS_COLUMNS = [
    "State", "Country", "Timezone", "Turning_Loop",
    "Number", "Wind_Chill(F)", "Precipitation(in)",
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

# Fixed physical limits (not computed from the data, so they cannot leak)
CLIPPING_RULES = {
    "Temperature(F)": (-20, 120),
    "Humidity(%)": (0, 100),
    "Visibility(mi)": (0, 10),
    "Wind_Speed(mph)": (0, 100),
    "Pressure(in)": (20, 32),  # wide range, because mountain stations have lower pressure
}


def load_california():
    """Returns the raw California records.

    If data/processed/CA.csv (a ready-made extract) exists, it is read. Otherwise the raw file
    is filtered in chunks, so the whole of the US is never held in memory.
    """
    if os.path.exists(CA_CSV_PATH):
        return pd.read_csv(CA_CSV_PATH)

    parts = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=500_000):
        parts.append(chunk[chunk["State"] == "CA"])
    return pd.concat(parts, ignore_index=True)


def clean(df):
    """Row-by-row cleaning: types, dropped columns, fixed limits, event number."""
    df = df.copy()

    # The start time is needed for features and for the split; rows without it are dropped
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Event number: same time and place = the same accident recorded several times.
    # Rows with the same number must end up in the same part of the split.
    # dropna=False so that rows with a missing key value also get a number.
    df["event_id"] = df.groupby(["Start_Time", "Start_Lat", "Start_Lng"], dropna=False).ngroup()

    df = df.drop(columns=LEAKY_COLUMNS + USELESS_COLUMNS, errors="ignore")

    for col, (low, high) in CLIPPING_RULES.items():
        df[col] = df[col].clip(lower=low, upper=high)

    return df.reset_index(drop=True)


def load_clean(use_cache=True):
    """Reads the cleaned data from the intermediate file or creates it from scratch."""
    if use_cache and os.path.exists(CLEAN_PATH):
        return pd.read_pickle(CLEAN_PATH)
    df = clean(load_california())
    df.to_pickle(CLEAN_PATH)
    return df


# ---------------------------------------------------------------------------
# 3. Row-by-row features
# ---------------------------------------------------------------------------

# Weather groups in order of PRIORITY: the first match wins
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
    """Maps a detailed weather description to a few groups (the first match wins)."""
    text = weather.fillna("").str.lower()
    conditions = [text.str.contains(pattern) for _, pattern in WEATHER_GROUPS]
    names = [name for name, _ in WEATHER_GROUPS]
    # np.select takes the first condition in the list that is true
    result = pd.Series(np.select(conditions, names, default="Other"), index=weather.index)
    # A missing description stays missing; the imputer in the Pipeline fills it in
    result[weather.isna()] = np.nan
    return result


def classify_road(street):
    """Road type from the street name (the first match wins)."""
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
    """Adds features computed row by row (no statistics from the data)."""
    df = df.copy()
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    # Year is only used for the split and the drift analysis, it is not a model feature
    df["Year"] = df["Start_Time"].dt.year

    df["Weather_Aggregated"] = aggregate_weather(df["Weather_Condition"])
    df["Wind_Direction_Mapped"] = df["Wind_Direction"].map(WIND_DIRECTION_MAP)
    df["road_type"] = classify_road(df["Street"])

    for col in BOOL_COLS:
        df[col] = df[col].astype(float)
    return df


# ---------------------------------------------------------------------------
# 4. Split, sample and folds
# ---------------------------------------------------------------------------

OUT_OF_TIME_START = pd.Timestamp("2022-01-01")
TEST_IN_FRACTION = 0.15


def sample_events(df, fraction, random_state=42):
    """Draws a fraction of the events (event_id), separately in each Severity × Source stratum.

    Whole events are drawn, so duplicate records of one accident never end up in different
    parts. Drawing within strata keeps the class and source proportions.
    Returns an array of the drawn event numbers.
    """
    rng = np.random.default_rng(random_state)
    events = df[["event_id", "Severity", "Source"]].drop_duplicates("event_id")
    chosen = []
    for _, group in events.groupby(["Severity", "Source"]):
        n = int(round(len(group) * fraction))
        chosen.append(rng.choice(group["event_id"].to_numpy(), size=n, replace=False))
    return np.concatenate(chosen)


def assign_split(df, random_state=42):
    """Split label for every row: 'train', 'test_in' or 'test_out'.

    - test_out: events from 2022-01-01 on (checks how the model copes with the future),
    - test_in:  a random 15% of the events from 2016–2021,
    - train:    the rest of 2016–2021.
    """
    split = pd.Series("train", index=df.index)
    split[df["Start_Time"] >= OUT_OF_TIME_START] = "test_out"

    test_events = sample_events(df[split == "train"], TEST_IN_FRACTION, random_state)
    split[(split == "train") & df["event_id"].isin(test_events)] = "test_in"
    return split


def save_split(df, split):
    """Saves the ID -> part assignment, so that every notebook uses the same split."""
    pd.DataFrame({"ID": df["ID"], "split": split}).to_csv(SPLIT_PATH, index=False)


def load_split(df):
    """Reads the saved split and matches it to the rows of df by ID."""
    saved = pd.read_csv(SPLIT_PATH)
    return df["ID"].map(saved.set_index("ID")["split"])


def stratified_sample(df, frac=0.1, random_state=42):
    """A ~10% sample of the events for quick model comparison."""
    chosen = sample_events(df, frac, random_state)
    return df[df["event_id"].isin(chosen)].reset_index(drop=True)


def cv_folds(df, n_splits=5, random_state=42):
    """Cross-validation folds: an event is always in one fold, Severity × Source proportions kept."""
    strata = df["Severity"].astype(str) + "_" + df["Source"].astype(str)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(df, strata, groups=df["event_id"]))


# ---------------------------------------------------------------------------
# 5. Everything together
# ---------------------------------------------------------------------------

def load_all():
    """Cleaned data with features and a 'split' column.

    On the first run it creates the split and saves it to a file; after that it always
    reads it, so that every notebook uses the same split.
    """
    df = add_features(load_clean())
    if not os.path.exists(SPLIT_PATH):
        save_split(df, assign_split(df))
    df["split"] = load_split(df)
    return df


def prototype_data(n_splits=5):
    """A 10% sample of the events from the train part, 3-class labels and CV folds."""
    df = load_all()
    train = df[df["split"] == "train"]
    sample = stratified_sample(train, frac=0.1, random_state=42)
    y = to_3_classes(sample["Severity"].to_numpy())
    folds = cv_folds(sample, n_splits=n_splits)
    return sample, y, folds
