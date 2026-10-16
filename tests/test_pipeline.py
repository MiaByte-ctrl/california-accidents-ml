"""Testy chroniące przed wyciekiem danych. Uruchomienie: python -m pytest tests"""
import numpy as np
import pandas as pd

from src.data import (LEAKY_COLUMNS, add_features, aggregate_weather, assign_split, clean,
                      cv_folds, to_3_classes)
from src.models import class_weights_for
from src.preprocessing import FEATURES, build_preprocessor


def make_fake_data(n=2000, seed=42):
    """Mały sztuczny zbiór o tych samych kolumnach co dane z Kalifornii."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 7 * 365, n), unit="D")
    df = pd.DataFrame({
        "ID": [f"A-{i}" for i in range(n)],
        "Source": rng.choice(["Source1", "Source2"], n),
        "Severity": rng.choice([1, 2, 3, 4], n, p=[0.05, 0.7, 0.2, 0.05]),
        "Start_Time": start.astype(str),
        "End_Time": start.astype(str),
        "Start_Lat": rng.uniform(32, 42, n).round(3),
        "Start_Lng": rng.uniform(-124, -114, n).round(3),
        "End_Lat": np.nan, "End_Lng": np.nan,
        "Distance(mi)": rng.uniform(0, 2, n),
        "Description": "Accident on I-5",
        "Street": rng.choice(["I-5 N", "Main St", "CA-1", None], n),
        "County": rng.choice(["Los Angeles", "Orange"], n),
        "Temperature(F)": rng.normal(65, 10, n),
        "Humidity(%)": rng.uniform(10, 100, n),
        "Pressure(in)": rng.normal(29.9, 0.2, n),
        "Visibility(mi)": rng.uniform(0, 10, n),
        "Wind_Speed(mph)": rng.uniform(0, 20, n),
        "Wind_Direction": rng.choice(["N", "SW", "Calm", None], n),
        "Weather_Condition": rng.choice(["Clear", "Light Rain", None], n),
        "Sunrise_Sunset": rng.choice(["Day", "Night"], n),
    })
    for col in ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
                "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"]:
        df[col] = rng.random(n) < 0.1
    # Duplikaty zdarzeń: kopiujemy 100 wierszy z nowym ID
    dup = df.iloc[:100].copy()
    dup["ID"] = [f"D-{i}" for i in range(100)]
    return pd.concat([df, dup], ignore_index=True)


def prepared_data():
    df = add_features(clean(make_fake_data()))
    df["split"] = assign_split(df)
    return df


def test_leaky_columns_are_removed():
    df = clean(make_fake_data())
    for col in LEAKY_COLUMNS:
        assert col not in df.columns
        assert col not in FEATURES


def test_split_keeps_each_event_in_one_part():
    df = prepared_data()
    parts_per_event = df.groupby("event_id")["split"].nunique()
    assert parts_per_event.max() == 1


def test_out_of_time_test_starts_in_2022():
    df = prepared_data()
    assert (df.loc[df["split"] == "test_out", "Start_Time"] >= "2022-01-01").all()
    assert (df.loc[df["split"] != "test_out", "Start_Time"] < "2022-01-01").all()


def test_cv_folds_do_not_share_events():
    df = prepared_data()
    train = df[df["split"] == "train"].reset_index(drop=True)
    for train_idx, val_idx in cv_folds(train, n_splits=3):
        shared = set(train.loc[train_idx, "event_id"]) & set(train.loc[val_idx, "event_id"])
        assert not shared


def test_imputer_uses_only_training_data():
    df = prepared_data()
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] != "train"].copy()
    # Celowo zmieniamy test: gdyby imputer go widział, mediana by się przesunęła
    test["Temperature(F)"] = 1000.0
    train.loc[train.index[:50], "Temperature(F)"] = np.nan

    pre = build_preprocessor()
    pre.fit(train[FEATURES])
    learned_median = pre.named_transformers_["num"][0].statistics_[0]
    assert learned_median == train["Temperature(F)"].median()


def test_weather_priority_first_match_wins():
    weather = pd.Series(["Light Rain with Thunder", "Freezing Rain", "Fair / Windy", None])
    result = aggregate_weather(weather)
    assert list(result[:3]) == ["Storm", "Snow_Ice", "Clear_or_Fair"]
    assert pd.isna(result[3])


def test_class_weight_power_scales_between_none_and_balanced():
    y = np.array([0] * 90 + [1] * 10)
    assert np.allclose(class_weights_for(y, 0.0), 1.0)
    full = class_weights_for(y, 1.0)
    assert np.isclose(full[y == 1][0] / full[y == 0][0], 9.0)


def test_three_class_mapping():
    assert list(to_3_classes(np.array([1, 2, 3, 4]))) == [0, 0, 1, 2]
