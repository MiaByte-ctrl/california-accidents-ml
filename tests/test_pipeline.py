"""Tests that guard against data leakage. Run with: python -m pytest tests"""
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier

from src.data import (LEAKY_COLUMNS, add_features, aggregate_weather, assign_split, clean,
                      cv_folds, to_3_classes)
from src.evaluation import compute_metrics, cross_validate_model
from src.models import class_weights_for, fit_predict_lightgbm_ordinal, fit_with_weights
from src.preprocessing import FEATURES


def make_fake_data(n=2000, seed=42):
    """A small artificial dataset with the same columns as the California data."""
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
    # Duplicate events: copy 100 rows with a new ID
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


def test_preprocessing_learns_only_from_training_data():
    df = prepared_data()
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] != "train"].copy()
    train.loc[train.index[:50], "Temperature(F)"] = np.nan
    # Change the test set on purpose: if preprocessing saw it, the median would shift
    test["Temperature(F)"] = 1000.0
    y_train = to_3_classes(train["Severity"].to_numpy())

    # fit_with_weights is the function every model in the project uses for training
    model = fit_with_weights(DummyClassifier(), train, y_train, power=1.0, scale_numeric=False)
    model.predict(test)  # predicting must not refit anything
    learned_median = model[0].named_transformers_["num"][0].statistics_[0]
    assert learned_median == train["Temperature(F)"].median()


def test_cross_validation_trains_only_on_the_training_fold():
    df = prepared_data()
    train = df[df["split"] == "train"].reset_index(drop=True)
    y = to_3_classes(train["Severity"].to_numpy())
    folds = cv_folds(train, n_splits=3)
    seen = []

    def recording_model(df_train, y_train, df_val, params):
        # Remembers which rows were used for training and returns dummy probabilities
        seen.append((set(df_train.index), set(df_val.index)))
        return np.full((len(df_val), 3), 1 / 3)

    cross_validate_model(recording_model, train, y, folds)
    for (train_rows, val_rows), (train_idx, val_idx) in zip(seen, folds):
        assert train_rows == set(train_idx)
        assert not train_rows & val_rows


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


def test_metrics_match_hand_calculation():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    m = compute_metrics(y_true, y_pred)
    # F1 per class: 2/3, 0.8 and 0 -> macro-F1 = 0.4889; recall: 2/3, 1, 0 -> mean 0.5556
    assert np.isclose(m["macro_f1"], (2 / 3 + 0.8 + 0) / 3)
    assert np.isclose(m["balanced_acc"], (2 / 3 + 1 + 0) / 3)
    assert np.isclose(m["recall_Sev4"], 0.0)


def test_class_missing_from_a_group_is_left_out():
    # No Sev3 in this group: its recall must be missing, not 0, and not lower macro-F1
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 0])
    m = compute_metrics(y_true, y_pred)
    assert np.isnan(m["recall_Sev3"])
    assert np.isclose(m["macro_f1"], (0.8 + 2 / 3) / 2)


def test_ordinal_model_returns_valid_probabilities():
    df = prepared_data()
    train, val = df.iloc[:1500], df.iloc[1500:]
    y_train = to_3_classes(train["Severity"].to_numpy())
    proba = fit_predict_lightgbm_ordinal(train, y_train, val, {"n_estimators": 20})
    assert proba.shape == (len(val), 3)
    assert (proba >= 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
