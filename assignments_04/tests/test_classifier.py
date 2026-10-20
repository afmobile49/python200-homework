
"""Tests for the reusable WeatherClassifier component."""

import joblib
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from weather_model import WeatherClassifier, Prediction


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@pytest.fixture
def classifier(tmp_path):
    """Create and load a small classifier for testing."""

    data = [
        [20, 10, 0, 10],
        [22, 12, 0, 12],
        [24, 14, 0, 15],
        [2, -3, 10, 50],
        [4, -1, 12, 45],
        [5, 0, 8, 40],
    ]

    labels = [1, 1, 1, 0, 0, 0]

    frame = pd.DataFrame(data, columns=FEATURES)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=1)),
    ])

    pipeline.fit(frame, labels)

    model_path = tmp_path / "test_weather_model.pkl"
    joblib.dump(pipeline, model_path)

    return WeatherClassifier(model_path)



# Test 1: Predict expected labels for different weather conditions

@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 22,
                "temperature_2m_min": 12,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 12,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": 2,
                "temperature_2m_min": -3,
                "precipitation_sum": 10,
                "wind_speed_10m_max": 50,
            },
            "skip",
        ),
    ],
)
def test_predict_expected_label(classifier, day, expected_label):
    result = classifier.predict([day])

    assert len(result) == 1
    assert isinstance(result[0], Prediction)
    assert result[0].label == expected_label


# Test 2: Probability must be between 0 and 1.

def test_prediction_probability_range(classifier):
    day = {
        "temperature_2m_max": 22,
        "temperature_2m_min": 12,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 12,
    }

    result = classifier.predict([day])

    assert 0.0 <= result[0].probability <= 1.0


# Test 3: Three inputs must produce three predictions
# in the same order.

def test_predict_three_days_in_order(classifier):
    days = [
        {
            "temperature_2m_max": 20,
            "temperature_2m_min": 10,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 10,
        },
        {
            "temperature_2m_max": 2,
            "temperature_2m_min": -3,
            "precipitation_sum": 10,
            "wind_speed_10m_max": 50,
        },
        {
            "temperature_2m_max": 24,
            "temperature_2m_min": 14,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 15,
        },
    ]

    results = classifier.predict(days)

    assert len(results) == 3
    assert all(isinstance(item, Prediction) for item in results)
    assert [item.label for item in results] == [
        "good", "skip", "good"
    ]


# Test 4: Empty input must return an empty list.

def test_predict_empty_list(classifier):
    assert classifier.predict([]) == []


# Test 5: Missing features must raise ValueError.

def test_predict_missing_feature(classifier):
    incomplete_day = {
        "temperature_2m_max": 22,
        "temperature_2m_min": 12,
        "precipitation_sum": 0,
    }

    with pytest.raises(
        ValueError,
        match="wind_speed_10m_max",
    ):
        classifier.predict([incomplete_day])


# Test 6: A missing model file must raise FileNotFoundError.

def test_missing_model_file(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(
        FileNotFoundError,
        match="Weather model file not found",
    ):
        WeatherClassifier(missing_path)
