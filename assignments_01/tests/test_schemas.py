import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import HourlyBlock, WeatherResponse


def test_valid_weather_response():
    data_path = Path(__file__).parent.parent / "weather_raw.json"

    # A plain relative path is unreliable because pytest may be run
    # from a different working directory.
    with data_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    weather = WeatherResponse.model_validate(raw_data)

    assert len(weather.hourly.time) == 168


def test_invalid_latitude_raises():
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [20.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)

def test_mismatched_hourly_lengths_raise():
    data = {
        "time": ["00:00", "01:00", "02:00"],
        "temperature_2m": [20.0, 21.0],
        "precipitation": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(data)

def test_null_temperature_raises():
    data = {
        "time": ["00:00", "01:00"],
        "temperature_2m": [20.0, None],
        "precipitation": [0.0, 0.0],
    }

    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(data)