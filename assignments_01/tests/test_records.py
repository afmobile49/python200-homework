import json
from pathlib import Path

from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse


def load_weather_response() -> WeatherResponse:
    data_path = Path(__file__).parent.parent / "weather_raw.json"

    with data_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    return WeatherResponse.model_validate(raw_data)


def test_to_readings_preserves_count_and_order():
    weather = load_weather_response()

    readings = to_readings(weather)

    assert len(readings) == len(weather.hourly.time)
    assert readings[0].timestamp == weather.hourly.time[0]
    assert readings[-1].timestamp == weather.hourly.time[-1]


def test_to_readings_matches_values_by_index():
    weather = load_weather_response()

    readings = to_readings(weather)

    index = 10

    assert readings[index].timestamp == weather.hourly.time[index]
    assert readings[index].temperature_c == weather.hourly.temperature_2m[index]
    assert readings[index].precipitation_mm == weather.hourly.precipitation[index]


def test_identical_hourly_readings_are_equal():
    reading_a = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    reading_b = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    assert reading_a == reading_b