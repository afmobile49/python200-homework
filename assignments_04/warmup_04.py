
import json
from pathlib import Path


# Warmup Question 1: ModelInfo

class ModelInfo:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


def main():
    sample = {
        "model_name": "Weather AI",
        "features": [
            "temperature",
            "precipitation",
            "wind_speed"
        ]
    }

    json_path = Path(__file__).resolve().parent / "sample_model_info.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(sample, file, indent=4)

    model_info = ModelInfo(json_path)

    print("Number of features:", model_info.feature_count())

    # Loading JSON in __init__ reads the file only once.
    # Repeated feature_count() calls reuse the data in memory.


if __name__ == "__main__":
    main()


# Warmup Question 2: DayForecast Dataclass

from dataclasses import dataclass


@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain_status = "rain expected" if self.will_rain else "no rain expected"

        return f"{self.date}: high {self.high_c:.1f}C, {rain_status}"


forecast1 = DayForecast("2026-10-21", 24.1, True)
forecast2 = DayForecast("2026-10-22", 27.5, False)

print(forecast1.summary())
print(forecast2.summary())

# @dataclass automatically generates an __init__ method.
# We do not need to write the constructor manually.


# Warmup Question 3: Parsing Temperature and Pytest

import pytest


def parse_temperature(text: str) -> float:
    """Convert a temperature string to a float."""
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid temperature: {text}"
        ) from exc


def test_parse_temperature_valid():
    """Test valid temperature conversion."""
    result = parse_temperature("24.1")
    assert result == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    """Test invalid temperature input."""
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Using pytest.raises(ValueError) without match could pass
# even if an unrelated ValueError were raised.
# match checks that the error message refers to temperature.
