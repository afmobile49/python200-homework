import json
from pathlib import Path

from weatherkit import DailyAggregator, WeatherResponse, to_readings


def main() -> None:
    """Load, validate, summarize, and display the weather data."""
    data_path = Path(__file__).parent / "weather_raw.json"

    with data_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    weather = WeatherResponse.model_validate(raw_data)
    readings = to_readings(weather)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High':>8}"
        f"{'Low':>8}"
        f"{'Precip':>10}"
        f"{'Range':>10}"
    )

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>8.1f}"
            f"{summary.temp_min:>8.1f}"
            f"{summary.precipitation_sum:>10.1f}"
            f"{summary.temp_range():>10.1f}"
        )

    if incomplete:
        print("Warning: incomplete days dropped:", ", ".join(incomplete))


# Reflection
#
# 1. I used Pydantic for WeatherResponse because the JSON data comes from
#    outside the program and needs validation before I trust it.
#
# 2. I used dataclasses for HourlyReading and DailySummary because those
#    objects are used inside the program after the external data has already
#    been validated.
#
# 3. The most useful tests were the validation and aggregation tests because
#    they helped catch incorrect input and logic errors. When I intentionally
#    changed temp_range() from subtraction to addition, the tests immediately
#    showed the wrong calculated values.
#
# 4. One improvement I would make next is to add more validation for timestamps
#    and more edge-case tests for missing or incomplete weather data.


# Without this guard, main() would run automatically whenever report.py
# was imported, causing the file loading and report printing as a side effect.
if __name__ == "__main__":
    main()