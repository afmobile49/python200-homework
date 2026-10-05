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
# 1. Rejecting the whole file when one temperature is null can be useful when
#    complete data is required, for example before training a model where missing
#    values could produce unreliable results. In a reporting pipeline, however,
#    I might prefer to tolerate a small gap and still process the rest of the day.
#    To allow that, I would change the schema to use list[float | None] for
#    temperature_2m and then handle None values later in the pipeline.
#
# 2. If the pipeline runs at noon, the current day only has about 12 hourly
#    observations, so min_hours=24 would treat that day as incomplete and drop it.
#    incomplete_days() helps by identifying which dates were dropped, so the
#    pipeline can report that the missing summary is caused by incomplete data.
#
# 3. Keeping weatherkit as a package makes it easier in Week 10 for another
#    pipeline module to import only the classes and functions it needs, such as
#    WeatherResponse, to_readings, and DailyAggregator, without running report.py.


# Without this guard, main() would run automatically whenever report.py
# was imported, causing the file loading and report printing as a side effect.
if __name__ == "__main__":
    main()