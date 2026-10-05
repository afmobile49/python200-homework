import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 15.0, 1.0),
        HourlyReading("2026-04-09T00:00", 20.0, 0.5),
        HourlyReading("2026-04-09T01:00", 25.0, 1.5),
    ]

def test_grouping_produces_two_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temperature_max_and_min(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    first_day = summaries[0]

    assert first_day.temp_max == 15.0
    assert first_day.temp_min == 10.0


def test_precipitation_sum(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(1.0)


def test_incomplete_day_is_dropped_and_reported():
    readings = [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.0),
    ]

    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    assert summaries == []
    assert incomplete == ["2026-04-08"]


def test_lower_min_hours_keeps_same_day():
    readings = [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.0),
    ]

    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"

# Verification step: I temporarily changed DailySummary.temp_range() from
# subtraction to addition. test_temp_range failed for all three parametrized
# cases, confirming that the test catches that deliberate bug. I then restored
# the correct subtraction implementation.

@pytest.mark.parametrize(
    "temp_max, temp_min, expected_range",
    [
        (20.0, 10.0, 10.0),
        (5.0, -5.0, 10.0),
        (12.5, 12.5, 0.0),
    ],
)
def test_temp_range(temp_max, temp_min, expected_range):
    from weatherkit.summarize import DailySummary

    summary = DailySummary(
        date="2026-04-08",
        temp_max=temp_max,
        temp_min=temp_min,
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == pytest.approx(expected_range)