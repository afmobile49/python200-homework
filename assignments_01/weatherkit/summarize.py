from dataclasses import dataclass

from weatherkit import HourlyReading

@dataclass
class DailySummary:
    """Represents a daily summary of hourly weather readings."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the maximum and minimum temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly readings into daily weather summaries."""

    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours


    def summarize(
        self,
        readings: list[HourlyReading],
    ) -> list[DailySummary]:
        """Group hourly readings into daily summaries."""

        grouped: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = []

            grouped[date].append(reading)

        summaries = []

        for date, daily_readings in grouped.items():
            if len(daily_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c
                for reading in daily_readings
            ]

            precipitation_values = [
                reading.precipitation_mm
                for reading in daily_readings
            ]

            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=sum(precipitation_values),
                hours_observed=len(daily_readings),
            )

            summaries.append(summary)

        return sorted(summaries, key=lambda summary: summary.date)

    def incomplete_days(
        self,
        readings: list[HourlyReading],
    ) -> list[str]:
        """Return dates that have fewer than the required number of hours."""

        counts: dict[str, int] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            counts[date] = counts.get(date, 0) + 1

        return sorted(
            date
            for date, count in counts.items()
            if count < self.min_hours
        )