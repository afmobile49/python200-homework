from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Represents one hourly weather observation.

    Temperature is measured in degrees Celsius.
    Precipitation is measured in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly reading objects.

    Args:
        response: A validated WeatherResponse object.

    Returns:
        A list of HourlyReading objects in the original hourly order.
    """

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly reading objects.

    Args:
        response: A validated WeatherResponse object.

    Returns:
        A list of HourlyReading objects in the original hourly order.
    """

    readings = []

    for timestamp, temperature, precipitation in zip(
        response.hourly.time,
        response.hourly.temperature_2m,
        response.hourly.precipitation,
    ):
        reading = HourlyReading(
            timestamp=timestamp,
            temperature_c=temperature,
            precipitation_mm=precipitation,
        )
        readings.append(reading)

    return readings