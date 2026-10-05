from dataclasses import FrozenInstanceError, dataclass, field

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


# --- Classes ---
# Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = [] if readings is None else readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        # The empty check avoids dividing by zero when there are no readings.
        # Without it, sum(self.readings) / len(self.readings) would raise ZeroDivisionError.
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self):
        # Without __repr__, Python normally displays something like
        # <__main__.Thermometer object at 0x...>, which does not show the useful state
        # of the object and is therefore much less helpful while debugging.
        avg = self.average()
        avg_display = "None" if avg is None else f"{avg:.1f}"

        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={avg_display})"
        )


my_thermometer = Thermometer("Los Angeles")
my_thermometer.add(21.5)
my_thermometer.add(23.0)
my_thermometer.add(19.8)
my_thermometer.add(25.2)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())
print(my_thermometer)

second_thermometer = Thermometer("San Diego", [18.5, 20.0, 21.5])
print([my_thermometer, second_thermometer])


# Q3
class TemperatureAlert:
    # The threshold belongs on TemperatureAlert because it is configuration/state
    # for that alert. One configured alert can then be reused across twenty
    # thermometers without passing the same threshold to breaches() every time.
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


alert_1 = TemperatureAlert(22.0)
alert_2 = TemperatureAlert(24.0)

print("Above 22.0:", alert_1.breaches(my_thermometer))
print("Above 24.0:", alert_2.breaches(my_thermometer))


# --- Dataclasses, Type Hints, and Docstrings ---
# Q1
@dataclass(frozen=True)
class Station:
    """Represents a weather station and its geographic location/elevation."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station(
    "LA01",
    "Los Angeles Central",
    34.0522,
    -118.2437,
    89.0,
)

station_b = Station(
    "LA01",
    "Los Angeles Central",
    34.0522,
    -118.2437,
    89.0,
)

print(station_a == station_b)

# Dataclasses generate value-based __eq__, so two Station objects with the
# same field values compare equal. The original hand-written class would use
# object identity here, so two separate instances would compare unequal.


# Q2
try:
    station_a.name = "New Name"
except FrozenInstanceError as e:
    print("FrozenInstanceError:", e)

station_c = Station(
    "SD01",
    "San Diego Central",
    32.7157,
    -117.1611,
    19.0,
)

stations = {station_a, station_b, station_c}
print("Number of unique stations:", len(stations))

# frozen=True also makes this value-based dataclass hashable, so Station
# objects can be used in sets or as dictionary keys. Here the two identical
# stations collapse to one set entry.


# Q3
# When I first tried:
#     stations: list[Station] = []
# dataclasses raised:
# ValueError: mutable default <class 'list'> for field stations is not allowed:
# use default_factory
# A shared mutable list could accidentally be reused by every StationBatch
# instance, so default_factory=list creates a fresh list for each object.
@dataclass
class StationBatch:
    """Represents a collection of weather stations in one region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to this batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)


batch = StationBatch("Southern California")
batch.add(station_a)
batch.add(station_c)
print("Highest station:", batch.highest())


# --- Pydantic ---
# Q1
class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_sensor_values(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Invalid sensor reading combination")
        return self


valid_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c=24.5,
    humidity=45.0,
)
print(valid_reading)


# Q2
try:
    Reading(
        station_id="LA01",
        temperature_c=20.0,
        humidity=40.0,
    )
except ValidationError as e:
    print("Missing field error:")
    print(e)

try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=150.0,
        humidity=40.0,
    )
except ValidationError as e:
    print("Temperature error:")
    print(e)

try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=20.0,
        humidity="very humid",
    )
except ValidationError as e:
    print("Humidity error:")
    print(e)

converted_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c="21.5",
    humidity=40,
)

print(converted_reading)
print(type(converted_reading.temperature_c))
print(type(converted_reading.humidity))

# Pydantic accepts "21.5" because it can safely parse that string into the
# declared float type. It rejects "very humid" because that string cannot be
# converted into a numeric float value.


# Q3
try:
    Reading(
        station_id="LA",
        temperature_c="not a number",
        humidity=50.0,
    )
except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])

# Three errors are reported: station_id is too short, timestamp is missing,
# and temperature_c is not numeric. Seeing all errors at once is useful because
# the caller can fix every invalid field in one pass instead of one at a time.


# Q4
valid_sensor_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c=-20.0,
    humidity=0.0,
)
print(valid_sensor_reading)

try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=-50.0,
        humidity=0.0,
    )
except ValidationError as e:
    print("Sensor combination error:")
    print(e)

# Field constraints validate one field at a time. This rule depends on the
# relationship between humidity and temperature_c, so it needs a model-level
# validator that can inspect both fields together.


# --- pytest ---
# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# pytest.approx is appropriate for computed floating-point values because
# binary floating-point arithmetic can introduce tiny representation errors;
# an exact == comparison can therefore be unnecessarily brittle.


# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean of the values."""
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])


# pytest.raises(ValueError) alone checks only the exception type. match=
# also verifies the message, so a wrong ValueError raised for another reason
# would not accidentally satisfy the test.


# Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0], 1.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([-1.0, -3.0], -2.0),
        ([10.0, 20.0, 30.0, 40.0], 25.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)


# Pytest summary after the corrected run: 6 passed.
# Parametrization keeps one behavior in one test while still reporting each
# input case separately; it avoids four nearly identical test functions.


# Q4
# I deliberately changed the conversion formula from 9 / 5 to 9 / 4 and reran:
#     pytest warmup_01.py -v
# The failure included:
#     assert 257.0 == 212
# for the 100 C case. Pytest showing both the obtained value (257.0) and the
# expected value (212) immediately points to the calculation error, which is
# much more useful than only saying "assertion failed". I then restored 9 / 5.