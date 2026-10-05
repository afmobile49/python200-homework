from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, ValidationError , model_validator
import pytest

# --- Classes ---
# Q1

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = [] if readings is None else readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):

        avg = self.average()
        avg_display = None if avg is None else round(avg, 1)

        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average():.1f})"
        )

#Q3

class TemperatureAlert:
    def __init__(self , threshold=30.0):
        self.threshold = threshold

    def breaches(self,thermometer):

        #result = []
        #
        #for reading in thermometer.readings:
        #    if reading > self.threshold:
        #        result.append(reading)
        #
        #return result

        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]

#Q4
@dataclass(frozen=True)
class Station:
    #def __init__(self , station_id, name , latitude , longitude, elevation):
    #    self.station_id = station_id
    #    self.latitude = latitude
    #    self.longitude = longitude
    #    self.elevation = elevation
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

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

# --- pytest ---
# Q1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

def mean(values: list[float]) -> float:
    """Return the arithmetic mean of the values."""
    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

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


#-------------------------------------------------------------

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

alert_1 = TemperatureAlert(22.0)
alert_2 = TemperatureAlert(24.0)

print("Above 22.0:", alert_1.breaches(my_thermometer))
print("Above 24.0:", alert_2.breaches(my_thermometer))


station_a = Station(
    "LA01",
    "Los Angeles Central",
    34.0522,
    -118.2437,
    89.0
)

station_b = Station(
    "LA01",
    "Los Angeles Central",
    34.0522,
    -118.2437,
    89.0
)

print(station_a == station_b)


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
    19.0
)

stations = {station_a, station_b, station_c}

print("Number of unique stations:", len(stations))

batch = StationBatch("Southern California")

batch.add(station_a)
batch.add(station_c)

print("Highest station:", batch.highest())

valid_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c=24.5,
    humidity=45.0
)

print(valid_reading)


# Q2

try:
    Reading(
        station_id="LA01",
        temperature_c=20.0,
        humidity=40.0
    )
except ValidationError as e:
    print("Missing field error:")
    print(e)

try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=150.0,
        humidity=40.0
    )
except ValidationError as e:
    print("Temperature error:")
    print(e)


try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=20.0,
        humidity="very humid"
    )
except ValidationError as e:
    print("Humidity error:")
    print(e)

converted_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c="21.5",
    humidity=40
)

print(converted_reading)
print(type(converted_reading.temperature_c))
print(type(converted_reading.humidity))

# Q3

try:
    Reading(
        station_id="LA",
        temperature_c="not a number",
        humidity=50.0
    )
except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])


valid_sensor_reading = Reading(
    station_id="LA01",
    timestamp="2026-10-04T15:00:00",
    temperature_c=-20.0,
    humidity=0.0
)

print(valid_sensor_reading)

try:
    Reading(
        station_id="LA01",
        timestamp="2026-10-04T15:00:00",
        temperature_c=-50.0,
        humidity=0.0
    )
except ValidationError as e:
    print("Sensor combination error:")
    print(e)


pytest.raises(ValueError)
match="empty"
