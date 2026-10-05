from pydantic import BaseModel, Field , model_validator

class HourlyBlock(BaseModel):
    """Represents hourly weather data stored in parallel lists."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def validate_equal_lengths(self):
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError("Hourly data lists must have the same length")

        return self

class WeatherResponse(BaseModel):
    """Represents the validated top-level weather API response."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock