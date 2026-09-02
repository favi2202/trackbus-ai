from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EventSource = Literal["apc", "vision", "payment", "manual", "import"]


class PassengerCountEvent(BaseModel):
    schema_version: Literal["1.0"] = Field(default="1.0", alias="schemaVersion")
    event_id: str = Field(alias="eventId", min_length=1)
    observed_at: datetime = Field(alias="observedAt")
    source: EventSource
    bus_id: str = Field(alias="busId", min_length=1)
    route_id: str = Field(alias="routeId", min_length=1)
    stop_id: str = Field(alias="stopId", min_length=1)
    door_id: str = Field(alias="doorId", min_length=1)
    boardings: int = Field(ge=0)
    alightings: int = Field(ge=0)
    occupancy: int = Field(ge=0)
    capacity: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    quality_flags: list[str] = Field(default_factory=list, alias="qualityFlags")

    @model_validator(mode="after")
    def validate_operating_bounds(self) -> "PassengerCountEvent":
        if self.observed_at.tzinfo is None:
            raise ValueError("observedAt must include a timezone")
        if self.occupancy > self.capacity:
            raise ValueError("occupancy cannot exceed capacity")
        if len(set(self.quality_flags)) != len(self.quality_flags):
            raise ValueError("qualityFlags must not contain duplicates")
        return self

    model_config = {"populate_by_name": True}


class VerifiedEmptyReset(BaseModel):
    route_id: str = Field(alias="routeId", min_length=1)
    stop_id: str = Field(alias="stopId", min_length=1)
    door_id: str = Field(alias="doorId", min_length=1)
    capacity: int = Field(gt=0)

    model_config = {"populate_by_name": True}


class ForecastRequest(BaseModel):
    recent_occupancy: list[int] = Field(alias="recentOccupancy", min_length=1, max_length=288)
    capacity: int = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    day_type: Literal["weekday", "weekend"] = Field(default="weekday", alias="dayType")
    weather: Literal["clear", "rain"] = "clear"
    event_nearby: bool = Field(default=False, alias="eventNearby")

    model_config = {"populate_by_name": True}
