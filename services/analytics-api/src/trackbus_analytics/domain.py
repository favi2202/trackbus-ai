from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PassengerCountEvent(BaseModel):
    event_id: str = Field(alias="eventId", min_length=1)
    observed_at: datetime = Field(alias="observedAt")
    bus_id: str = Field(alias="busId", min_length=1)
    route_id: str = Field(alias="routeId", min_length=1)
    stop_id: str = Field(alias="stopId", min_length=1)
    boardings: int = Field(ge=0)
    alightings: int = Field(ge=0)
    occupancy: int = Field(ge=0)
    capacity: int = Field(gt=0)
    source: Literal["apc-camera", "manual", "simulator"]
    quality_score: float = Field(alias="qualityScore", ge=0, le=1)

    @model_validator(mode="after")
    def occupancy_fits_capacity(self) -> "PassengerCountEvent":
        if self.occupancy > self.capacity:
            raise ValueError("occupancy cannot exceed capacity")
        return self

    model_config = {"populate_by_name": True}


class ForecastRequest(BaseModel):
    recent_occupancy: list[int] = Field(alias="recentOccupancy", min_length=1, max_length=288)
    capacity: int = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    day_type: Literal["weekday", "weekend"] = Field(default="weekday", alias="dayType")
    weather: Literal["clear", "rain"] = "clear"
    event_nearby: bool = Field(default=False, alias="eventNearby")

    model_config = {"populate_by_name": True}
