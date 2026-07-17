import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from .domain import ForecastRequest, PassengerCountEvent
from .forecast import ForecastInput, forecast_occupancy
from .quality import assess_event
from .store import SQLiteEventStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.event_store = SQLiteEventStore(
        os.getenv("TRACKBUS_DATABASE_PATH", "data/trackbus.sqlite3")
    )
    yield
    app.state.event_store.close()


app = FastAPI(
    title="TrackBus Analytics API",
    version="0.2.0",
    description="Anonymous passenger-count ingestion and occupancy forecasting.",
    lifespan=lifespan,
)


def get_store(request: Request) -> SQLiteEventStore:
    return request.app.state.event_store


Store = Annotated[SQLiteEventStore, Depends(get_store)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trackbus-analytics-api", "version": "0.2.0"}


@app.post("/v1/events/passenger-counts", status_code=status.HTTP_202_ACCEPTED)
def accept_passenger_count(event: PassengerCountEvent, store: Store) -> dict[str, object]:
    previous = store.latest_for_bus(event.bus_id)
    issues = [asdict(issue) for issue in assess_event(event, previous)]
    result = store.append(event, issues)
    return {
        "accepted": result.accepted,
        "duplicate": result.duplicate,
        "eventId": event.event_id,
        "qualityIssues": issues,
    }


@app.get("/v1/events/passenger-counts")
def list_passenger_counts(
    store: Store,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    bus_id: Annotated[str | None, Query(alias="busId")] = None,
    route_id: Annotated[str | None, Query(alias="routeId")] = None,
) -> dict[str, object]:
    events = store.list_recent(limit=limit, bus_id=bus_id, route_id=route_id)
    return {"events": events, "count": len(events)}


@app.get("/v1/buses/{bus_id}/occupancy")
def current_bus_occupancy(bus_id: str, store: Store) -> dict[str, object]:
    event = store.latest_for_bus(bus_id)
    if event is None:
        raise HTTPException(status_code=404, detail="No occupancy event found for this bus")
    return {
        "busId": event.bus_id,
        "routeId": event.route_id,
        "observedAt": event.observed_at,
        "occupancy": event.occupancy,
        "capacity": event.capacity,
        "occupancyPercent": round(event.occupancy / event.capacity * 100),
        "source": event.source,
        "qualityScore": event.quality_score,
    }


@app.post("/v1/forecasts/occupancy")
def occupancy_forecast(data: ForecastRequest) -> dict[str, int | float]:
    result = forecast_occupancy(
        ForecastInput(
            recent_occupancy=data.recent_occupancy,
            capacity=data.capacity,
            hour=data.hour,
            day_type=data.day_type,
            weather=data.weather,
            event_nearby=data.event_nearby,
        )
    )
    return asdict(result)
