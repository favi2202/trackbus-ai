import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from .domain import ForecastRequest, PassengerCountEvent, VerifiedEmptyReset
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
    version="0.4.0",
    description="Vendor-neutral anonymous passenger counts and transport analytics.",
    lifespan=lifespan,
)


def get_store(request: Request) -> SQLiteEventStore:
    return request.app.state.event_store


Store = Annotated[SQLiteEventStore, Depends(get_store)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trackbus-analytics-api", "version": "0.4.0"}


@app.post("/v1/events/passenger-counts", status_code=status.HTTP_202_ACCEPTED)
def accept_passenger_count(event: PassengerCountEvent, store: Store) -> dict[str, object]:
    previous = store.latest_for_bus(event.bus_id)
    issues = [asdict(issue) for issue in assess_event(event, previous)]
    result = store.append(event, issues)
    return {
        "accepted": result.accepted,
        "persisted": True,
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
        "confidence": event.confidence,
        "qualityFlags": event.quality_flags,
    }


@app.post("/v1/buses/{bus_id}/verified-empty", status_code=status.HTTP_202_ACCEPTED)
def verified_empty(bus_id: str, data: VerifiedEmptyReset, store: Store) -> dict[str, object]:
    previous = store.latest_for_bus(bus_id)
    observed_at = datetime.now(UTC)
    if previous is not None and observed_at <= previous.observed_at:
        observed_at = previous.observed_at.astimezone(UTC) + timedelta(microseconds=1)
    event = PassengerCountEvent(
        eventId=f"manual-reset-{uuid4()}",
        observedAt=observed_at,
        source="manual",
        busId=bus_id,
        routeId=data.route_id,
        stopId=data.stop_id,
        doorId=data.door_id,
        boardings=0,
        alightings=0,
        occupancy=0,
        capacity=data.capacity,
        confidence=1.0,
        qualityFlags=["verified_empty_reset"],
    )
    result = store.append(event, [])
    return {
        "accepted": result.accepted,
        "persisted": True,
        "eventId": event.event_id,
        "observedAt": event.observed_at,
        "occupancy": 0,
    }


@app.get("/v1/operations/summary")
def operations_summary(
    store: Store,
    stale_after_minutes: Annotated[int, Query(alias="staleAfterMinutes", ge=1, le=1440)] = 15,
) -> dict[str, object]:
    return store.operational_summary(stale_after_minutes=stale_after_minutes)


@app.get("/v1/operations/source-health")
def source_health(
    store: Store,
    stale_after_minutes: Annotated[int, Query(alias="staleAfterMinutes", ge=1, le=1440)] = 15,
) -> dict[str, object]:
    sources = store.source_health(stale_after_minutes=stale_after_minutes)
    return {"sources": sources, "count": len(sources)}


@app.get("/v1/operations/reconciliation")
def reconciliation(store: Store) -> dict[str, object]:
    routes = store.reconciliation()
    return {"routes": routes, "count": len(routes)}


@app.get("/v1/operations/routes")
def operations_routes(store: Store) -> dict[str, object]:
    routes = store.route_summaries()
    return {"routes": routes, "count": len(routes)}


@app.post("/v1/forecasts/occupancy")
def occupancy_forecast(data: ForecastRequest) -> dict[str, int | float]:
    return asdict(
        forecast_occupancy(
            ForecastInput(
                recent_occupancy=data.recent_occupancy,
                capacity=data.capacity,
                hour=data.hour,
                day_type=data.day_type,
                weather=data.weather,
                event_nearby=data.event_nearby,
            )
        )
    )
