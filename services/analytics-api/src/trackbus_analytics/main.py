from dataclasses import asdict

from fastapi import FastAPI, status

from .domain import PassengerCountEvent
from .forecast import ForecastInput, forecast_occupancy

app = FastAPI(title="TrackBus Analytics API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trackbus-analytics-api"}


@app.post("/v1/events/passenger-counts", status_code=status.HTTP_202_ACCEPTED)
def accept_passenger_count(event: PassengerCountEvent) -> dict[str, str | bool]:
    # Milestone 01 validates the public contract. A durable event store is the
    # next adapter; keeping it behind this boundary avoids coupling sensors to storage.
    return {"accepted": True, "eventId": event.event_id}


@app.post("/v1/forecasts/occupancy")
def occupancy_forecast(data: ForecastInput) -> dict[str, int | float]:
    return asdict(forecast_occupancy(data))
