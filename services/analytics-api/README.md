# Analytics API

This is the first production service boundary. Sensor vendors post the shared
passenger-count contract here; operator and passenger applications consume
occupancy forecasts through versioned endpoints.

Run locally:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn trackbus_analytics.main:app --reload
```

The baseline model is deliberately transparent. A trained model should replace
it only after offline evaluation demonstrates better error and calibration.

Passenger-count events are stored idempotently in SQLite for the pilot. Set
`TRACKBUS_DATABASE_PATH` to choose the database file. The storage adapter is
isolated so a later PostgreSQL/Timescale implementation does not change sensor
or application contracts.

Useful URLs after startup:

- `http://localhost:8000/docs` — interactive API documentation
- `http://localhost:8000/health` — service health
- `http://localhost:8000/v1/events/passenger-counts` — ingestion and queries
