# TrackBus API

The Python analytics service exposes interactive OpenAPI documentation at
`/docs` and its machine-readable schema at `/openapi.json`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Deployment and container health |
| `POST` | `/v1/events/passenger-counts` | Idempotently accept a sensor event |
| `GET` | `/v1/events/passenger-counts` | Query recent events by bus or route |
| `GET` | `/v1/buses/{busId}/occupancy` | Read the latest accepted occupancy |
| `POST` | `/v1/forecasts/occupancy` | Run the transparent pilot baseline |

## Idempotency

`eventId` is the idempotency key. Replaying an already accepted event returns
`accepted: true` and `duplicate: true` without creating another database row.
This matters because buses can lose connectivity and retry buffered messages.

## Quality findings

Accepted responses can include `qualityIssues`. The first pilot checks:

- low sensor confidence;
- out-of-order timestamps;
- occupancy that does not match the previous count plus boardings minus
  alightings within a small tolerance.

Findings remain visible for audit and calibration. The platform does not
silently rewrite the original sensor value.

## Example ingestion

The sample file is an array, so use the provided replay tool:

```bash
python tools/replay_passenger_events.py
python tools/replay_passenger_events.py --send --target http://localhost:8000
```

The first command is a dry run. Events are transmitted only when `--send` is
explicitly provided.
