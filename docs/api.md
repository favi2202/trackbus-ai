# TrackBus Analytics API

The Python service exposes OpenAPI at `/docs` and `/openapi.json`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service/version health |
| `POST` | `/v1/events/passenger-counts` | Validate and idempotently persist a canonical event |
| `GET` | `/v1/events/passenger-counts` | Query recent events by bus or route |
| `GET` | `/v1/buses/{busId}/occupancy` | Latest reconstructed occupancy |
| `POST` | `/v1/buses/{busId}/verified-empty` | Auditable operator reset ordered after device time |
| `GET` | `/v1/operations/summary` | Pilot fleet/data status |
| `GET` | `/v1/operations/source-health` | Freshness and findings by source family |
| `GET` | `/v1/operations/reconciliation` | APC/Vision versus payment gaps |
| `GET` | `/v1/operations/routes` | Latest route load summaries |
| `POST` | `/v1/forecasts/occupancy` | Transparent bounded pilot forecast |

`eventId` is the idempotency key. A retry returns `accepted: true`,
`persisted: true`, and `duplicate: true` without adding a row. Accepted events
retain both source `qualityFlags` and API `qualityIssues` for audit.

The hosted web gateway validates the same contract. It forwards only when
`TRACKBUS_ANALYTICS_API_URL` is configured; otherwise it returns a retryable
`503` with `persisted: false`. It never presents a no-op as successful storage.

Use the safe dry-run replay before sending sample events:

```bash
python tools/replay_passenger_events.py
python tools/replay_passenger_events.py --send --target http://localhost:8000
```
