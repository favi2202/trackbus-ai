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

The hosted web gateway exposes the matching path under `/api`:

| Method | Hosted path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/events/passenger-counts` | Authenticate and idempotently store a camera event in D1 |
| `GET` | `/api/v1/events/passenger-counts?limit=50` | Return recent stored JSON for the site and direct inspection |
| `GET` | `/api/v1/operations/pilot` | Return current buses, source health, reconciliation, and recent events |

POST requests require `Authorization: Bearer <TRACKBUS_INGEST_KEY>`. The edge
client reads the matching value from `TRACKBUS_API_KEY`. GET is intentionally
read-only and public because the payload contains anonymous counts and
operational IDs—not frames, faces, or persistent passenger identities.

Use the safe dry-run replay before sending sample events:

```bash
python tools/replay_passenger_events.py
python tools/replay_passenger_events.py --send --target http://localhost:8000
```
