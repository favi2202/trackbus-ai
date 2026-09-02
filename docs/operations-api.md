# Operations API

Operations endpoints expose pilot evidence without camera images or identity.

- `GET /v1/operations/summary?staleAfterMinutes=15` returns event/bus/route
  counts, latest observation, flagged events, stale buses, and overall status.
- `GET /v1/operations/source-health` groups freshness, coverage, and findings by
  canonical source family.
- `GET /v1/operations/reconciliation` compares sensor boardings (`apc` plus
  optional `vision`) with payment events by route. Missing sources are reported
  as `insufficient-evidence`; gaps are review tasks, not silently corrected.
- `GET /v1/operations/routes` uses the latest observation for each bus to expose
  active buses, average load, freshness, and flagged buses.

These endpoints are read-only. Production authentication, roles, and audit
policy belong at the deployment gateway.
