# Operations API

The operations endpoints answer the questions an operator needs during a pilot
without exposing camera images or passenger identities.

## System summary

`GET /v1/operations/summary?staleAfterMinutes=15`

Returns total stored events, observed buses and routes, flagged events, the
latest observation time, and the number of buses that have stopped reporting.
The status is `empty`, `operational`, or `degraded`.

## Route summary

`GET /v1/operations/routes`

Returns one row per route using the latest event from every bus. Rows include
active bus count, average occupancy percentage, latest observation time, and
the number of buses whose latest event has a quality finding.

These endpoints are read-only. Authentication and role policies belong at the
deployment gateway before a real operator deployment is opened to users.
