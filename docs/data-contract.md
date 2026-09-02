# Canonical passenger-count contract

The authoritative JSON Schema is
`contracts/passenger-count-event.schema.json`. Version `1.0` is the boundary
between adapters, analytics, storage, and applications.

| Field | Meaning |
| --- | --- |
| `eventId` | Globally unique idempotency key retained across retries |
| `observedAt` | Timezone-aware source observation time |
| `source` | `apc`, `vision`, `payment`, `manual`, or `import` |
| `busId`, `routeId`, `stopId`, `doorId` | Operator-issued operational context |
| `boardings`, `alightings` | Anonymous event deltas |
| `occupancy`, `capacity` | Reconstructed onboard count and physical bound |
| `confidence` | Source confidence from 0 to 1 |
| `qualityFlags` | Explainable source-side limitations; never silently discarded |

Vendor-specific names and protocols belong in adapters. Domain logic must never
depend on a camera/APC vendor. `vision` uses temporary video-local track IDs, but
those IDs are not included in the transport event and never become passenger
identity.

The contract does not require passenger names, faces, biometric templates,
payment-card details, or raw cabin video.
