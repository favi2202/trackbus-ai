# Passenger-count data contract

The canonical schema is `contracts/passenger-count-event.schema.json`.

Required fields identify the event, observation time, bus, route and stop;
capture boardings, alightings, current occupancy and capacity; and include the
source and a 0–1 quality score.

## Sensor adapter rule

Magnetic North or another vehicle system may use different field names. A small
adapter converts its payload to the canonical contract. The rest of TrackBus
must never depend directly on one vendor's private protocol.

## Data we do not require

- passenger name or account;
- face recognition or biometric identity;
- raw cabin video in the analytics pipeline;
- payment-card details.
