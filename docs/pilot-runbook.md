# Route pilot runbook

This runbook turns the showcase into a controlled one-route field pilot.

## Before connecting a bus

1. Obtain written approval for the route, vehicle and data fields in scope.
2. Assign stable route, bus and stop identifiers.
3. Confirm vehicle capacity and the sensor's boarding/alighting semantics.
4. Verify that the integration sends anonymous counts rather than identities.
5. Record the device clock, timezone and firmware version.

## Integration rehearsal

1. Export at least one hour of approved sensor payloads.
2. Keep the original export read-only.
3. Implement a vendor adapter that produces the canonical JSON contract.
4. Run the replay tool without `--send` and resolve missing fields.
5. Start the local API and replay into an isolated pilot database.
6. Confirm duplicate events do not create duplicate rows.

## Field validation

For a small sample of stops, compare anonymous manual totals with sensor totals.
Track absolute count error, missed events, connectivity, freshness and the
frequency of quality findings. Do not promote forecasts until count quality is
stable enough to produce trustworthy occupancy history.

## Forecast acceptance gate

The trained model must outperform the transparent baseline on held-out days.
At minimum report mean absolute error, overload recall and interval calibration
by route, hour and day type. Keep a human operator responsible for dispatch
decisions during the pilot.

## Incident response

- If sensor data becomes stale, show the last update time and stop presenting
  it as live.
- If occupancy exceeds physical capacity, quarantine the event for review.
- If model confidence falls below the agreed threshold, fall back to historical
  schedules and show that fallback to operators.
- Keep an audit log of data corrections and dispatch recommendations.
