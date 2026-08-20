# 60-day route pilot runbook

## Stage 1 — install and calibrate

1. Obtain written approval for the route, bus, doorway, sources, and retention.
2. Assign stable route/bus/stop/door IDs and confirm physical capacity.
3. Prefer an approved APC feed; add Vision only where it provides a measurable gap.
4. Record camera/APC model, firmware, clock, timezone, geometry, and health signal.
5. Calibrate the Vision `OUTSIDE`, `DOOR`, and `INSIDE` polygons for the doorway.

## Stage 2 — collect and label

1. Keep original approved evidence read-only and access-controlled.
2. Label completed IN/OUT event frames, not persistent passenger identity.
3. Include daylight/night, children, bags, mobility aids, crowding, occlusion,
   vibration, camera drift, and simultaneous opposite crossings.
4. Separate calibration footage from held-out pilot journeys.

## Stage 3 — tune AI and API

1. Measure event precision, recall, F1, timing error, count error, fragmentation,
   processing FPS, delivery uptime, and retry recovery.
2. Validate the canonical adapter and idempotent API with offline replay.
3. Reconcile APC/Vision counts with payment and approved manual observations.
4. Quarantine impossible occupancy and expose source staleness; never rewrite
   events invisibly.
5. Keep operator recommendations advisory throughout the pilot.

## Stage 4 — pilot report

Report measured results by operating category, failure modes, uptime, hardware
and support cost per bus, and acceptance-criteria outcomes. Do not report a
generic detector benchmark, synthetic dashboard values, or a single aggregate
count as production accuracy or ROI.

If a source becomes stale, the UI must show its last update and stop calling it
live. A manual verified-empty reset must remain auditable and sort after the
latest device timestamp, even when a device clock is ahead.
