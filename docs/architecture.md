# TrackBus v0.1.1 architecture

TrackBus deliberately uses a small sequential pipeline. It is easier to reason
about, test, and calibrate than a distributed system and matches the recorded
video scope of the first prototype.

## Modules

- `config.py` loads YAML into validated dataclasses and applies CLI overrides.
- `zones.py` scales normalized polygons and classifies an anchor point.
- `detector.py` loads YOLO and performs person-only inference at the configured
  image size.
- `tracker.py` enables Ultralytics' supported ByteTrack integration and converts
  results into `TrackedPerson` records.
- `counter.py` owns the model-independent per-ID transition state machine.
- `event_logger.py` writes confirmed events and the final summary.
- `video_processor.py` coordinates frames, accumulates detection/tracking
  diagnostics, and draws the output overlays.
- `main.py` validates CLI input and assembles the components.

## Counting state machine

Each active tracking ID has one stable side and a human-readable state:
`UNKNOWN`, `OUTSIDE`, `TRANSITIONING_IN`, `INSIDE`, or
`TRANSITIONING_OUT`.

The first confirmed zone establishes where the person started and never creates
an event. A different destination zone must be observed for
`minimum_zone_frames` consecutive frames. OUTSIDE to INSIDE creates one `IN`
event; INSIDE to OUTSIDE creates one `OUT` event. Repeated observations in the
same zone do not create events. Returning to the stable origin cancels an
incomplete movement.

State is removed after `stale_track_timeout` unobserved frames. This bounds memory
and reduces the damage from a tracker ID eventually being reused.

The TrackBus ByteTrack profile retains lost tracks for 60 frames. It can restore
an ID when a matching detection returns after a brief miss, while the counter's
longer 90-frame default keeps movement state available. Lost tracks do not
produce synthetic bounding boxes or zone observations.

## Run diagnostics

The JSON summary records tracked person observations per frame, frames with and
without observations, the maximum simultaneous observations, and unique tracking
IDs. It also records source dimensions, FPS, confidence, and inference image
size. These values support repeatable comparisons but are not ground-truth
precision, recall, or counting-accuracy measurements.

## Coordinate model

Zone polygons use normalized image coordinates from 0.0 to 1.0. At runtime they
are scaled to the decoded frame dimensions. The tracked box's bottom-center is
used because it generally approximates a person's location on the floor better
than the center of an upright bounding box.

## Boundaries

YOLO and ByteTrack are isolated from counting logic. Unit tests can therefore
exercise every transition with synthetic IDs and zone observations. The pipeline
stores no biometric template and makes no attempt to identify a person beyond
the temporary ID assigned within one processing run.
