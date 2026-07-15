# TrackBus roadmap

The roadmap is evidence-first: improve and measure the doorway counter before
building operational integrations around it.

## v0.1 - recorded-video prototype

- [x] Pretrained person detection
- [x] ByteTrack temporary IDs
- [x] Configurable two-zone transition counting
- [x] Annotated video, CSV events, and JSON summary
- [x] Model-independent counter tests

## v0.1.1 - low-resolution diagnostics

- [x] Configurable inference image size
- [x] CLI precedence for model, confidence, and image size
- [x] ByteTrack profile for short detection gaps
- [x] Frame and tracking statistics in JSON summaries
- [x] CPU/GPU comparison guidance

## v0.1.2 - doorway calibration diagnostics

- [x] Optional normalized single detection ROI
- [x] Source-coordinate translation and clipping
- [x] Optional left, center, and right doorway lanes
- [x] Known static-structure exclusion polygons
- [x] Calibration overlay and excluded-box drawing
- [x] Gap, overlap, boundary, restart, wide-box, multi-lane, and likely-static
  diagnostics without speculative counting

## v0.2 - explicit multi-view detection and evaluation

- [x] Separate person-only `YOLO.predict` from tracking
- [x] Model-independent detection/tracking records and backend protocols
- [x] Named normalized inference views with source-coordinate translation
- [x] Exclusion filtering before fusion and tracking
- [x] Class-aware NMS with view provenance and suppression metadata
- [x] One version-isolated ByteTrack adapter updated once per source frame
- [x] Empty-frame tracker updates with no fallback to `YOLO.track`
- [x] Backward-compatible full-frame default and explicit legacy ROI warnings
- [x] Warnings for insufficient sole-view zone coverage and unsafe exclusions
- [x] Optional raw, fused, and tracked diagnostic CSV exports
- [x] Multi-view/source/fusion/trajectory visual debugging
- [x] Resumable manual IN/OUT event annotation
- [x] Direction-aware, one-to-one event evaluation
- [x] Clearly labelled aggregate-only count evaluation
- [x] Bounded reproducible experiment matrices and ranked leaderboard artifacts
- [x] Five ordinary YAML comparison presets
- [x] Complete and publish the limited test-video benchmark interpretation in
  the v0.2 experiment report

The benchmark item must compare detector coverage, event counts, false-event and
fragmentation indicators, and FPS. Matching `6 IN / 2 OUT` is not sufficient to
declare a winner. Event F1 requires manually labelled event frames.

## Next - representative validation and detector evidence

- Collect consented, privacy-reviewed videos from representative bus doors.
- Define a consistent annotation protocol for completed crossing frames.
- Measure direction-aware event precision, recall, F1, timing error, absolute
  count error per door cycle, ID fragmentation, and processing speed.
- Stratify evaluation by camera, crowding, occlusion, bags, children, lighting,
  vibration, door geometry, and simultaneous crossings.
- Calibrate views, zones, exclusions, tracker settings, and fusion thresholds on
  training/calibration footage, then evaluate on held-out buses and journeys.
- Establish acceptable failure behavior and operational accuracy requirements.

If representative evidence confirms that generic person detection is the main
bottleneck, build a properly licensed, privacy-reviewed dataset and evaluate a
custom overhead-head or overhead-person detector. Keep the conservative counter
unchanged while testing whether detector recall and separation improve.

## Later, only after the counter is validated

- Gate processing with a reliable physical door-state signal.
- Support live camera ingestion and robust local service supervision.
- Add explicit privacy retention, deletion, access-control, and audit policies.
- Define operational monitoring for camera obstruction, drift, and degraded
  confidence.

Backends, databases, web dashboards, cloud deployment, passenger-facing apps,
maps, forecasting, Yandex integration, and CAN-bus integration remain deferred.
They do not solve the current detector, tracking, or ground-truth limitations.
