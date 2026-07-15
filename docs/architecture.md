# TrackBus v0.2.1 architecture

TrackBus v0.2.1 is a sequential, source-frame pipeline. It separates prediction
from tracking so several overlapping detector inputs can contribute to one
coherent tracking timeline. The design remains intentionally local and small:
there is no service, database, identity system, or distributed processing layer.

## Processing flow

```text
decoded source frame
  |
  +-> enabled inference view 1 -> YOLO.predict person detections --+
  +-> enabled inference view 2 -> YOLO.predict person detections --+ (optional)
  +-> enabled inference view N -> YOLO.predict person detections --+
                                                                   |
                         translate and clip every xyxy box <---------+
                                      |
                        source-coordinate raw detections
                                      |
                 bottom-center exclusion-polygon filtering
                                      |
                         class-aware greedy NMS fusion
                                      |
                          one fused detection set
                                      |
               one BYTETracker.update call for this source frame
                                      |
                  temporary IDs and tracked source-coordinate boxes
                                      |
       configurable anchor + stabilized zones + latched counter
                  + diagnostics + annotation
```

The same tracker update happens with an empty `N x 6` detection input when no
view finds a person. Thus every decoded source frame advances exactly one
ByteTrack instance exactly once. There is never a tracker per view.

## Model-independent records and interfaces

`Detection` represents one untracked prediction with an `xyxy` box, confidence,
class ID, source-view name, contributing views, and optional metadata.
`TrackedDetection` adds a temporary tracking ID while retaining confidence,
class, box, view provenance, and metadata. Both expose centre, bottom-centre,
and top-centre anchors.

Three small protocols separate responsibilities:

- `DetectorBackend.detect(image, source_view=...)` returns untracked detections;
- `DetectionFusion.fuse(detections)` merges source-coordinate predictions;
- `TrackerBackend.update(detections, frame)` advances one tracker.

The video processor depends on these protocols, so unit tests use fake backends
without loading a model, downloading weights, or importing a real tracker.

## Module responsibilities

- `config.py` loads validated YAML dataclasses, applies CLI overrides, resolves
  default or legacy views, and emits calibration/migration warnings.
- `detection.py` defines model-independent `Detection`, `TrackedDetection`, and
  bounding-box types.
- `interfaces.py` defines the detector, fusion, and tracker protocols.
- `detector.py` wraps person-only `YOLO.predict`, device selection, confidence,
  image size, trusted model input, and validated CUDA-only FP16 precision.
- `views.py` resolves normalized views into pixel crops, invokes the detector for
  each enabled view, translates boxes, and clips them to source boundaries.
- `calibration.py` resolves source-coordinate lanes, crossing corridor, and
  exclusions and applies the established bottom-centre exclusion anchor.
- `fusion.py` implements class-aware greedy NMS and preserves contributing-view
  and suppression metadata.
- `tracker.py` isolates the version-sensitive Ultralytics `BYTETracker` API and
  converts fused detections into tracked records.
- `zones.py` scales normalized counting polygons and records raw plus
  boundary-stabilized anchor classifications.
- `counter.py` owns the model-independent per-ID dwell, neutral-traversal,
  confirmation, latch, gap, and cooldown state machine.
- `diagnostics.py` observes gaps, overlaps, lanes, border contact, likely-static
  behavior, large boxes, and possible ID restarts without changing counts.
- `detection_export.py` optionally streams raw, fused, and tracked frame-level
  CSV diagnostics.
- `event_logger.py` streams confirmed crossing events and writes the run summary.
- `video_processor.py` decodes frames, executes stages in their required order,
  annotates the unchanged source frame, and encodes output.
- `annotate_events.py` provides human-authored, resumable event labels.
- `calibrate_camera.py` provides model-free, normalized camera-zone editing.
- `evaluation.py` performs direction-aware one-to-one event matching or clearly
  limited aggregate-only evaluation.
- `experiments.py` resolves bounded YAML matrices, runs the public processing
  command, and writes ranked comparison artifacts.
- `main.py` validates CLI input and assembles the production components.

## Detector boundary

`UltralyticsDetector` uses `YOLO.predict`, not a persistent Ultralytics tracker.
Each call requests class `0` only and preserves the confidence and class values
returned by the model. Empty `boxes` results become an empty list. Standard
Ultralytics weight names and trusted local paths are supported; explicit HTTP or
HTTPS model URLs are rejected.

Device resolution accepts CPU, CUDA, a CUDA index, or automatic selection. FP32
is the default. With pinned Ultralytics 8.4.95, requested FP16 uses the supported
`quantize=16` prediction form only on CUDA; this selects FP16 and does not enable
INT8/export quantization. CPU requests downgrade once to FP32. The deprecated
configuration key `half` is converted once at startup and never forwarded per
frame.

## Coordinate flow and views

An inference view stores normalized `[left, top, right, bottom]` bounds. Runtime
pixel bounds use floor for left/top and ceiling for right/bottom so fractional
edges retain coverage. Full-frame input is passed through; other views are
cropped from the original decoded frame.

Detector boxes are initially view-local. The view origin is added to every box,
then coordinates are clipped to `[0, width - 1] x [0, height - 1]`. From that
point onward, exclusions, NMS, ByteTrack, zones, lanes, diagnostics, trajectories,
and annotation all use source coordinates. The output video is never cropped.

If neither `camera.inference_views` nor the deprecated `camera.detection_roi` is
configured, the resolver creates one enabled `full` view. A legacy ROI becomes
one `legacy_detection_roi` view with a warning. Mixing the two configuration
styles is an error rather than an ambiguous reinterpretation.

## Exclusion filtering

Exclusions represent known static camera or door structure. Each translated raw
detection's bottom-center anchor is tested against every exclusion polygon. An
excluded detection remains available to raw diagnostic export and debug drawing,
but it never reaches fusion or ByteTrack:

```text
translated raw -> classify excluded -> fuse included only -> track
```

TrackBus does not infer static exclusions. Likely-static behavior remains a
diagnostic signal. Configuration loading warns when an exclusion overlaps a
counting zone, corridor, or configured passenger lane because such a polygon
can erase real passenger paths. Runtime pixel calibration rejects configured
lane or corridor overlap as a stronger safeguard.

## Detection fusion

`NmsDetectionFusion` groups detections by class and applies conservative greedy
source-space NMS across inference views. With the default
`confidence_strategy: maximum`, the highest-confidence box normally wins and a
candidate from another view whose IoU is at least the configured positive
threshold can be suppressed. A second NMS pass is never applied within one YOLO
view, at most one candidate per view joins a group, and different classes are
never fused.

The winner records contributing view names, source detection count, suppression
count, view bounds, suppressed boxes, method, threshold, and confidence
strategy. When `prefer_full_frame: true`, a full-frame candidate supplies the
coordinates while the confidence remains the maximum among contributors; this
is an explicit trade-off and is false by default.

NMS only merges detector hypotheses and cannot recover an undetected person. A
large box which spans two mutually distinct smaller hypotheses is treated as
ambiguous and discarded with explicit metadata so it cannot erase both. Without
those smaller hypotheses, fusion cannot split the large box. A threshold that
is too low can suppress adjacent passengers; one that is too high can send
duplicate boxes into ByteTrack. Synthetic tests cover identical and shifted
duplicates, same-view preservation, nearby people, one large box against two
smaller boxes, empty input, and frame boundaries.

## ByteTrack adapter boundary

`ByteTrackAdapter` owns one Ultralytics `BYTETracker`. It converts each fused
detection into an Ultralytics `Boxes` row:

```text
[left, top, right, bottom, confidence, class_id]
```

It then maps returned tracks back to the originating fused detection using the
source-detection index in the tracker result, preserving view provenance and
fusion metadata. Non-person detections, unexpected result shapes, invalid source
indices, and internal update failures raise a precise `TrackerAdapterError`.
There is deliberately no automatic fallback to `YOLO.track`.

The adapter is tested with Ultralytics `8.4.95`, which is pinned in project
dependencies. All internal imports and array-column assumptions live in this one
module. The effective tracker thresholds and installed Ultralytics version are
recorded in the run summary.

The legacy combined adapter remains available only for explicitly constructed
v0.1 callers and compatibility tests. The supported `python -m trackbus.main`
path always assembles prediction, fusion, and explicit tracking separately.

## Counting state machine

Each active tracking ID has a stable side and one of these states: `UNKNOWN`,
`OUTSIDE`, `TRANSITIONING_IN`, `INSIDE`, or `TRANSITIONING_OUT`.

The first confirmed zone establishes the origin and never creates an event. It
requires `minimum_origin_zone_frames` observations (falling back to the legacy
`minimum_zone_frames`). A crossing is eligible only after a raw neutral-region
sample. The destination must then remain confirmed for
`minimum_destination_zone_frames`. `OUTSIDE -> neutral -> INSIDE` creates `IN`;
`INSIDE -> neutral -> OUTSIDE` creates `OUT`.

Emission latches the destination as the new stable side. A reverse event needs
fresh stable-side dwell, departure, another neutral traversal, and opposite-side
confirmation. Direct side flips never rebase a track. A cooldown is a secondary
guard and can hold a fully confirmed reverse pending; it cannot replace the
spatial transition. Short unseen gaps preserve pending state, while a gap beyond
`maximum_transition_gap_frames` invalidates uncertain dwell and confirmation.
Track memory expires after `stale_track_timeout` unseen frames.

`zone_anchor` selects centre, bottom-centre, or top-centre. Optional normalized
boundary hysteresis turns shallow zone-edge hits into an effective neutral
classification for stability, but the counter separately retains the raw zone
and never accepts a synthetic hysteresis sample as the required neutral
traversal.

These rules are independent of model, view, and diagnostic settings. TrackBus
does not invent an event when a fragmented track lacks both sides of a complete
transition.

## Diagnostics and artifacts

The normal annotated output shows zones, tracked boxes and IDs, confidence,
counter state, lane, counts, occupancy, and capacity. Calibration debug adds view
rectangles, lanes, exclusions, and excluded boxes. Full visual debug adds faint
raw boxes, fused boxes with confidence and view contributions, track trajectories,
possible restart labels, and raw/fused frame totals.

The summary distinguishes raw detections, fused detections, and tracked
observations; records empty-frame counts, unique IDs, tracker updates, enabled
views, fusion settings, duration, FPS, selected anchor, hysteresis, stability
settings, and suppression-reason observation counts; and retains the v0.1
`person_detections_total` field with its tracked-observation semantics. Doorway
occupancy/overlap diagnostics require a lane or crossing corridor. Unavailable
metrics are `null` with a reason instead of an apparently successful zero.

Large diagnostic CSVs are created only when
`diagnostics.export_detection_csv: true`:

- raw detections include frame, timestamp, view, source box, confidence, class,
  and exclusion status;
- fused detections include the selected box, confidence, contributing views, and
  JSON fusion metadata;
- tracks include ID, box, confidence, selected anchor, raw/effective/stable
  zones, state, pending direction, dwell/confirmation progress, cooldown, gap,
  suppression reasons, emitted event, lane, and contributing views.

## Privacy and system boundaries

Tracking IDs are temporary identifiers within a single processing run. TrackBus
stores no face template, biometric, passenger identity, or cross-journey
re-identification record. The v0.2 architecture contains no backend, database,
dashboard, cloud deployment, map, forecasting, Yandex, or CAN-bus integration.
