# TrackBus AI

TrackBus is a local computer-vision prototype for estimating bus occupancy from
an overhead doorway camera. It detects people with an Ultralytics YOLO model,
associates temporary video-local IDs with ByteTrack, and counts only confirmed
movements between configurable `OUTSIDE` and `INSIDE` zones.

> **Prototype status:** TrackBus v0.2 has not been validated on representative
> real-bus footage and must not be treated as an operational passenger-counting
> system. A result that matches expected aggregate totals is not, by itself, an
> accuracy measurement.

## What changed in v0.2

Detection and tracking are now separate stages. This makes overlapping inference
views possible without creating a tracker per crop or unrelated ID spaces:

```text
source frame
  -> one or more named inference views
  -> person-only YOLO.predict detections in view coordinates
  -> translation and clipping into source coordinates
  -> static exclusion filtering
  -> class-aware NMS fusion
  -> exactly one update of one ByteTrack instance
  -> zones, lanes, conservative counting, diagnostics, and annotation
```

The default remains one full-frame view. Counting, zones, lanes, exclusions,
tracking, and output rendering always use original source-frame coordinates;
inference views never crop the output video.

The counting rules are unchanged. A temporary track's first stable zone only
establishes its origin. `UNKNOWN -> INSIDE` and `UNKNOWN -> OUTSIDE` are not
events. Only a stable `OUTSIDE -> INSIDE` transition creates `IN`, and only a
stable `INSIDE -> OUTSIDE` transition creates `OUT`.

See [the architecture](docs/architecture.md), [the detection pipeline](docs/v0.2-detection-pipeline.md),
and [the multi-view guide](docs/multiview-inference.md) for design details.

## Test-video experiment result

The CPU A-E matrix used aggregate ground truth only. Full-frame nano reproduced
the v0.1.2 result of `3 IN / 2 OUT`. Multi-view nano increased raw detections
from 415 to 1,170 relative to the matching sensitive full-frame run, but counted
`3 IN / 3 OUT`, expanded IDs from 18 to 42, and fell from 8.18 to 2.09 FPS.
Multi-view small counted `2 IN / 2 OUT`, expanded IDs from 21 to 36, and ran at
1.28 FPS. Multi-view improved detector coverage but worsened aggregate direction
error and greatly increased fragmentation and CPU cost. See
[the v0.2 experiment report](docs/v0.2-experiment-report.md).

No frame-level labels were available, so these results do not provide event
precision, recall, or F1 and do not prove general accuracy.

## Requirements

- Python 3.11 or newer
- a local video readable by OpenCV
- standard Ultralytics YOLO weights or a trusted local model file
- enough disk space for weights and generated video
- optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation

The explicit ByteTrack adapter is tested against and pinned to Ultralytics
`8.4.95`. Its internal API boundary has contract tests so a future dependency
change fails visibly. The runtime does not silently fall back to `YOLO.track`.

## Installation

From the repository root, create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TrackBus rejects remote model URLs. A standard Ultralytics weight name such as
`yolo11n.pt` is supported, as is a trusted local path. Do not use third-party
weights whose origin and license you have not verified.

## Run TrackBus

Normal full-frame processing remains compatible with the existing command:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/result.mp4 --config configs/default.yaml --device cpu
```

Use a documented preset by passing it as the configuration:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/multiview_nano.mp4 --config configs/presets/multiview_nano.yaml --device cpu
```

Useful command-line options are:

```text
--input PATH                required local input video
--output PATH               annotated output video
--config PATH               YAML configuration
--model NAME_OR_PATH        standard Ultralytics name or trusted local weights
--confidence FLOAT          YOLO threshold in (0, 1]
--imgsz INTEGER             YOLO inference size, at least 32 pixels
--device DEVICE             auto, cpu, cuda, cuda:N, or a GPU index
--capacity INTEGER          nominal passenger capacity
--initial-occupancy INTEGER passengers already aboard at video start
--show                      display a preview; press q to stop
```

Command-line model, confidence, image-size, device, output, capacity, and initial
occupancy values override YAML. `--show` requires a desktop environment.

`device: auto` selects the first available CUDA GPU and otherwise uses CPU.
`model.half: true` enables FP16 only on a supported CUDA device; TrackBus safely
disables it with a warning on CPU. This task does not require replacing the
installed PyTorch or CUDA build.

## Configuration

All camera geometry uses normalized source coordinates: `[0, 0]` is the top-left
and `[1, 1]` is the bottom-right. Polygons still require calibration for each
camera position even though normalized coordinates scale across resolutions.

### Inference views and fusion

The default is one full-frame view. Multi-view inference independently predicts
on every enabled crop, translates each box back to the original frame, then
fuses duplicates before tracking:

```yaml
camera:
  detection_roi: null
  inference_views:
    - name: full
      bounds: [0.00, 0.00, 1.00, 1.00]
      enabled: true
    - name: left_doorway
      bounds: [0.00, 0.15, 0.62, 1.00]
      enabled: true
    - name: right_doorway
      bounds: [0.38, 0.15, 1.00, 1.00]
      enabled: true

detection_fusion:
  method: nms
  iou_threshold: 0.50
  confidence_strategy: maximum
  prefer_full_frame: false
```

These side-view bounds are examples for the included test-video configuration,
not universal bus-door coordinates. Fusion is conservative and cross-view: it
never performs a second suppression pass between detections from the same YOLO
prediction. It normally keeps the highest-confidence cross-view box and records
all contributing views. A large box that ambiguously spans two mutually distinct
smaller hypotheses is dropped with explicit suppression metadata instead of
erasing both people. Set `prefer_full_frame: true` only when deliberately using
full-frame coordinates; the fused confidence still remains the maximum among
contributors. Lowering the positive IoU threshold merges more boxes but
increases the risk of suppressing adjacent people.

`camera.detection_roi` is deprecated. An old configuration containing only that
key still loads as one named `legacy_detection_roi` view and emits a migration
warning. A configuration cannot contain both `detection_roi` and
`inference_views`. A sole crop that covers less than 50% of either counting zone
also warns that complete crossing history may be lost. Omitting both settings
retains full-frame inference.

### Presets

The ordinary YAML presets are comparison starting points, not accuracy claims:

| Preset | Model and mode |
| --- | --- |
| `baseline_full_frame_cpu.yaml` | `yolo11n.pt`, 0.35, 640, full frame, CPU |
| `sensitive_full_frame_cpu.yaml` | `yolo11n.pt`, 0.25, 640, full frame, CPU |
| `balanced_full_frame.yaml` | `yolo11s.pt`, 0.20, 640, full frame |
| `multiview_nano.yaml` | `yolo11n.pt`, 0.25, full + left + right |
| `multiview_small.yaml` | `yolo11s.pt`, 0.20, full + left + right |

They live in `configs/presets/`. `configs/test_video.yaml` is a camera-specific
example and is not suitable for every bus camera.

### Zones, lanes, exclusions, and tracking

`zones.outside` and `zones.inside` define the counting areas. Keep a neutral gap
between them. The bottom-center of a tracked box is the counting anchor. A
destination zone must persist for `tracking.minimum_zone_frames` consecutive
frames before an event is confirmed; returning to the origin cancels an
incomplete transition.

Optional `camera.doorway_lanes` are diagnostic only. Their configured `center`
or `bottom_center` anchor does not change counting. `camera.exclusion_polygons`
are only for known static structures. A raw detection is excluded when its
bottom-center anchor is inside one of these polygons, after source-coordinate
translation and before fusion or tracking. TrackBus warns when an exclusion
overlaps a configured lane or counting zone, and pixel calibration rejects an
overlap with a configured passenger lane. It does not automatically suppress
objects merely because they appear stationary.

ByteTrack thresholds can be changed in the referenced tracker YAML or overridden
under `tracking` with `track_high_thresh`, `track_low_thresh`,
`new_track_thresh`, `track_buffer`, `match_thresh`, and `fuse_score`. One tracker
is advanced once per original frame, including frames with no detections.

## Outputs and visual diagnostics

For an output named `result.mp4`, null artifact paths produce:

- `result.mp4`: annotated source-resolution video;
- `result.events.csv`: confirmed `IN` and `OUT` events;
- `result.summary.json`: counts, performance, detector/fusion/tracker statistics,
  enabled views, dependency version, and doorway diagnostics.

Potentially large frame-level exports are off by default. Enable all three with:

```yaml
diagnostics:
  export_detection_csv: true

outputs:
  raw_detections_csv: null
  fused_detections_csv: null
  tracks_csv: null
```

This adds `result.raw_detections.csv`, `result.fused_detections.csv`, and
`result.tracks.csv`. The raw export includes source view, source-coordinate box,
confidence, class, timestamp, and exclusion status. The fused export includes
contributing views and NMS metadata. The track export includes ID, box,
confidence, anchor, zone, lane, counter state, and view provenance.

`camera.debug_calibration_overlay: true` draws inference-view rectangles, lanes,
exclusions, and excluded boxes. `diagnostics.debug_visualization: true` also
draws faint raw boxes, fused boxes with confidence and view labels, tracked view
provenance, trajectories, possible restart warnings, and raw/fused totals. Normal
output stays less cluttered when these flags are false.

## Manual ground truth and evaluation

Create or resume human-authored event labels:

```powershell
python -m trackbus.annotate_events --input data/input/test_video.mp4 --output data/ground_truth/test_video.events.json
```

Controls are:

| Key | Action |
| --- | --- |
| `Space` or `P` | Play or pause |
| `A` or `,` | Pause and step back one frame |
| `D` or `.` | Pause and step forward one frame |
| `I` | Mark a completed IN crossing |
| `O` | Mark a completed OUT crossing |
| `U` | Undo the latest mark |
| `S` | Save without quitting |
| `Q` or `Esc` | Save and quit |

The screen shows frame number, timestamp, current totals, existing annotations,
and an event timeline. No ground-truth events are inferred automatically.

Evaluate direction-aware, one-to-one event matches against the processing CSV:

```powershell
python -m trackbus.evaluation --ground-truth data/ground_truth/test_video.events.json --predictions data/output/result.events.csv --tolerance-frames 25 --output data/output/result.evaluation.json
```

If frame labels do not exist, report aggregate count error without claiming
precision, recall, or F1:

```powershell
python -m trackbus.evaluation --expected-entered 6 --expected-exited 2 --predicted-entered 3 --predicted-exited 2
```

See [the evaluation guide](docs/evaluation.md) for the JSON schema and metrics.

## Reproducible experiments

The included matrix intentionally contains only five comparisons: baseline and
sensitive full-frame nano, full-frame small, multi-view nano, and multi-view
small. Run it with aggregate ground truth from the matrix:

```powershell
python -m trackbus.experiments --input data/input/test_video.mp4 --matrix configs/experiments/test_video.yaml --output-dir data/experiments/test_video
```

Add `--ground-truth data/ground_truth/test_video.events.json` when manually
labelled events exist. The runner writes a resolved config, log, artifacts, and
result JSON for each run plus `leaderboard.csv`, `leaderboard.json`, and
`report.md`. With frame labels it ranks primarily by event F1. Otherwise it
ranks by direction count error, false-event risk indicators, then processing
FPS. Aggregate-only ranking cannot establish event accuracy.

Generated videos, model weights, experiment output, virtual environments, and
local absolute paths must not be committed.

## Tests and checks

Ordinary tests use fake detector and tracker implementations and do not download
weights:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pip check
git diff --check
```

## Limitations

- Generic person models are weak on low-resolution overhead imagery, partial
  bodies, children, occlusion, and crowded doorways.
- Multiple views can recover detections, but they also multiply inference cost
  and can add false positives or duplicate risk when calibration or fusion is
  poor.
- ByteTrack IDs can fragment or switch after gaps, which can prevent a complete
  transition from being observed under one ID.
- A large detector box can merge two nearby passengers; NMS cannot reconstruct
  people the detector never separated.
- Starting occupancy must be supplied, and recorded-video processing has no
  physical door-state signal.
- One short, low-resolution video cannot demonstrate accuracy or generalization.

Representative, privacy-reviewed footage and manual event labels are required
before making accuracy claims. If the generic models remain unreliable, a
properly licensed custom detector trained for overhead heads or people is the
recommended next computer-vision step—not weaker counting rules.

## Privacy and scope

TrackBus performs ordinary person detection and assigns temporary IDs only
within one video-processing run. It does not use face recognition, biometrics,
passenger identity storage, or cross-journey re-identification. It creates count
events and an optional annotated copy of the supplied video; real deployments
still need appropriate consent, retention, access-control, signage, and legal
review.

This version deliberately excludes backends, databases, dashboards, cloud
deployment, maps, forecasting, Yandex integration, and CAN-bus integration.
See [the roadmap](docs/roadmap.md) for evidence-first next steps.
