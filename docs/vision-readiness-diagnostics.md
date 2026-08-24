# Vision readiness diagnostics

TrackBus now separates three questions that were previously easy to mix up:

1. Did YOLO observe a person?
2. Did ByteTrack preserve the anonymous local track?
3. Did the track complete a calibrated OUTSIDE → DOOR → INSIDE transition?

The tools in this guide produce diagnostic evidence. They do not claim detector
accuracy, tracking accuracy, or event-counting accuracy without annotations.

## Detector floor and ByteTrack thresholds

`model.detector_floor` is the lowest YOLO confidence submitted to ByteTrack.
`model.confidence` remains the high-confidence reference used in summaries and
failure diagnostics. When `detector_floor` is omitted, TrackBus preserves the old
behavior and uses `model.confidence` as the YOLO cutoff.

The default handoff is:

```yaml
model:
  confidence: 0.35
  detector_floor: 0.10

tracking:
  tracker: configs/bytetrack_trackbus.yaml
```

The supplied ByteTrack profile uses a `0.10` low threshold and `0.25` high/new
thresholds. A `0.10–0.24` observation can therefore maintain an existing track,
but cannot start a new one. TrackBus rejects tracker profiles where the low
threshold exceeds the high threshold or the new-track threshold is below the
high threshold.

PowerShell override:

```powershell
python -m trackbus.main `
  --input ".\videos\bus-test.mp4" `
  --config ".\configs\default.yaml" `
  --confidence 0.35 `
  --detector-floor 0.10 `
  --output ".\data\output\bus-test.mp4"
```

The JSON summary contains `detector_floor`,
`detector_tracker_confidence_contract`, and `detector_confidence_bands`. These
are behavior counters, not precision or recall.

## Camera-quality analyzer

Run a bounded analysis before tuning models or trackers:

```powershell
python -m trackbus.camera_quality `
  --source ".\videos\bus-test.mp4" `
  --config ".\configs\default.yaml" `
  --sample-every 15 `
  --max-samples 120 `
  --output ".\data\output\bus-test.camera-quality.json"
```

The analyzer reports:

- resolution, FPS, duration, and sampling evidence;
- brightness, exposure clipping, and blur heuristics;
- detection confidence distribution and person-height ratio;
- edge-clipped detections and sudden count drops;
- configured inference-view coverage of the OUTSIDE and INSIDE zones;
- device and sampled-pipeline speed.

Statuses are `GOOD`, `MARGINAL`, or `UNSUITABLE`. Every status has concrete
reasons and recommendations. The report contains `diagnostic_not_accuracy: true`
because a quality heuristic cannot replace annotated people, tracks, or events.

To assess only optical and calibration quality without loading YOLO:

```powershell
python -m trackbus.camera_quality `
  --source ".\videos\bus-test.mp4" `
  --no-detection
```

No frames are stored by this analyzer.

## Failure mining

Failure mining is disabled by default. Enable metadata-only JSONL evidence for a
specific experiment:

```yaml
diagnostics:
  failure_mining:
    enabled: true
    output_jsonl: data/output/bus-test.failures.jsonl
    capture_frames: false
    frames_directory: null
    maximum_records: 500
    maximum_captured_frames: 50
    jpeg_quality: 85
    low_confidence_threshold: 0.35
    short_track_maximum_frames: 5
    sudden_detection_drop_ratio: 0.50
```

The miner can flag low-confidence detections, sudden detection drops, track
disappearance/recovery, short tracks, possible restarts, continuity stitches,
edge clipping, heavy overlap, ambiguous zone transitions, incomplete
transitions, and occupancy-boundary clamps.

Every ID is temporary and scoped to one video. The miner performs no facial
recognition, biometric matching, or upload.

### Optional frame capture

Set `capture_frames: true` only for an approved local debugging session. Frame
capture has a separate hard limit and writes JPEGs to the configured local
directory. The images may contain passengers and must be handled as personal
data. TrackBus never enables frame capture implicitly and never uploads these
files.

## Recommended video-test order

1. Run `trackbus.camera_quality` and correct unsuitable camera conditions.
2. Run the detector-only benchmark with box annotations when available.
3. Hold the detector settings constant and compare ByteTrack profiles.
4. Enable metadata-only failure mining for the strongest candidate.
5. Evaluate IN/OUT events against manual event annotations.
6. Keep a change only when the relevant measured stage improves without an
   unacceptable speed or false-event regression.

Do not promote a model or profile based only on more boxes, fewer anonymous IDs,
or a visually smoother overlay.
