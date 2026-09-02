# TrackBus detector-only benchmark

TrackBus counting combines several independent failure sources. A missed final
IN/OUT event can begin in the detector, tracker, calibration, three-zone state
machine, or event confirmation logic. The detector-only benchmark stops after
YOLO class-0 (`person`) prediction so those causes are not mixed together.

It does **not** run ByteTrack, zone membership, passenger counting, API delivery,
or the dashboard.

## Single detector run

PowerShell:

```powershell
python -m trackbus.benchmark_detection `
  --video test_video.mp4 `
  --model yolo11s.pt `
  --imgsz 960 `
  --detector-floor 0.10 `
  --device auto `
  --output data/experiments/detector-small-960.json
```

The command also writes a `.frames.csv` file unless `--frames-csv` is supplied.
The JSON records:

- detections per frame and zero-detection streaks;
- confidence distribution and configurable low-confidence count;
- detections clipped by a source-frame boundary;
- inference latency, decode-inclusive pipeline FPS, and a detector FPS estimate;
- consecutive-frame bounding-box matching and IoU as a stability proxy;
- the exact model, image size, threshold, device, video resolution, and frame count.

The stability value is only a proxy. People genuinely move between frames, and
there is no identity association in this command.

Use `--max-frames 100` only for an executable smoke test. Do not compare a
100-frame run with a complete-video run.

## Box-level ground truth

Without manual person boxes the report sets:

```json
"metric_status": "proxy_only_no_box_ground_truth"
```

It does not invent precision, recall, F1, missed-person counts, or a winning
configuration. Existing TrackBus event annotations label completed IN/OUT
events; they are not detector boxes and cannot be substituted.

Optional detector annotations use this schema:

```json
{
  "schema_version": 1,
  "coordinate_space": "pixels",
  "video": {
    "filename": "test_video.mp4",
    "width": 320,
    "height": 240
  },
  "frames": [
    {
      "frame": 0,
      "persons": [
        {"id": "optional-local-label", "bbox": [12, 20, 70, 180]}
      ]
    },
    {"frame": 1, "persons": []}
  ]
}
```

Only listed frames are treated as annotated. Include an explicit empty
`persons` list when a reviewed frame contains no people. Unlisted frames do not
become false positives. `coordinate_space` may be `pixels` or `normalized`.

Run with labels:

```powershell
python -m trackbus.benchmark_detection `
  --video test_video.mp4 `
  --model yolo11s.pt `
  --imgsz 960 `
  --detector-floor 0.10 `
  --ground-truth data/ground_truth/test_video.detections.json `
  --ground-truth-iou 0.50
```

Predictions and labels are matched independently per annotated frame using a
maximum-cardinality IoU-threshold assignment. The report then includes true
positives, false positives, false negatives/missed persons, precision, recall,
and F1. These remain detector metrics, not passenger-counting accuracy.

## Bounded model sweep

The checked-in matrix is explicit rather than an enormous model × resolution ×
threshold product:

```powershell
python -m trackbus.sweep_detection `
  --video test_video.mp4 `
  --matrix configs/detection_sweep.yaml `
  --output-dir data/experiments/detection-sweep
```

Add `--ground-truth` to compare real detector recall/F1. Each run gets its own
`benchmark.json` and `frames.csv`; the directory also receives `results.json`
and `leaderboard.csv`. A matrix is limited to at most 24 explicit runs, and may
set a smaller `maximum_runs` guard.

On CPU, start with nano and small at 640. On an RTX-class CUDA setup, keep the
960 and 1280 small-model comparisons. `device: auto` uses CUDA only when the installed
PyTorch build can access it and remains CPU-compatible.

The matrix may opt into `preprocessing_profile: low_light` or `contrast` for a
specific run. Both preserve the source dimensions and report their measured
latency. They are experiments, disabled by default, and must be compared on the
same independently labeled frames; an attractive image is not accuracy evidence.

## Interpreting results

Use box ground truth to answer whether the detector misses visible people. Use
TrackBus track exports and diagnostics to measure ID fragmentation. Use event
annotations for boarding/alighting precision and recall. Never call detector F1
or mAP “passenger counting accuracy.”

The next experiment should change one component at a time:

1. detector model/resolution/confidence;
2. ByteTrack thresholds and buffer profiles;
3. temporary continuity/stitching;
4. calibrated doorway high-resolution inference;
5. optional appearance-assisted association;
6. TrackBus-specific fine-tuning on independently split footage.
