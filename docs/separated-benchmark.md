# TrackBus separated benchmark report

Detection, tracking, doorway counting, and runtime answer different questions.
`trackbus.benchmark_report` combines their existing artifacts into one auditable
JSON report and comparison CSV while keeping the metrics separate.

PowerShell:

```powershell
python -m trackbus.benchmark_report `
  --detection ".\data\experiments\detector\benchmark.json" `
  --processing ".\data\output\processing-summary.json" `
  --counting ".\data\output\counting-evaluation.json" `
  --output ".\data\experiments\separated-report.json"
```

`--csv` is optional. Without it, the command writes
`separated-report.csv` beside the JSON file. One or two input artifacts may be
omitted; the affected sections are explicitly marked unavailable or
diagnostics-only.

## Metric contract

- Detection precision, recall, and F1 require per-frame person boxes.
- Counting precision, recall, and F1 require event-level IN/OUT ground truth.
- Fragmentation, track lifetime, possible restarts, and recovered gaps are
  tracking diagnostics. ID-switch accuracy requires identity ground truth.
- FPS and latency are runtime results for the recorded device and effective
  configuration. Peak memory remains unavailable unless a future profiler
  records it.

The output includes the effective model, image size, detector floor,
preprocessing profile, precision, device, tracker configuration, and continuity
settings found in the source artifacts. It never treats detector F1 as passenger
counting accuracy or uses neighboring frames from the same recording as
independent evidence.

Use a bounded comparison: first YOLO11n and YOLO11s, then 640/960/1280 where
hardware permits, detector floors 0.05/0.10/0.15, the three checked-in ByteTrack
profiles, and at most one justified preprocessing profile. Change one component
at a time and retain the JSON/CSV artifacts with the source-video and annotation
version.
