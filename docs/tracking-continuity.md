# ByteTrack profiles and short-gap continuity

This iteration addresses two different tracking problems without changing the
YOLO detector, zone state machine, or event rules:

1. ByteTrack thresholds and buffers can be compared through explicit profiles.
2. An optional TrackBus wrapper can reconnect a newly assigned raw ByteTrack ID
   to a recently missing anonymous local track when the geometry is consistent.

Neither feature is a claimed accuracy improvement until it is evaluated on
annotated footage. Continuity is disabled by default.

## ByteTrack profiles

The repository includes three bounded experimental profiles:

| Profile | Intended comparison | Buffer |
| --- | --- | ---: |
| `configs/tracking_fast.yaml` | stricter, shorter memory | 30 frames |
| `configs/tracking_balanced.yaml` | middle reference | 45 frames |
| `configs/tracking_occlusion.yaml` | weaker detections and brief occlusion | 75 frames |

The existing `configs/bytetrack_trackbus.yaml` remains supported and unchanged.
The names describe experiment intent, not proven ranking.

Select one in a camera or run configuration:

```yaml
tracking:
  tracker: configs/tracking_balanced.yaml
```

Detector and tracker thresholds interact. YOLO removes predictions below
`model.confidence` before ByteTrack sees them. A ByteTrack `track_low_thresh` of
`0.10` cannot recover `0.10–0.19` predictions if YOLO ran at `0.20`. Establish a
detector threshold with `trackbus.benchmark_detection`, then hold it constant
while comparing tracking profiles.

## Optional continuity layer

Enable it only in an experiment configuration:

```yaml
tracking:
  tracker: configs/tracking_balanced.yaml
  continuity:
    enabled: true
    max_gap_frames: 3
    max_centroid_distance: 0.08
    minimum_iou: 0.05
    maximum_size_ratio: 1.8
    minimum_direction_cosine: -0.25
    minimum_match_score: 0.55
```

For a new raw ID, TrackBus predicts the recently missing track position from its
last box and velocity. It considers normalized centroid distance, predicted-box
IoU, box-size similarity, direction, and time gap. Matching is deterministic and
one-to-one, so two simultaneous passengers cannot be merged into the same stable
ID in one frame.

The adapter deliberately does **not**:

- generate a box during missing frames;
- use faces, clothing embeddings, or persistent identity;
- keep a memory beyond the configured short gap;
- override ByteTrack when the original raw ID continues;
- claim that a geometric stitch is certainly the same person.

A stitched observation includes `raw_tracking_id`, the previous raw ID, gap,
match score, IoU, centroid distance, and the quality flag
`track_continuity_stitch`. These are auditable quality signals, not calibrated
probabilities.

## Evaluation sequence

Use the same complete video, camera calibration, detector model, resolution, and
confidence for every run.

1. Run the existing ByteTrack profile with continuity disabled.
2. Run `tracking_fast`, `tracking_balanced`, and `tracking_occlusion`, still with
   continuity disabled.
3. Compare exported tracks: unique IDs, possible restarts, detection gaps,
   fragmentation, total pipeline FPS, and event-level metrics when annotations
   exist.
4. Enable continuity only for the strongest profile and repeat.
5. Reject the change if false merges, false events, or speed loss outweigh fewer
   fragments.

PowerShell example:

```powershell
python -m trackbus.main `
  --input "C:\TrackBus\videos\door-test.mp4" `
  --config "configs\my_camera_tracking_test.yaml" `
  --output "data\experiments\balanced-continuity.mp4"
```

The JSON summary records the effective ByteTrack settings and a
`tracking_continuity` section containing stitch attempts, accepted fragments,
new stable tracks, expired memories, and maximum accepted gap. These counters
measure behavior, not correctness. Event annotations remain necessary to prove
whether IN/OUT accuracy improved.

## Current evidence boundary

The repository contains deterministic regression tests for short gaps, excessive
gaps, distance, scale, direction, simultaneous passengers, one-to-one matching,
reset behavior, configuration bounds, and the real Ultralytics profile contract.
It does not contain the user's real bus video or box/track ground truth, so no
profile is promoted as the production winner in this commit.
