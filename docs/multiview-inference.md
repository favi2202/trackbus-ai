# Multi-view inference

Multi-view inference runs the same detector on several overlapping crops from one
source frame, then combines the translated detections before tracking. Its goal
is to test whether difficult doorway regions benefit from a larger or differently
composed detector input while retaining a full-frame view for complete crossing
history.

It is an experimental tool, not a guaranteed accuracy improvement.

## Why separate detection from tracking

The old `YOLO.track(..., persist=True)` call owned both prediction and tracker
state for one image stream. Running it independently on left and right crops
would create separate tracker histories and ID spaces. The same passenger could
then receive several IDs, and there would be no safe point to merge detections.

v0.2 instead runs prediction without tracking for every view:

```text
one source frame
  -> predict full crop ---------+
  -> predict left crop ---------+-> translate -> exclude -> NMS -> one tracker
  -> predict right crop --------+
```

All boxes are in source coordinates before fusion. ByteTrack sees one consolidated
set and advances once for the original frame, including empty frames.

## Configure views

Views are ordinary YAML entries with unique names, normalized bounds, and an
enabled flag:

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
```

These bounds illustrate the included 320 x 240 test-video calibration. They are
not universal. A different camera, lens, mounting height, door, or zone layout
requires different bounds.

At runtime, left/top coordinates are rounded down and right/bottom coordinates
up to retain coverage. Detections from a crop are offset by its source origin and
clipped to the source frame. View bounds do not change zone polygons, lane
polygons, exclusion polygons, tracker coordinates, or output dimensions.

## Choose useful views

A practical first comparison is:

1. keep one enabled full-frame view;
2. add broad overlapping side crops around detector weak spots;
3. make the side crops overlap each other and the center enough that a person is
   not lost at a hard seam;
4. ensure passenger motion through both counting zones remains observable;
5. inspect raw, fused, and tracked output before changing more settings.

Do not create narrow crops around specific people or exact timestamps. View
bounds should follow stable camera geometry and a repeatable hypothesis, such as
"the generic detector undersamples people near both door edges."

Full-frame plus side views is safer for spatial history than replacing the full
frame with one lower-door crop. It still increases compute and may add crop-edge
false positives. The experiment must measure both benefits and failures.

## Duplicate fusion

Overlapping views often detect the same passenger more than once. TrackBus uses
class-aware source-coordinate NMS before ByteTrack:

```yaml
detection_fusion:
  method: nms
  iou_threshold: 0.50
  confidence_strategy: maximum
  prefer_full_frame: false
```

With the default settings, the highest-confidence box wins. A same-class box
from a different inference view whose IoU with that winner is at least `0.50`
can be treated as a duplicate. Detections produced by the same YOLO prediction
are never suppressed again, and at most one candidate from each view joins a
fused group. This avoids undoing the detector's own decision to retain two
nearby people. The final detection records all contributing view names and
detailed suppression metadata.

If one large winner overlaps at least two smaller candidates that do not overlap
one another enough to be duplicates, fusion treats the large box as ambiguous,
retains the smaller hypotheses, and records the discarded spanning box in their
metadata. This safeguard cannot create a person for whom no smaller hypothesis
exists.

Fusion threshold trade-offs are important:

- lower thresholds merge more shifted duplicates but can suppress two adjacent
  passengers;
- higher thresholds preserve overlapping neighbors but can let duplicate boxes
  become separate tracks;
- large, poorly localized boxes remain ambiguous when no separate smaller
  hypotheses exist.

`prefer_full_frame: true` prioritizes a full-frame box for coordinates during
winner selection. The output confidence is still the maximum confidence among
contributors. Use this only when evidence shows full-frame localization is
preferable. Weighted box fusion is not implemented in the first v0.2 pipeline.

## Exclusions happen before fusion

Known static structure is filtered after view translation but before NMS. This
ordering prevents a high-confidence door false positive from suppressing a real
passenger during fusion or consuming a tracker ID. Only a detection whose
bottom-center anchor lies inside a configured exclusion polygon is removed.

Excluded detections remain visible in raw CSV and debug output. Likely-static
diagnostics do not automatically create exclusions. TrackBus warns if an
exclusion overlaps a counting zone or configured passenger lane, and rejects a
configured lane overlap during pixel calibration.

## The old ROI warning

`camera.detection_roi` is deprecated but remains a compatibility path. A legacy
ROI is treated as one `legacy_detection_roi` inference view and produces a
migration warning. If a sole view covers less than 50% of either counting zone,
TrackBus also warns that complete history may be lost. Defining both the old ROI
and named views is an error.

The test video's old `[0.03, 0.30, 0.97, 1.00]` ROI produced detections but zero
events. It omitted part of the configured origin-zone history, created an
artificial boundary, and contributed to blinking or restarted IDs. Because the
counter correctly refuses `UNKNOWN -> zone` events, tracks without a stable
observation in both zones could not count. The correct response is to preserve
coverage and improve detection/tracking evidence, not to weaken the counter.

## Run the provided comparisons

Full-frame baseline settings through the v0.2 pipeline:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/baseline_full_frame_cpu.mp4 --config configs/presets/baseline_full_frame_cpu.yaml
```

Multi-view nano settings:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/multiview_nano.mp4 --config configs/presets/multiview_nano.yaml --device cpu
```

The full limited matrix:

```powershell
python -m trackbus.experiments --input data/input/test_video.mp4 --matrix configs/experiments/test_video.yaml --output-dir data/experiments/test_video
```

When a manually annotated event file exists:

```powershell
python -m trackbus.experiments --input data/input/test_video.mp4 --ground-truth data/ground_truth/test_video.events.json --matrix configs/experiments/test_video.yaml --output-dir data/experiments/test_video
```

The matrix contains five intentionally limited configurations: baseline and
sensitive full-frame nano, balanced full-frame small, multi-view nano, and
multi-view small. Avoid large CPU sweeps; each enabled view causes another model
prediction for every source frame.

## Inspect what changed

Enable frame-level CSV exports in a copied config:

```yaml
diagnostics:
  export_detection_csv: true
  debug_visualization: true
  trajectory_length: 30

camera:
  debug_calibration_overlay: true
```

Then compare:

- raw detections by `source_view` and frames without raw detections;
- exclusions and suspected door false positives;
- raw-to-fused reduction and `suppressed_count` metadata;
- view combinations in fused and tracked records;
- unique tracking IDs and possible restart warnings;
- duplicate-looking simultaneous tracks;
- complete `IN` and `OUT` events under unchanged rules;
- processing duration and FPS.

Debug video draws view rectangles, faint raw boxes, fused boxes with confidence
and contributing views, final IDs, counter states, lanes, zones, exclusions,
trajectories, and possible restart labels. Generated debug videos and CSVs are
local experiment artifacts and should not be committed.

## Decide whether multi-view helped

With frame-level ground truth, prefer direction-aware one-to-one event F1, then
count error, false-event risk indicators, and FPS. Without frame labels, report
only aggregate direction count error plus diagnostic risk and performance. Do not
call a configuration accurate merely because its totals equal the aggregate
ground truth.

A useful multi-view result should show more relevant detections or fewer empty
frames, convert those detections into stable complete tracks, avoid a material
increase in false events or duplicate IDs, and retain acceptable throughput. If
raw detections improve but event evidence or association does not, the conclusion
is narrower: multi-view helped detection but not the complete system.

One low-resolution clip cannot prove general accuracy. If failures remain
dominated by blinking overhead boxes, one box covering two people, or door
structure classified as a person, a properly licensed custom overhead-head or
overhead-person detector is the recommended next experiment.
