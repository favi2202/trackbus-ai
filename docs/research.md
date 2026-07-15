# Research notes

TrackBus tests a narrow technical hypothesis: a detector, temporary ByteTrack
IDs, and a conservative two-zone state machine can produce useful doorway
crossing events from overhead video. The counting logic works when one stable ID
is observed in both zones. The unresolved question is whether the visual front
end supplies detections and associations that are complete enough for real bus
conditions.

## Evidence before v0.2

The safeguarded v0.1.2 full-frame run on the 320 x 240, 25 FPS, 589-frame test
video used `yolo11n.pt`, confidence `0.35`, and image size `640` on CPU. Its
preserved summary reported `3 IN / 2 OUT` against manually verified aggregate
totals of `6 IN / 2 OUT`, with approximately 24 FPS processing. These totals are
a historical baseline, not event precision or recall.

Manual review found a recognizable detector pattern:

- people through the center of the doorway were detected more reliably;
- left- and right-side boxes blinked, disappeared, or began too late;
- one large box sometimes covered a passenger and door structure;
- static door structure could be classified as a person;
- two visible people could collapse into one detection.

These failures are consistent with a generic upright-person detector operating
on low-resolution overhead imagery. Tracking and counting cannot recover a
person whom the detector never separates or cannot observe on both sides.

## Why the old ROI experiment produced zero events

The v0.1.2 experiment cropped inference to `[0.03, 0.30, 0.97, 1.00]`. It still
produced 224 person detections in 165 frames, 14 IDs, possible restart signals,
and many tracks touching the artificial crop boundary, yet counted no events.
Therefore the result was not simply "no detections."

The old combined `YOLO.track(..., persist=True)` path performed both detection
and tracking inside the crop. Removing the upper part of the scene removed part
of the spatial history required for a complete `OUTSIDE -> INSIDE` or
`INSIDE -> OUTSIDE` transition. A person could first appear after the origin
zone, disappear at the crop boundary, or receive a new ID before reaching the
other zone. The conservative state machine correctly refused to turn an
`UNKNOWN` first observation into an event.

This explains why a crop may retain many detections yet lose every complete
crossing history. v0.2 consequently warns when a sole inference crop covers less
than 50% of either counting zone. It does not weaken the transition rules.

## v0.2 multi-view hypothesis

Multi-view inference tests whether overlapping crops give side-door passengers
more detector pixels or a more favorable composition while a full-frame view
retains complete spatial coverage. It is not valid to give each crop its own
persistent tracker: that would create unrelated IDs for the same passenger.

The v0.2 experimental unit is one original video frame:

```text
predict each view -> source-coordinate translation -> exclusions -> NMS
                  -> one fused set -> one ByteTrack update
```

This design lets the experiment distinguish several outcomes:

- additional useful raw detections that become stable tracks;
- redundant detections correctly removed by NMS;
- duplicates that survive fusion and fragment into extra tracks;
- new false positives from enlarged crop inference;
- changed event count under unchanged counting rules;
- loss of throughput from multiple model invocations per frame.

NMS is intentionally conservative, class-aware, and cross-view. It does not
re-suppress boxes retained by one detector call. When a large candidate spans
two smaller, mutually distinct hypotheses, it preserves the smaller boxes and
records the ambiguous discard; it still cannot split a large box when the
detector emitted no separate hypotheses. A lower fusion threshold may reduce
duplicates while erasing adjacent people; a higher threshold may preserve
neighbors while allowing duplicate tracks. That trade-off must be measured, not
selected from aggregate totals alone.

## Evaluation method

The included matrix compares only five bounded configurations:

1. nano baseline at confidence 0.35, full frame;
2. nano at confidence 0.25, full frame;
3. small model at confidence 0.20, full frame;
4. nano at confidence 0.25, full plus overlapping side views;
5. small model at confidence 0.20, full plus overlapping side views.

Every run records counts, raw and fused detections, empty frames, unique IDs,
restart and false-event risk indicators, processing duration, and FPS. With
manual frame-level labels, direction-aware one-to-one matching supplies TP, FP,
FN, precision, recall, F1, timing error, and direction-specific metrics. Without
those labels, only aggregate IN/OUT and total-crossing errors are available.

Aggregate equality can hide a missed real event replaced by a false event, wrong
direction, a duplicate, or incorrect timing. Therefore aggregate-only runs rank
by direction count error, diagnostic risk, and FPS, and must not be described as
event-accurate. With event labels, F1 is the primary ranking metric.

## v0.2 test-video result

The bounded CPU matrix completed all five planned runs. Full-frame nano,
sensitive nano, and full-frame small each counted `3 IN / 2 OUT`. Multi-view
nano counted `3 IN / 3 OUT`, while multi-view small regressed to `2 IN / 2 OUT`.
Both matched multi-view comparisons increased aggregate direction count error
from 3 to 4.

Multi-view did improve the detector-coverage proxies. Nano raw detections rose
from 415 full-frame to 1,170 multi-view and raw-empty frames fell from 350 to
236. Small-model raw detections rose from 778 to 1,613 and raw-empty frames fell
from 216 to 152. However, nano IDs grew from 18 to 42 and possible restart
warnings from 1 to 7; small-model IDs grew from 21 to 36 and warnings from 3 to
8. FPS fell from 8.18 to 2.09 for nano and from 4.17 to 1.28 for small.

The supported conclusion is narrow: the tested views generated more hypotheses,
but fusion and tracking did not turn them into more complete events. See
`docs/v0.2-experiment-report.md` for the full aggregate-only leaderboard and
runtime caveats.

## Threats to validity

One short video from one fixed 320 x 240 viewpoint cannot establish accuracy,
robustness, or generalization. It does not sample enough cameras, doors, clothing,
body sizes, lighting, vibration, boarding patterns, crowd density, or occlusion.
Repeatedly tuning view bounds, confidence, tracker thresholds, fusion, or zones
against this one clip risks overfitting even if the counting rules themselves do
not mention specific people or timestamps.

The current false-event risk score is a comparison aid built from possible ID
restarts, likely-static tracks, overlap disappearance, and excess unique IDs. It
is not a substitute for human event labels. Likewise, likely-static diagnostics
do not authorize automatic suppression.

Throughput on one CPU or GPU is environment-specific. Multi-view processing
usually performs one model prediction per enabled view per source frame, so it
should be expected to reduce FPS; the measured trade-off belongs beside any
detection improvement.

## Likely next detector work

If multi-view experiments still show blinking edge detections, merged people, or
door false positives, custom detector training is recommended. A carefully
licensed dataset should represent overhead heads or people, full and partial
bodies, simultaneous crossings, children, bags, door hardware, lighting changes,
and multiple buses. It must be split by bus/camera or journey so evaluation is
not merely memorization of one doorway.

A custom overhead-head detector may separate adjacent passengers better than a
generic person model, but it still requires representative labels and held-out
evaluation. It should feed the same model-independent detection interface so the
fusion, tracker, counter, evaluator, and experiment framework remain unchanged.

No research step requires face recognition or persistent identity. Ground-truth
labels describe only completed `IN` and `OUT` events, and runtime IDs remain
temporary within one video.
