# TrackBus recognition failure audit

## Current pipeline

The current foundation uses the intended model-independent separation:

```text
source video
→ decoded BGR frame
→ configured inference views
→ YOLO person-only prediction
→ source-coordinate translation
→ exclusion diagnostics/filtering
→ class-aware NMS fusion
→ one explicit ByteTrack update
→ temporary trajectory
→ calibrated zone membership
→ temporal three-zone event state
→ IN/OUT event and occupancy
```

The deprecated combined `YOLO.track()` adapter remains only for compatibility;
the production v0.2.1 path does not fall back to it.

## Where a visible passenger can be lost

| Stage | Failure evidence | Correct measurement |
|---|---|---|
| Video/camera | Subject absent, too small, blurred, dark, blocked or compressed | Camera-quality buckets and manual review |
| Detector | No person box, intermittent box, weak confidence, clipped box | Box annotations plus detector precision/recall/F1 |
| View translation/fusion | Crop omits the path or two people merge/suppress | Raw vs fused detection exports |
| Tracker | ID switch, fragmentation or track loss after occlusion | ID switches, fragments and continuity |
| Calibration | Anchor does not traverse OUTSIDE/DOOR/INSIDE polygons | Per-camera overlay and labelled trajectories |
| Event state | Valid trajectory is suppressed or times out | Event annotations and suppression reasons |
| Integration | Valid local event is not delivered/persisted | API retry queue and gateway/database tests |

Aggregate totals alone cannot identify the failing component.

## Evidence already established

- The 589-frame historical clip produced 3 IN / 2 OUT against aggregate totals
  of 6 IN / 2 OUT, but it has no box-level person labels.
- The event-stability branch improved a separately annotated real-bus event F1
  from 0.1818 to 0.6000. That is event evidence, not detector F1.
- Raising model size/input alone did not automatically fix counts.
- A sole doorway crop produced 0 IN / 0 OUT because it removed complete spatial
  transition history.
- Multiview inference increased raw detections but also duplicates,
  fragmentation, and CPU cost; it is not promoted as the default.

## Baseline for this measurement iteration

Before detector-benchmark implementation on `agent/trackbus-foundation`:

- Ruff: passed;
- Python tests: 250 passed;
- no local input video or model weights are version-controlled;
- no box-level detector ground truth is version-controlled;
- therefore no new detector accuracy number can be honestly reported yet.

This iteration adds measurement only. It does not change detector predictions,
ByteTrack association, zone geometry, or event-count behavior.
