# Event annotation and evaluation

Aggregate totals can show count error, but they cannot reveal whether a run
detected the correct crossings. TrackBus therefore supports manual frame-level
labels and direction-aware event evaluation. Ground truth is always created by a
person; the annotation tool does not infer or pre-fill events.

## Annotate a video

```powershell
python -m trackbus.annotate_events `
  --input data/input/test_video.mp4 `
  --output data/ground_truth/test_video.events.json
```

If the output already exists, the tool validates its video filename and FPS,
loads its events, and resumes at the last marked frame. The controls are:

| Key | Action |
| --- | --- |
| `Space` or `P` | Play or pause |
| `A` or `,` | Pause and step back one frame |
| `D` or `.` | Pause and step forward one frame |
| `I` | Mark a completed IN crossing at the displayed frame |
| `O` | Mark a completed OUT crossing at the displayed frame |
| `U` | Undo the most recently added event |
| `S` | Save without quitting |
| `Q` or `Esc` | Save and quit safely |

The overlay displays the current frame and timestamp, the event totals, and a
timeline with all existing IN and OUT marks. Repeating the same direction on the
same frame does not create a duplicate.

The JSON document records a schema version, the video filename (and optional
identifier), source FPS, and events. Each event contains a zero-based frame,
timestamp in seconds, direction (`IN` or `OUT`), and an optional note:

```json
{
  "schema_version": "trackbus.events/v1",
  "video": {"filename": "test_video.mp4"},
  "source_fps": 25.0,
  "events": [
    {"frame": 120, "timestamp": 4.8, "direction": "IN"}
  ]
}
```

## Evaluate frame-level events

The prediction input can be another event JSON file or the `.events.csv` file
written by normal TrackBus processing.

```powershell
python -m trackbus.evaluation `
  --ground-truth data/ground_truth/test_video.events.json `
  --predictions data/output/test_video.events.csv `
  --tolerance-frames 25 `
  --output data/output/test_video.evaluation.json
```

Use `--tolerance-seconds` instead of `--tolerance-frames` when a time-based
tolerance is more appropriate. Matching is direction-aware and one-to-one. It
first maximizes the number of matched events and then minimizes their total
absolute timing error, so input ordering and duplicate predictions cannot inflate
true positives.

The report contains TP, FP, FN, precision, recall, F1, signed and absolute mean
timing error, the same metrics for IN and OUT separately, matched pairs, and
absolute IN, OUT, and total-crossing count errors.

## Aggregate-only evaluation

When frame-level labels do not yet exist, report count errors without claiming
precision or recall:

```powershell
python -m trackbus.evaluation `
  --expected-entered 6 --expected-exited 2 `
  --predicted-entered 3 --predicted-exited 2
```

Aggregate-only output deliberately sets event precision, recall, F1, and timing
error to `null`. Matching totals can still hide a missed crossing replaced by a
false event, duplicate events, or incorrect timing; consequently, aggregate
results are insufficient for choosing an accurate configuration.
