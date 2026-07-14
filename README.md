
# TrackBus AI

TrackBus is an early computer-vision prototype that estimates bus occupancy by
counting people who cross a doorway. It detects people with a pretrained
Ultralytics YOLO model, assigns short-lived tracking IDs with ByteTrack, and
confirms movement between configurable OUTSIDE and INSIDE zones.

> **Prototype status:** TrackBus v0.1.1 has not been validated on real Tashkent
> buses and must not be treated as an operational passenger-counting system.

## v0.1.1 scope

This version processes one recorded local video. It:

- detects the YOLO `person` class and tracks detections with ByteTrack;
- counts only confirmed OUTSIDE -> INSIDE and INSIDE -> OUTSIDE transitions;
- renders zones, boxes, tracking IDs, totals, occupancy, and capacity percentage;
- writes an annotated video, CSV event log, and JSON run summary;
- keeps counting logic independent from the model so it can be unit-tested.

Door sensors, live cameras, APIs, databases, dashboards, forecasting, maps,
custom training, and cloud deployment are deliberately outside v0.1.1.

## Architecture

The processing flow is:

```text
video frame -> YOLO person detection -> ByteTrack IDs -> zone membership
            -> transition state machine -> counts/events -> video + CSV + JSON
```

The bottom-center of each person's bounding box is used as the zone anchor. A
track must remain in the destination zone for a configurable number of frames
before a crossing is confirmed. The neutral gap between zones is the transition
area. Returning to the origin zone cancels an incomplete crossing.

See [docs/architecture.md](docs/architecture.md) for module responsibilities.

## Requirements

- Python 3.11 or newer (Python 3.11 is recommended for this prototype)
- a local video readable by OpenCV
- enough disk space for model weights and output video
- optional NVIDIA GPU with a compatible CUDA-enabled PyTorch installation

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

The default `yolo11n.pt` weights are downloaded by Ultralytics on the first run
if they are not already available. A local weights path can be supplied instead.

### CPU

CPU mode needs no CUDA setup and is the simplest way to verify the prototype:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --device cpu
```

It will usually process more slowly than the source video's real-time frame rate.

### NVIDIA CUDA

Install the PyTorch build matching the NVIDIA driver/CUDA environment using the
official PyTorch installation selector, then install this project's requirements.
Verify CUDA before running:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Use `--device cuda`, `--device cuda:0`, or a GPU index such as `--device 0`.
`--device auto` selects the first available CUDA GPU and otherwise uses the CPU.

## Run TrackBus

Place a video in `data/input/`, then run:

```powershell
python -m trackbus.main `
  --input data/input/test_video.mp4 `
  --output data/output/result.mp4 `
  --capacity 40 `
  --device auto
```

Equivalent one-line command:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/result.mp4 --capacity 40 --device auto
```

Useful options:

```text
--input PATH                required local input video
--output PATH               annotated output video
--capacity INTEGER          nominal passenger capacity
--initial-occupancy INTEGER passengers already aboard at video start
--device DEVICE             auto, cpu, cuda, cuda:N, or GPU index
--confidence FLOAT          YOLO confidence in the range (0, 1]
--imgsz INTEGER             YOLO inference image size in pixels (minimum 32)
--model NAME_OR_PATH        Ultralytics model name or local weights
--config PATH               YAML configuration (default: configs/default.yaml)
--show                      show a preview; press q to stop
```

`--show` requires a desktop environment. Omit it on servers and remote shells.

## Recommended inference presets

These are starting points for comparison, not accuracy guarantees. Keep the
input, zones, and output review process identical when comparing presets.

Fast CPU smoke test:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/result_cpu.mp4 --model yolo11n.pt --confidence 0.25 --imgsz 416 --device cpu
```

Accuracy-oriented GPU comparison:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/result_gpu.mp4 --model yolo11m.pt --confidence 0.20 --imgsz 960 --device 0
```

Low-resolution overhead footage, including 320x240 sources:

```powershell
python -m trackbus.main --input data/input/test_video.mp4 --output data/output/result_overhead.mp4 --model yolo11s.pt --confidence 0.15 --imgsz 960 --device 0
```

The low-resolution preset allows weaker person detections to reach ByteTrack's
second-stage association. A larger inference size may help the detector process
small people, but it cannot restore detail absent from the source. Review false
positives as confidence is lowered. Use `--device cpu` if no GPU is available,
with substantially slower processing expected.

## Configuration and zones

Edit `configs/default.yaml` or copy it and pass the copy with `--config`.
Command-line values override the corresponding YAML values.

`model.imgsz` controls the square inference canvas given to YOLO. Larger values
cost more compute and memory. The TrackBus ByteTrack profile keeps lost IDs for
60 frames so a briefly missed person can recover the same ID. It does not emit a
predicted detection while the person is absent. At 25 FPS the buffer represents
about 2.4 seconds; longer retention can increase incorrect associations.

Zone points are normalized `[x, y]` coordinates:

- `[0, 0]` is the top-left of the image;
- `[1, 1]` is the bottom-right;
- `[0.5, 0.5]` is the center.

Normalized coordinates let one configuration scale to different resolutions,
but the polygons still need calibration for the actual camera position. The
default assumes OUTSIDE is near the top and INSIDE near the bottom. Swap or edit
them if the camera orientation is reversed. Keep the polygons separated so the
gap acts as a transition area.

`minimum_zone_frames` filters brief zone touches. Raising it is more conservative
but may miss very fast crossings. `stale_track_timeout` controls how long old ID
state is retained, in video frames.

## Outputs

For an output named `result.mp4`, null log paths in the default configuration
produce:

- `result.mp4` — annotated video;
- `result.events.csv` — one row per confirmed IN/OUT event;
- `result.summary.json` — totals and processing statistics.

The summary records the source resolution and FPS, resolved model settings,
per-frame tracked person detections, frames with and without detections, maximum
simultaneous detections, and the number of unique tracking IDs. These diagnostics
help compare runs but do not measure recall or prove passenger-count accuracy.

The CSV timestamp is the position in the recorded video, not wall-clock time.
Actual occupancy is preserved above capacity and marked `OVER CAPACITY`. The
occupancy estimate is clamped at zero if observed exits exceed known occupancy.

## Tests and code checks

Tests use synthetic zones and IDs and do not load YOLO or download weights:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

## Current limitations

- Results depend strongly on camera angle, lighting, occlusion, and zone setup.
- ByteTrack IDs can change after long occlusion, which can affect counts.
- The generic pretrained model is not tuned for crowded overhead bus footage.
- Starting occupancy must be provided; v0.1.1 has no external source of truth.
- Recorded-video processing does not react to a physical door opening or closing.
- Output uses OpenCV's broadly available `mp4v` encoder; codec support varies by OS.

## Privacy

TrackBus v0.1.1 performs ordinary person detection only. It does **not** implement
face recognition, identity recognition, biometrics, cross-journey
re-identification, or passenger image crops. It stores count events and an
annotated version of the input video. Real deployments would still require an
appropriate retention policy, access controls, signage, and legal review.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md). Later work should begin with real-bus
validation and measurement of counting accuracy before adding integrations.
