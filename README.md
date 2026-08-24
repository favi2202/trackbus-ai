# TrackBus AI

TrackBus is a vendor-neutral transport-intelligence platform for Tashkent. It
normalizes anonymous passenger-count events from approved APC sensors, optional
TrackBus Vision cameras, payments, manual corrections, and imports into current
occupancy, demand forecasts, and auditable operator recommendations.

> The dashboard's Live pilot tab shows stored camera events. Forecast and
> dispatch scenarios remain clearly labeled synthetic demonstrations, not
> production accuracy claims. Vision uses temporary anonymous track IDs,
> performs no facial recognition, and keeps raw video on the edge by default.

## What works now

- Live Pilot operator dashboard with D1-backed source health, current bus
  occupancy, recent JSON events, reconciliation, and two-second refresh;
- vendor-neutral passenger-count contract and durable idempotent analytics API;
- real Ultralytics YOLO + ByteTrack Vision pipeline and v0.2.1 calibration,
  diagnostics, and evaluation tools;
- explicit `OUTSIDE -> DOOR -> INSIDE` / reverse crossing state machine;
- webcam, video, and discoverable demo-footage modes in `showcase.py`;
- API delivery with retry-safe disk queue when connectivity is unavailable;
- forecasting, quality, replay, web, API, Vision, and trajectory tests.

## Repository map

```text
app/                       Hosted showcase and gateway API
components/                Operator, pilot proof, and passenger interface
contracts/                 Canonical vendor-neutral event schema
services/analytics-api/    Durable ingestion, quality, and operations API
trackbus/                   Vision detection, tracking, counting, and evaluation
showcase.py                 Executable camera/video presentation demo
configs/                    Vision, calibration, tracker, and experiment settings
tools/                      Sensor replay utilities
docs/                       Architecture, pilot, privacy, and showcase guides
```

## Hosted showcase

```bash
npm ci
npm run dev
```

The hosted gateway validates authenticated camera events and stores them
idempotently in Cloudflare D1. View the newest stored records directly at:

```text
https://trackbus-showcase.favi-2202.chatgpt.site/api/v1/events/passenger-counts?limit=50
```

The Live pilot tab polls the same source every two seconds. Forecast, dispatch,
and passenger-app scenarios remain labeled synthetic.

For a separate free `workers.dev` deployment, follow the
[Cloudflare Workers deployment guide](docs/cloudflare-deployment.md).

## Real Vision showcase

Install Python 3.11+ dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
```

Run a source:

```bash
python showcase.py --camera 0
python showcase.py --video path/to/video.mp4
python showcase.py --demo
python showcase.py --calibrate --camera 0
```

Send confirmed crossings to the hosted showcase (PowerShell):

```powershell
$env:TRACKBUS_API_KEY="<camera ingestion key>"
python showcase.py --camera 0 --api-url https://trackbus-showcase.favi-2202.chatgpt.site/api
```

On macOS or Linux, use `export TRACKBUS_API_KEY="<camera ingestion key>"`.

Useful overrides include `--model`, `--confidence`, `--detector-floor`, `--imgsz`,
`--device`, `--frame-skip`, `--tracker`, `--zones`, `--api-url`, `--api-key`,
`--bus-id`, `--route-id`, `--stop-id`, and `--door-id`. Prefer the
`TRACKBUS_API_KEY` environment variable over `--api-key` so the key does not
appear in shell history. The default zones are a safe presentation
starting point; a real camera must be calibrated for its doorway geometry.

Keyboard controls:

| Key | Action |
| --- | --- |
| `Space` | Pause or resume |
| `F` | Toggle fullscreen |
| `O` | Toggle the information overlay |
| `R` | Reset counters after confirmation |
| `C` / `Z` | Show the auditable `--calibrate` command |
| `S` | Save a screenshot |
| `E` | Export the current event log to CSV |
| `Q` / `Esc` | Quit |

If the API is offline, canonical events are retained under
`data/offline-queue/` and retried without changing their event IDs. A missing
demo video exits with discovery instructions instead of opening an empty window.

## Measure Vision recognition

Separate YOLO detection quality from ByteTrack and IN/OUT logic before tuning:

```powershell
python -m trackbus.benchmark_detection `
  --video test_video.mp4 `
  --model yolo11s.pt `
  --imgsz 960 `
  --detector-floor 0.10

python -m trackbus.sweep_detection `
  --video test_video.mp4 `
  --matrix configs/detection_sweep.yaml `
  --output-dir data/experiments/detection-sweep
```

Without box-level person annotations these commands report detection coverage,
confidence, zero-detection gaps, edge clipping, stability proxies, FPS, and
latency—but deliberately do not claim precision, recall, F1, or a best accuracy
configuration. See the [detector benchmark guide](docs/detection-benchmark.md)
and [failure audit](docs/vision-failure-audit.md). Optional `low_light` and
`contrast` preprocessing profiles are bounded, same-size experiments and stay
off by default.

Combine completed detector, pipeline, and event-evaluation artifacts without
mixing metric meanings:

```powershell
python -m trackbus.benchmark_report `
  --detection data/experiments/detector/benchmark.json `
  --processing data/output/processing-summary.json `
  --counting data/output/counting-evaluation.json `
  --output data/experiments/separated-report.json
```

The JSON and CSV outputs label every non-ground-truth section as diagnostics,
not accuracy. See the [separated benchmark guide](docs/separated-benchmark.md).

After establishing a detector baseline, compare the bounded ByteTrack profiles
in `configs/tracking_fast.yaml`, `configs/tracking_balanced.yaml`, and
`configs/tracking_occlusion.yaml`. TrackBus also provides an experimental,
off-by-default short-gap continuity layer for conservative local ID stitching.
See the [tracking continuity guide](docs/tracking-continuity.md) before enabling
it on a camera.

Before the next representative video test, run the bounded camera-quality
analyzer and, when necessary, enable metadata-only failure mining. TrackBus now
passes detections from a separate `model.detector_floor` to ByteTrack so weak
observations may preserve an existing anonymous track without starting a new
one. See the
[vision readiness diagnostics guide](docs/vision-readiness-diagnostics.md).

```powershell
python -m trackbus.camera_quality `
  --source ".\videos\bus-test.mp4" `
  --config ".\configs\default.yaml" `
  --output ".\data\output\bus-test.camera-quality.json"
```

## Verification

```bash
npm run lint
npm test

cd services/analytics-api
python -m pip install -e '.[dev]'
ruff check src tests
pytest -q

cd ../..
python -m pytest -q
PYTHONPATH=tools python -m unittest discover -s tools/tests -v
python showcase.py --help
```

See [the showcase guide](docs/showcase.md), [architecture](docs/architecture.md),
[privacy boundaries](docs/privacy-and-safety.md), and
[pilot runbook](docs/pilot-runbook.md) before field use. For a future local
training dataset, begin with the privacy-conscious
[dataset preparation workflow](docs/dataset-preparation.md).
