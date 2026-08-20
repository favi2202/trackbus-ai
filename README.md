# TrackBus AI

TrackBus is a vendor-neutral transport-intelligence platform for Tashkent. It
normalizes anonymous passenger-count events from approved APC sensors, optional
TrackBus Vision cameras, payments, manual corrections, and imports into current
occupancy, demand forecasts, and auditable operator recommendations.

> The hosted dashboard is a **synthetic showcase**, not a production accuracy
> claim. Vision uses temporary anonymous track IDs, performs no facial
> recognition, and keeps raw video on the edge by default.

## What works now

- Pilot Proof operator dashboard with source health, reconciliation, validation
  gaps, privacy boundaries, and the 60-day pilot sequence;
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

The dashboard labels all built-in values as synthetic. Its web gateway validates
events and forwards them only when `TRACKBUS_ANALYTICS_API_URL` is configured;
it never claims persistence while the analytics service is unavailable.

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

Useful overrides include `--model`, `--confidence`, `--imgsz`, `--device`,
`--frame-skip`, `--tracker`, `--zones`, `--api-url`, `--bus-id`, `--route-id`,
`--stop-id`, and `--door-id`. The default zones are a safe presentation
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
[pilot runbook](docs/pilot-runbook.md) before field use.
