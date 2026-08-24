# TrackBus Vision showcase

`showcase.py` is an executable demonstration of the real edge pipeline, not a
pre-rendered or fake-detection overlay. It calls Ultralytics person detection,
advances one ByteTrack instance, assigns temporary IDs, classifies a configurable
anchor in three polygons, and emits an event only for a stable full path.

```text
OUTSIDE -> DOOR -> INSIDE = IN / BOARDING / +1
INSIDE  -> DOOR -> OUTSIDE = OUT / ALIGHTING / -1
```

Incomplete, direct, noisy, expired, and reversed paths do not count.

## Commands

```bash
python showcase.py --camera 0
python showcase.py --source 0
python showcase.py --video data/input/door.mp4
python showcase.py --demo
python showcase.py --calibrate --camera 0 --save-zones configs/showcase_zones.yaml
python showcase.py --video door.mp4 --zones configs/showcase_zones.yaml --api-url http://localhost:8000
```

To connect a Windows camera directly to the hosted Live pilot dashboard:

```powershell
$env:TRACKBUS_API_KEY="<camera ingestion key>"
python showcase.py --camera 0 --api-url https://trackbus-showcase.favi-2202.chatgpt.site/api
```

Each confirmed crossing is POSTed as canonical JSON, stored under its `eventId`,
and appears in the dashboard within about two seconds. Inspect the raw records at
`/api/v1/events/passenger-counts?limit=50`. The API key is a bearer credential;
keep it out of source files and prefer the environment variable to `--api-key`.

Use `--model`, `--imgsz`, `--confidence`, `--detector-floor`, `--device`,
`--frame-skip`, and `--tracker` to match the edge hardware. The detector floor
defaults to `0.10`; weak detections may maintain an existing anonymous track but
cannot start one. CUDA requires a CUDA-enabled PyTorch install; use `--device
cpu` when it is unavailable. Ultralytics `8.4.95` and `lap` are declared
dependencies because the ByteTrack adapter is version-tested.

## Controls

- `Space`: pause/resume
- `F`: fullscreen
- `O`: overlay
- `R`: clear temporary tracks and counters
- `S`: screenshot
- `E`: CSV event export
- `C` or `Z`: remind the operator to run auditable calibration
- `Q` or `Esc`: quit

When `--api-url` is absent or unreachable, every canonical event is written
atomically to `data/offline-queue/` under the same idempotency key. The client
flushes queued events on the next connected start. `--demo` searches only local
approved media and exits with a useful instruction when no footage exists.

For non-graphical verification, use `--headless --max-frames 3`. This still loads
the real model and tracker; it only suppresses the OpenCV presentation window.
