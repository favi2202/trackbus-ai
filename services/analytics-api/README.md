# Analytics API

This is the first production service boundary. Sensor vendors post the shared
passenger-count contract here; operator and passenger applications consume
occupancy forecasts through versioned endpoints.

Run locally:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn trackbus_analytics.main:app --reload
```

The baseline model is deliberately transparent. A trained model should replace
it only after offline evaluation demonstrates better error and calibration.
