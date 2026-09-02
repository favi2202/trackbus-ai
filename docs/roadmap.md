# Evidence-first roadmap

## Completed foundation

- hosted synthetic operator and passenger showcase;
- canonical event, durable ingestion, forecasting, and operations APIs;
- v0.2.1 YOLO/ByteTrack separation, calibration, diagnostics, and evaluation;
- conservative temporary-ID doorway events and offline delivery boundary;
- Pilot Proof framing with source health, reconciliation, and privacy controls.

## 60-day pilot

1. **Install and calibrate:** select one approved bus/door, document the source,
   calibrate zones, and confirm privacy/retention controls.
2. **Collect and label:** capture consented representative door cycles and label
   completed IN/OUT event frames.
3. **Tune AI and API:** measure by camera, lighting, crowding, occlusion, bags,
   children, vibration, and simultaneous crossings; tune on calibration footage.
4. **Pilot report:** publish measured precision, recall, event F1, count error,
   timing error, failure modes, uptime, and per-bus cost. Do not invent accuracy
   or ROI claims.

## Network forecasting

- replace SQLite only when measured volume requires it;
- train route/stop/time models on approved history;
- compare MAE, overload recall, and calibration against the transparent baseline;
- add operator feedback and recommendation audit logs.

## Controlled rollout

- deploy to a limited route group;
- compare wait time, crowding, and vehicle utilization;
- complete security, privacy, and operational reviews;
- expand only after the pilot meets documented acceptance criteria.
