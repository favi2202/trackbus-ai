# TrackBus roadmap

## v0.1 — recorded-video prototype

- [x] Pretrained person detection
- [x] ByteTrack integration and temporary IDs
- [x] Configurable two-zone transition counting
- [x] Annotated video, CSV events, and JSON summary
- [x] Model-independent unit tests and setup documentation

## v0.1.1 — low-resolution diagnostics and tuning

- [x] Configurable YOLO inference image size
- [x] CLI precedence tests for model, confidence, and image size
- [x] ByteTrack profile for short detection misses
- [x] Detection and tracking diagnostics in JSON summaries
- [x] CPU, GPU, and low-resolution comparison presets

## Next: field validation

- Collect consented, privacy-reviewed sample footage from representative bus doors.
- Define an annotation protocol and measure precision, recall, and counting error.
- Calibrate zones, confirmation time, confidence, and tracker settings.
- Test crowds, occlusion, bags, children, lighting changes, and camera vibration.
- Decide acceptable accuracy and operational failure behavior.

## Later versions

- Start and stop processing from a real door-state signal.
- Support live camera input and robust service supervision.
- Evaluate a bus-specific model only if field measurements justify training.
- Add privacy retention controls and operational monitoring.
- Design backend or passenger-facing integrations only after the counter is proven.

This roadmap intentionally defers APIs, databases, dashboards, cloud deployment,
maps, forecasting, CAN-bus work, and mobile applications until the core counting
assumptions have been validated.
