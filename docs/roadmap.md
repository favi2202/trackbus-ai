# Foundation roadmap

## Milestone 01 — repository foundation (current)

- working operator, forecasting, pipeline and passenger showcase;
- versioned passenger-count event contract;
- ingestion and forecasting API boundaries;
- transparent, testable baseline forecast;
- synthetic Tashkent scenarios with clear labeling.

## Milestone 02 — pilot integration

- obtain one approved bus/sensor data sample;
- build the vendor adapter and replay harness;
- add PostgreSQL/Timescale storage and data-quality monitoring;
- calibrate capacity and count corrections with field observations.

## Milestone 03 — network forecasting

- train route/stop/time models on approved historical data;
- evaluate MAE, overload recall and calibration against the baseline;
- expose forecasts through a documented partner API;
- add operator feedback and audit logs for recommendations.

## Milestone 04 — controlled rollout

- deploy to a limited route group;
- compare passenger wait time, overcrowding and vehicle utilization;
- complete security, privacy and operational reviews;
- expand only after measurable results.
