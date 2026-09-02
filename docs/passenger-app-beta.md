# Passenger App beta

The Passenger App section demonstrates one TrackBus product question: **how crowded will the bus probably be when it reaches my stop?**

## Passenger crowd model

`lib/crowding.ts` owns the shared five-level model and configurable boundaries:

| Estimated occupancy | Passenger status |
| --- | --- |
| 0–34% | Plenty of space |
| 35–59% | Seats likely available |
| 60–79% | Moderate |
| 80–94% | Crowded |
| 95–100% | Very crowded / full |

Every level includes an ID, percentage range, severity, icon, and visual tone. Passenger screens lead with the text status and keep the percentage as secondary evidence.

## Forecast at the selected stop

`lib/passenger-demo.ts` adapts the existing transparent baseline forecast. The synthetic Route 22 scenario compares:

- Bus `22-04`: 3 minutes away, 84% now, forecast 91% at the stop.
- Bus `22-07`: 8 minutes away, 32% now, forecast 52% at the stop.

When both forecasts have sufficient confidence, TrackBus explains that waiting five additional minutes is expected to provide more space. Low-confidence estimates do not produce that recommendation.

## Community reports

The normalized `PassengerCrowdReport` contains only a report ID, bus and route IDs, timestamp, crowd level, optional seat availability, and `passenger_report` source. It contains no account, name, face, persistent passenger ID, or free text.

Report influence decays with age:

- 0–3 minutes: strong
- 3–10 minutes: medium
- 10–20 minutes: weak
- 20–30 minutes: fading
- 30+ minutes: ignored for the current estimate

One fresh report contributes at most 5% of the combined estimate. Several recent reports may contribute up to 18%. Sensor/APC data therefore remains the primary signal.

## Demo boundaries

- Route, vehicle, history, reports, and predictions are synthetic.
- Rider submissions are browser-session demo events and are not persisted.
- Forecast confidence is a prototype baseline, not validated production accuracy.
- Abuse prevention, rate limiting, device attestation, and moderation remain future production work.
- The component consumes typed interfaces so demo sources can later be replaced with a real passenger API, PWA, Android/iOS client, or approved map integration.

## Verification

```powershell
npm.cmd run lint
npx.cmd tsc --noEmit
npm.cmd test
```

The crowding tests cover threshold boundaries, configurable policies, time decay, stale-report removal, single-report influence, multi-report aggregation, forecast-to-status conversion, anonymous event shape, and low-confidence recommendation behavior.

To run the complete site locally:

```powershell
npm.cmd ci
npm.cmd run dev
```

Open the TrackBus dashboard and select **Passenger app**, then compare the two buses and submit a one-tap report.
