# TrackBus AI

TrackBus is a transport-intelligence platform for Tashkent. It converts
anonymous passenger counts into current occupancy, demand forecasts and
auditable recommendations for bus operators, while giving passengers simple
crowding information.

> The current dashboard is explicitly a **demo for showcasing**. It uses
> synthetic data. The repository around it is the foundation of the real
> platform: contracts, service boundaries, tests, deployment scaffolding and a
> staged roadmap.

## What works now

- interactive operator dashboard with live-demo playback;
- occupancy and scenario-based forecast views;
- end-to-end data-pipeline explanation and synthetic event stream;
- passenger-facing crowding view;
- versioned JSON passenger-count contract;
- web gateway endpoints under `/api/v1`;
- independent Python analytics API with validation and forecast tests.
- idempotent SQLite event storage with bus/route queries;
- replayable sensor-event tooling and explainable quality checks.

## Repository map

```text
app/                       Hosted showcase and gateway API
components/                Interactive TrackBus interface
contracts/                 Vendor-neutral sensor event schemas
data/sample/               Synthetic, reproducible input events
demo_for_showcasing/       Demo scope and presentation notes
docs/                      Architecture, roadmap, data and privacy decisions
infra/                     Local service orchestration
lib/                       Shared TypeScript domain logic
services/analytics-api/    Production ingestion and forecasting boundary
tools/                     Safe-by-default sensor replay utilities
```

## Run the showcase

```bash
npm ci
npm run dev
```

## Run the analytics tests

```bash
cd services/analytics-api
PYTHONPATH=src python -m unittest discover -s tests
```

Read `docs/roadmap.md` for the pilot sequence. The next practical dependency is
one approved, documented sample payload from an actual Tashkent bus system.
