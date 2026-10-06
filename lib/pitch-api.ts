import type { PassengerCountEvent } from "./contracts";

export const examplePassengerEvent: PassengerCountEvent = {
  schemaVersion: "1.0",
  eventId: "demo-evt-001",
  observedAt: "2026-10-06T07:00:00Z",
  source: "vision",
  busId: "BUS-2204",
  routeId: "22",
  stopId: "CHORSU",
  doorId: "DOOR-1",
  boardings: 3,
  alightings: 1,
  occupancy: 42,
  capacity: 72,
  confidence: 0.8,
  qualityFlags: ["synthetic_demo"],
};

export const apiEndpoints = [
  { method: "GET", path: "/api/v1/health", title: "Gateway health", detail: "Service status and whether D1 storage is configured. This is not a camera accuracy check.", publicUrl: "/api/v1/health" },
  { method: "GET", path: "/api/v1/network", title: "Showcase network", detail: "Synthetic routes, vehicle loads and event examples. The response explicitly labels dataMode as synthetic.", publicUrl: "/api/v1/network" },
  { method: "GET", path: "/api/v1/operations/pilot", title: "Pilot feed", detail: "Stored camera events, source health and occupancy snapshots. Empty or unavailable storage is reported honestly.", publicUrl: "/api/v1/operations/pilot" },
  { method: "GET", path: "/api/v1/events/passenger-counts?limit=10", title: "Recent count events", detail: "Read anonymous events from D1. Supports the busId query parameter.", publicUrl: "/api/v1/events/passenger-counts?limit=10" },
  { method: "POST", path: "/api/v1/events/passenger-counts", title: "Count ingestion", detail: "Validate and idempotently persist the canonical event. Requires a configured device bearer key. New events return 202; duplicate retries return 200.", publicUrl: null },
  { method: "POST", path: "/api/v1/forecast", title: "Baseline forecast", detail: "Provide recentOccupancy, capacity, hour and optional context. Returns bounded occupancy estimates with heuristic confidence and intervals.", publicUrl: null },
] as const;
