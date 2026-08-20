import type { BusSnapshot, PassengerCountEvent } from "./contracts";

export const route22Buses: BusSnapshot[] = [
  { id: "BUS-2201", routeId: "22", label: "22-01", occupancy: 31, capacity: 72, etaMinutes: 2, latitude: 41.326, longitude: 69.236, trend: "up" },
  { id: "BUS-2202", routeId: "22", label: "22-02", occupancy: 27, capacity: 72, etaMinutes: 6, latitude: 41.303, longitude: 69.269, trend: "down" },
  { id: "BUS-2203", routeId: "22", label: "22-03", occupancy: 38, capacity: 72, etaMinutes: 9, latitude: 41.296, longitude: 69.257, trend: "steady" },
  { id: "BUS-2204", routeId: "22", label: "22-04", occupancy: 66, capacity: 72, etaMinutes: 12, latitude: 41.337, longitude: 69.228, trend: "up" },
  { id: "BUS-2205", routeId: "22", label: "22-05", occupancy: 56, capacity: 72, etaMinutes: 16, latitude: 41.313, longitude: 69.244, trend: "up" },
];

export const passengerEvents: PassengerCountEvent[] = [
  { schemaVersion: "1.0", eventId: "evt-10041", observedAt: "2026-07-18T17:28:12+05:00", source: "apc", busId: "BUS-2204", routeId: "22", stopId: "CHORSU", doorId: "DOOR-1", boardings: 12, alightings: 3, occupancy: 66, capacity: 72, confidence: 0.96, qualityFlags: [] },
  { schemaVersion: "1.0", eventId: "evt-10042", observedAt: "2026-07-18T17:29:50+05:00", source: "vision", busId: "BUS-2205", routeId: "22", stopId: "NAVOI", doorId: "DOOR-1", boardings: 8, alightings: 5, occupancy: 56, capacity: 72, confidence: 0.94, qualityFlags: ["synthetic_demo"] },
  { schemaVersion: "1.0", eventId: "evt-10043", observedAt: "2026-07-18T17:31:04+05:00", source: "apc", busId: "BUS-2202", routeId: "22", stopId: "OYBEK", doorId: "DOOR-2", boardings: 2, alightings: 9, occupancy: 27, capacity: 72, confidence: 0.98, qualityFlags: [] },
];

export const networkRoutes = [
  { id: "22", name: "Chorsu — Do'stlik", buses: 8, average: 64, status: "attention" },
  { id: "67", name: "Yunusobod — Chilonzor", buses: 11, average: 47, status: "normal" },
  { id: "93", name: "Sergeli — Amir Temur", buses: 9, average: 71, status: "attention" },
  { id: "17", name: "Beruniy — Buyuk Ipak Yo'li", buses: 10, average: 36, status: "normal" },
];

export const hourlyForecast = [34, 38, 43, 51, 47, 55, 61, 58, 66, 64, 72, 79, 88, 91, 84, 76, 62, 48];
