export type OccupancyLevel = "low" | "medium" | "high" | "critical";

export type PassengerCountEvent = {
  eventId: string;
  observedAt: string;
  busId: string;
  routeId: string;
  stopId: string;
  boardings: number;
  alightings: number;
  occupancy: number;
  capacity: number;
  source: "apc-camera" | "manual" | "simulator";
  qualityScore: number;
};

export type BusSnapshot = {
  id: string;
  routeId: string;
  label: string;
  occupancy: number;
  capacity: number;
  etaMinutes: number;
  latitude: number;
  longitude: number;
  trend: "up" | "down" | "steady";
};

export const occupancyLevel = (occupancy: number, capacity: number): OccupancyLevel => {
  const ratio = capacity === 0 ? 0 : occupancy / capacity;
  if (ratio >= 0.9) return "critical";
  if (ratio >= 0.72) return "high";
  if (ratio >= 0.45) return "medium";
  return "low";
};

export const occupancyPercent = (occupancy: number, capacity: number) =>
  Math.round(capacity === 0 ? 0 : (occupancy / capacity) * 100);
