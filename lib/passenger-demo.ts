import { forecastOccupancy } from "@/lib/forecast";
import {
  aggregateCrowdEstimate,
  crowdLevelForPercent,
  type PassengerCrowdReport,
} from "@/lib/crowding";

export type PassengerBusOption = {
  busId: string;
  routeId: string;
  etaMinutes: number;
  capacity: number;
  occupancy: number;
  sensorConfidence: number;
  recentOccupancy: number[];
};

export const passengerDemoBuses: PassengerBusOption[] = [
  {
    busId: "22-04",
    routeId: "22",
    etaMinutes: 3,
    capacity: 90,
    occupancy: 76,
    sensorConfidence: 0.88,
    recentOccupancy: [62, 66, 69, 72, 75, 77],
  },
  {
    busId: "22-07",
    routeId: "22",
    etaMinutes: 8,
    capacity: 90,
    occupancy: 29,
    sensorConfidence: 0.84,
    recentOccupancy: [29, 34, 38, 42, 44, 45],
  },
];

export function createDemoPassengerReports(now: Date): PassengerCrowdReport[] {
  const minutesAgo = (minutes: number) => new Date(now.getTime() - minutes * 60_000).toISOString();
  return [
    { reportId: "rpt-demo-1", busId: "22-04", routeId: "22", timestamp: minutesAgo(2), crowdLevel: "crowded", seatsAvailable: "no", source: "passenger_report" },
    { reportId: "rpt-demo-2", busId: "22-04", routeId: "22", timestamp: minutesAgo(5), crowdLevel: "crowded", seatsAvailable: "few", source: "passenger_report" },
    { reportId: "rpt-demo-3", busId: "22-04", routeId: "22", timestamp: minutesAgo(8), crowdLevel: "crowded", seatsAvailable: "no", source: "passenger_report" },
    { reportId: "rpt-demo-4", busId: "22-04", routeId: "22", timestamp: minutesAgo(14), crowdLevel: "moderate", seatsAvailable: "few", source: "passenger_report" },
    { reportId: "rpt-demo-5", busId: "22-07", routeId: "22", timestamp: minutesAgo(4), crowdLevel: "plenty", seatsAvailable: "yes", source: "passenger_report" },
    { reportId: "rpt-demo-6", busId: "22-07", routeId: "22", timestamp: minutesAgo(9), crowdLevel: "seats_available", seatsAvailable: "yes", source: "passenger_report" },
  ];
}

export function estimatePassengerBus(
  bus: PassengerBusOption,
  allReports: PassengerCrowdReport[],
  now: Date,
) {
  const reports = allReports.filter(report => report.busId === bus.busId);
  const sensorPercent = Math.round((bus.occupancy / bus.capacity) * 100);
  const current = aggregateCrowdEstimate({
    sensorPercent,
    sensorConfidence: bus.sensorConfidence,
    reports,
    now,
  });
  const forecast = forecastOccupancy({
    recentOccupancy: bus.recentOccupancy,
    capacity: bus.capacity,
    hour: 18,
    dayType: "weekday",
  });
  const communityAdjustment = Math.round((current.estimatedPercent - sensorPercent) * 0.4);
  const predictedPercent = Math.min(100, Math.max(0, forecast.expectedPercent + communityAdjustment));

  return {
    ...bus,
    sensorPercent,
    current,
    predictedPercent,
    predictedLevel: crowdLevelForPercent(predictedPercent),
    predictedConfidence: Math.min(0.96, forecast.confidence + current.communityInfluence * 0.25),
  };
}
