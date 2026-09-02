import { networkRoutes, passengerEvents, route22Buses } from "@/lib/sample-data";

export async function GET() {
  return Response.json({
    generatedAt: "2026-07-18T17:32:00+05:00",
    dataMode: "synthetic",
    routes: networkRoutes,
    buses: route22Buses,
    latestEvents: passengerEvents,
  });
}
