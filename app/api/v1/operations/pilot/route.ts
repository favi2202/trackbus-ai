import { pilotSnapshot } from "@/db/passenger-events";

export async function GET() {
  try {
    return Response.json(await pilotSnapshot());
  } catch {
    return Response.json(
      {
        dataMode: "unavailable",
        summary: { status: "storage-unavailable", eventCount: 0, busCount: 0, routeCount: 0, flaggedEventCount: 0, staleBusCount: 0 },
        sources: [],
        reconciliation: null,
        buses: [],
        recentEvents: [],
        refreshedAt: new Date().toISOString(),
      },
      { status: 503 },
    );
  }
}
