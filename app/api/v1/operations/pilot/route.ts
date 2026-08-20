const syntheticPilot = {
  dataMode: "synthetic",
  summary: {
    status: "demonstration",
    eventCount: 1842,
    busCount: 5,
    routeCount: 1,
    flaggedEventCount: 23,
    staleBusCount: 1,
  },
  sources: [
    { source: "apc", status: "healthy", coverage: "3 buses", latest: "4 sec ago" },
    { source: "vision", status: "pilot", coverage: "1 doorway", latest: "8 sec ago" },
    { source: "payment", status: "healthy", coverage: "Route 22", latest: "1 min ago" },
  ],
  reconciliation: {
    routeId: "22",
    sensorBoardings: 1264,
    paymentBoardings: 1238,
    boardingGap: 26,
    status: "needs-review",
  },
};

async function upstream(path: string, base: string) {
  const response = await fetch(`${base}${path}`, { signal: AbortSignal.timeout(4000) });
  if (!response.ok) throw new Error(`analytics HTTP ${response.status}`);
  return response.json();
}

export async function GET() {
  const base = process.env.TRACKBUS_ANALYTICS_API_URL?.replace(/\/$/, "");
  if (!base) return Response.json(syntheticPilot);
  try {
    const [summary, health, reconciliation] = await Promise.all([
      upstream("/v1/operations/summary", base),
      upstream("/v1/operations/source-health", base),
      upstream("/v1/operations/reconciliation", base),
    ]);
    return Response.json({
      dataMode: "live",
      summary,
      sources: health.sources,
      reconciliation: reconciliation.routes[0] ?? null,
    });
  } catch {
    return Response.json(
      { ...syntheticPilot, dataMode: "fallback-synthetic", upstreamStatus: "unavailable" },
      { status: 200 },
    );
  }
}
