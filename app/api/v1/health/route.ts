export async function GET() {
  const analyticsConfigured = Boolean(process.env.TRACKBUS_ANALYTICS_API_URL);
  return Response.json({
    service: "trackbus-showcase-gateway",
    status: "ok",
    version: "0.4.0",
    dataMode: analyticsConfigured ? "connected" : "synthetic",
    analyticsConfigured,
  });
}
