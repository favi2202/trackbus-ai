export async function GET() {
  return Response.json({
    service: "trackbus-showcase-gateway",
    status: "ok",
    version: "0.1.0",
    dataMode: "synthetic",
  });
}
