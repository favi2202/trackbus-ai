import { env } from "cloudflare:workers";

export async function GET() {
  const storageConfigured = Boolean((env as unknown as { DB?: D1Database }).DB);
  return Response.json({
    service: "trackbus-showcase-gateway",
    status: "ok",
    version: "0.5.0",
    dataMode: storageConfigured ? "connected" : "unavailable",
    storageConfigured,
  });
}
