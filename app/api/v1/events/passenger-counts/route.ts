import type { PassengerCountEvent } from "@/lib/contracts";

const sources = new Set(["apc", "vision", "payment", "manual", "import"]);
const requiredStrings = ["eventId", "observedAt", "busId", "routeId", "stopId", "doorId"] as const;
const requiredNumbers = ["boardings", "alightings", "occupancy", "capacity", "confidence"] as const;

function valid(event: PassengerCountEvent | null): event is PassengerCountEvent {
  return Boolean(
    event &&
    event.schemaVersion === "1.0" &&
    sources.has(event.source) &&
    requiredStrings.every((field) => typeof event[field] === "string" && event[field].length > 0) &&
    requiredNumbers.every((field) => typeof event[field] === "number" && Number.isFinite(event[field]) && event[field] >= 0) &&
    Number.isInteger(event.boardings) &&
    Number.isInteger(event.alightings) &&
    Number.isInteger(event.occupancy) &&
    Number.isInteger(event.capacity) &&
    event.capacity > 0 &&
    event.occupancy <= event.capacity &&
    event.confidence <= 1 &&
    Array.isArray(event.qualityFlags) &&
    event.qualityFlags.every((flag) => typeof flag === "string" && flag.length > 0),
  );
}

export async function POST(request: Request) {
  const event = (await request.json().catch(() => null)) as PassengerCountEvent | null;
  if (!valid(event)) {
    return Response.json(
      { accepted: false, persisted: false, error: "Event does not match the canonical passenger-count contract" },
      { status: 422 },
    );
  }

  const analyticsUrl = process.env.TRACKBUS_ANALYTICS_API_URL?.replace(/\/$/, "");
  if (!analyticsUrl) {
    return Response.json(
      {
        accepted: false,
        persisted: false,
        retryable: true,
        eventId: event.eventId,
        error: "Analytics persistence is not configured for this showcase deployment",
      },
      { status: 503 },
    );
  }

  try {
    const upstream = await fetch(`${analyticsUrl}/v1/events/passenger-counts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(event),
      signal: AbortSignal.timeout(5000),
    });
    const payload = await upstream.json().catch(() => ({
      accepted: false,
      persisted: false,
      error: "Analytics service returned an invalid response",
    }));
    return Response.json(payload, { status: upstream.status });
  } catch {
    return Response.json(
      {
        accepted: false,
        persisted: false,
        retryable: true,
        eventId: event.eventId,
        error: "Analytics service is unavailable",
      },
      { status: 502 },
    );
  }
}
