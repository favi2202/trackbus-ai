import type { PassengerCountEvent } from "@/lib/contracts";
import { getIngestKey, persistPassengerEvent, recentPassengerEvents } from "@/db/passenger-events";

const sources = new Set(["apc", "vision", "payment", "manual", "import"]);
const requiredStrings = ["eventId", "observedAt", "busId", "routeId", "stopId", "doorId"] as const;
const requiredNumbers = ["boardings", "alightings", "occupancy", "capacity", "confidence"] as const;

function valid(event: PassengerCountEvent | null): event is PassengerCountEvent {
  return Boolean(
    event &&
    event.schemaVersion === "1.0" &&
    Number.isFinite(Date.parse(event.observedAt)) &&
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

  const ingestKey = getIngestKey();
  if (!ingestKey) {
    return Response.json(
      {
        accepted: false,
        persisted: false,
        retryable: true,
        eventId: event.eventId,
        error: "Camera ingestion is not configured for this deployment",
      },
      { status: 503 },
    );
  }
  if (request.headers.get("authorization") !== `Bearer ${ingestKey}`) {
    return Response.json(
      { accepted: false, persisted: false, retryable: false, eventId: event.eventId, error: "Unauthorized camera source" },
      { status: 401 },
    );
  }

  try {
    const result = await persistPassengerEvent(event);
    return Response.json(
      {
        accepted: true,
        persisted: true,
        duplicate: result.duplicate,
        eventId: event.eventId,
        receivedAt: result.receivedAt,
      },
      { status: result.duplicate ? 200 : 202 },
    );
  } catch {
    return Response.json(
      {
        accepted: false,
        persisted: false,
        retryable: true,
        eventId: event.eventId,
        error: "TrackBus event storage is unavailable",
      },
      { status: 503 },
    );
  }
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const parsedLimit = Number.parseInt(url.searchParams.get("limit") ?? "25", 10);
  const limit = Number.isFinite(parsedLimit) ? parsedLimit : 25;
  const busId = url.searchParams.get("busId")?.trim() || undefined;
  try {
    const events = await recentPassengerEvents(limit, busId);
    return Response.json({
      dataMode: events.length ? "live" : "live-empty",
      count: events.length,
      events,
      refreshedAt: new Date().toISOString(),
    });
  } catch {
    return Response.json(
      { dataMode: "unavailable", count: 0, events: [], error: "TrackBus event storage is unavailable" },
      { status: 503 },
    );
  }
}
