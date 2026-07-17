import type { PassengerCountEvent } from "@/lib/contracts";

const requiredStringFields = ["eventId", "observedAt", "busId", "routeId", "stopId"] as const;
const requiredNumberFields = ["boardings", "alightings", "occupancy", "capacity", "qualityScore"] as const;

export async function POST(request: Request) {
  const event = (await request.json().catch(() => null)) as PassengerCountEvent | null;
  const stringsValid = event && requiredStringFields.every((field) => typeof event[field] === "string" && event[field].length > 0);
  const numbersValid = event && requiredNumberFields.every((field) => typeof event[field] === "number" && event[field] >= 0);

  if (!event || !stringsValid || !numbersValid || event.occupancy > event.capacity || event.qualityScore > 1) {
    return Response.json(
      { accepted: false, error: "Event does not match the passenger-count contract" },
      { status: 422 },
    );
  }

  return Response.json(
    { accepted: true, eventId: event.eventId, mode: "showcase-no-persistence" },
    { status: 202 },
  );
}
