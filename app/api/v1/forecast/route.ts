import { forecastOccupancy } from "@/lib/forecast";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  if (
    !body ||
    !Array.isArray(body.recentOccupancy) ||
    !body.recentOccupancy.every((value) => typeof value === "number" && Number.isFinite(value)) ||
    typeof body.capacity !== "number" ||
    typeof body.hour !== "number"
  ) {
    return Response.json(
      { error: "recentOccupancy, capacity and hour are required" },
      { status: 400 },
    );
  }
  return Response.json(
    forecastOccupancy({
      recentOccupancy: body.recentOccupancy as number[],
      capacity: body.capacity,
      hour: body.hour,
      dayType: body.dayType === "weekend" ? "weekend" : "weekday",
      weather: body.weather === "rain" ? "rain" : "clear",
      eventNearby: Boolean(body.eventNearby),
    }),
  );
}
