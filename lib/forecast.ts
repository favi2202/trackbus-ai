export type ForecastInput = {
  recentOccupancy: number[];
  capacity: number;
  hour: number;
  dayType: "weekday" | "weekend";
  weather?: "clear" | "rain";
  eventNearby?: boolean;
};

export type ForecastResult = {
  expectedOccupancy: number;
  expectedPercent: number;
  lowerBound: number;
  upperBound: number;
  confidence: number;
};

/**
 * Transparent baseline for the first pilot. It is intentionally simple and
 * measurable; later ML models must beat it before being promoted.
 */
export function forecastOccupancy(input: ForecastInput): ForecastResult {
  const history = input.recentOccupancy.slice(-6);
  const weighted = history.reduce(
    (sum, value, index) => sum + value * (index + 1),
    0,
  );
  const divisor = history.reduce((sum, _, index) => sum + index + 1, 0) || 1;
  const base = weighted / divisor;
  const rushHourFactor =
    input.dayType === "weekday" &&
    ((input.hour >= 7 && input.hour <= 9) || (input.hour >= 17 && input.hour <= 20))
      ? 1.13
      : 1;
  const weatherFactor = input.weather === "rain" ? 1.07 : 1;
  const eventFactor = input.eventNearby ? 1.1 : 1;
  const expected = Math.min(
    input.capacity,
    Math.max(0, Math.round(base * rushHourFactor * weatherFactor * eventFactor)),
  );
  const spread = Math.max(3, Math.round(input.capacity * 0.08));

  return {
    expectedOccupancy: expected,
    expectedPercent: Math.round((expected / input.capacity) * 100),
    lowerBound: Math.max(0, expected - spread),
    upperBound: Math.min(input.capacity, expected + spread),
    confidence: history.length >= 6 ? 0.86 : 0.68,
  };
}
