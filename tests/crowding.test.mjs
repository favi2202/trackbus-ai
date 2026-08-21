import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateCrowdEstimate,
  crowdLevelForPercent,
  recommendPassengerChoice,
  reportTimeWeight,
} from "../lib/crowding.ts";
import { forecastOccupancy } from "../lib/forecast.ts";

const NOW = new Date("2026-08-21T12:00:00.000Z");
const report = (minutesAgo, crowdLevel = "crowded") => ({
  reportId: `rpt-${minutesAgo}-${crowdLevel}`,
  busId: "22-04",
  routeId: "22",
  timestamp: new Date(NOW.getTime() - minutesAgo * 60_000).toISOString(),
  crowdLevel,
  seatsAvailable: "few",
  source: "passenger_report",
});

test("maps configurable occupancy boundaries to five passenger crowd levels", () => {
  assert.equal(crowdLevelForPercent(0).id, "plenty");
  assert.equal(crowdLevelForPercent(34.99).id, "plenty");
  assert.equal(crowdLevelForPercent(35).id, "seats_available");
  assert.equal(crowdLevelForPercent(59.99).id, "seats_available");
  assert.equal(crowdLevelForPercent(60).id, "moderate");
  assert.equal(crowdLevelForPercent(80).id, "crowded");
  assert.equal(crowdLevelForPercent(95).id, "full");
  assert.equal(crowdLevelForPercent(120).id, "full");
});

test("supports a configurable threshold policy", () => {
  const custom = { plentyMax: 30, seatsAvailableMax: 55, moderateMax: 75, crowdedMax: 90 };
  assert.equal(crowdLevelForPercent(32, custom).id, "seats_available");
  assert.equal(crowdLevelForPercent(76, custom).id, "crowded");
  assert.equal(crowdLevelForPercent(91, custom).id, "full");
});

test("decays passenger reports and ignores stale reports", () => {
  assert.equal(reportTimeWeight(report(2).timestamp, NOW), 1);
  assert.equal(reportTimeWeight(report(6).timestamp, NOW), 0.55);
  assert.equal(reportTimeWeight(report(15).timestamp, NOW), 0.2);
  assert.equal(reportTimeWeight(report(25).timestamp, NOW), 0.05);
  assert.equal(reportTimeWeight(report(30).timestamp, NOW), 0);
  assert.equal(reportTimeWeight(report(90).timestamp, NOW), 0);
});

test("one passenger report cannot dominate the sensor estimate", () => {
  const result = aggregateCrowdEstimate({
    sensorPercent: 62,
    sensorConfidence: 0.86,
    reports: [report(1, "full")],
    now: NOW,
  });
  assert.equal(result.recentReportCount, 1);
  assert.ok(result.communityInfluence <= 0.05);
  assert.ok(result.estimatedPercent - 62 <= 2);
  assert.equal(result.level.id, "moderate");
});

test("several recent agreeing reports influence more than one report", () => {
  const one = aggregateCrowdEstimate({
    sensorPercent: 72,
    sensorConfidence: 0.82,
    reports: [report(1, "crowded")],
    now: NOW,
  });
  const several = aggregateCrowdEstimate({
    sensorPercent: 72,
    sensorConfidence: 0.82,
    reports: [report(1, "crowded"), report(2, "crowded"), report(4, "full"), report(7, "crowded")],
    now: NOW,
  });
  assert.ok(several.communityInfluence > one.communityInfluence);
  assert.ok(several.estimatedPercent > one.estimatedPercent);
  assert.equal(several.communityDirection, "higher");
});

test("stale reports have no effect on current occupancy", () => {
  const result = aggregateCrowdEstimate({
    sensorPercent: 48,
    sensorConfidence: 0.8,
    reports: [report(31, "full"), report(90, "full")],
    now: NOW,
  });
  assert.equal(result.estimatedPercent, 48);
  assert.equal(result.recentReportCount, 0);
  assert.equal(result.communityDirection, "none");
});

test("converts the baseline at-stop forecast into a passenger crowd state", () => {
  const forecast = forecastOccupancy({
    recentOccupancy: [62, 66, 69, 72, 75, 77],
    capacity: 90,
    hour: 18,
    dayType: "weekday",
  });
  assert.equal(forecast.confidence, 0.86);
  assert.equal(crowdLevelForPercent(forecast.expectedPercent).id, "crowded");
});

test("passenger report contract contains no identity or free text", () => {
  const event = report(1, "moderate");
  assert.equal(event.source, "passenger_report");
  assert.deepEqual(Object.keys(event).sort(), [
    "busId",
    "crowdLevel",
    "reportId",
    "routeId",
    "seatsAvailable",
    "source",
    "timestamp",
  ]);
});

test("recommends waiting only when both predictions are confident", () => {
  const first = {
    busId: "22-04",
    etaMinutes: 3,
    currentLevel: crowdLevelForPercent(84),
    predictedLevel: crowdLevelForPercent(96),
    predictedConfidence: 0.86,
  };
  const second = {
    busId: "22-07",
    etaMinutes: 8,
    currentLevel: crowdLevelForPercent(31),
    predictedLevel: crowdLevelForPercent(53),
    predictedConfidence: 0.82,
  };
  assert.deepEqual(recommendPassengerChoice(first, second), { kind: "wait", waitMinutes: 5 });
  assert.deepEqual(
    recommendPassengerChoice(first, { ...second, predictedConfidence: 0.5 }),
    { kind: "insufficient_data" },
  );
});
