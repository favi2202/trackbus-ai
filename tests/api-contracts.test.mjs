import assert from "node:assert/strict";
import test from "node:test";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("api-test", `${process.pid}-${Date.now()}`);
const { default: worker } = await import(workerUrl.href);
const env = {
  ASSETS: {
    fetch: async () => new Response("Not found", { status: 404 }),
  },
};
const ctx = { waitUntil() {}, passThroughOnException() {} };

async function request(path, init) {
  return worker.fetch(new Request(`http://localhost${path}`, init), env, ctx);
}

test("health endpoint identifies synthetic data mode", async () => {
  const response = await request("/api/v1/health");
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.status, "ok");
  assert.equal(body.dataMode, "synthetic");
});

test("forecast endpoint returns a bounded baseline result", async () => {
  const response = await request("/api/v1/forecast", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      recentOccupancy: [42, 47, 51, 56, 61, 66],
      capacity: 72,
      hour: 19,
      dayType: "weekday",
      weather: "rain",
    }),
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.ok(body.expectedOccupancy <= 72);
  assert.ok(body.lowerBound >= 0);
  assert.ok(body.upperBound <= 72);
});

test("passenger-count endpoint rejects an invalid contract", async () => {
  const response = await request("/api/v1/events/passenger-counts", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ eventId: "incomplete" }),
  });
  assert.equal(response.status, 422);
  assert.equal((await response.json()).accepted, false);
});
