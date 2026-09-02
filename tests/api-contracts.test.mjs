import assert from "node:assert/strict";
import { DatabaseSync } from "node:sqlite";
import test from "node:test";
import { env as runtimeEnv } from "cloudflare:workers";

class FakeD1Statement {
  constructor(database, sql, values = []) {
    this.database = database;
    this.sql = sql;
    this.values = values;
  }

  bind(...values) {
    return new FakeD1Statement(this.database, this.sql, values);
  }

  async run() {
    const result = this.database.prepare(this.sql).run(...this.values);
    return { success: true, results: [], meta: { changes: Number(result.changes) } };
  }

  async all() {
    const results = this.database.prepare(this.sql).all(...this.values);
    return { success: true, results, meta: { changes: 0 } };
  }
}

class FakeD1 {
  constructor() {
    this.database = new DatabaseSync(":memory:");
  }

  prepare(sql) {
    return new FakeD1Statement(this.database, sql);
  }

  async batch(statements) {
    return Promise.all(statements.map((statement) => /^\s*(SELECT|WITH)\b/i.test(statement.sql) ? statement.all() : statement.run()));
  }
}

Object.assign(runtimeEnv, { DB: new FakeD1(), TRACKBUS_INGEST_KEY: "test-camera-key" });

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

test("health endpoint identifies connected D1 storage", async () => {
  const response = await request("/api/v1/health");
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.status, "ok");
  assert.equal(body.dataMode, "connected");
  assert.equal(body.storageConfigured, true);
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

const canonicalEvent = {
  schemaVersion: "1.0",
  eventId: "vision-test-1",
  observedAt: "2026-08-20T10:00:00Z",
  source: "vision",
  busId: "BUS-1",
  routeId: "22",
  stopId: "STOP-1",
  doorId: "DOOR-1",
  boardings: 1,
  alightings: 0,
  occupancy: 1,
  capacity: 72,
  confidence: 0.93,
  qualityFlags: [],
};

test("gateway rejects an unauthenticated camera event", async () => {
  const response = await request("/api/v1/events/passenger-counts", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(canonicalEvent),
  });
  assert.equal(response.status, 401);
  const body = await response.json();
  assert.equal(body.accepted, false);
  assert.equal(body.persisted, false);
  assert.equal(body.retryable, false);
});

test("authenticated event persists once and is readable as JSON", async () => {
  const first = await request("/api/v1/events/passenger-counts", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer test-camera-key" },
    body: JSON.stringify(canonicalEvent),
  });
  assert.equal(first.status, 202);
  assert.deepEqual((await first.json()).duplicate, false);

  const duplicate = await request("/api/v1/events/passenger-counts", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: "Bearer test-camera-key" },
    body: JSON.stringify(canonicalEvent),
  });
  assert.equal(duplicate.status, 200);
  assert.deepEqual((await duplicate.json()).duplicate, true);

  const recent = await request("/api/v1/events/passenger-counts?limit=10");
  assert.equal(recent.status, 200);
  const body = await recent.json();
  assert.equal(body.dataMode, "live");
  assert.equal(body.count, 1);
  assert.equal(body.events[0].eventId, canonicalEvent.eventId);
  assert.equal(body.events[0].occupancy, 1);
});

test("pilot proof endpoint returns the stored camera event", async () => {
  const response = await request("/api/v1/operations/pilot");
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.dataMode, "live");
  assert.equal(body.summary.eventCount, 1);
  assert.equal(body.summary.busCount, 1);
  assert.equal(body.sources[0].source, "vision");
  assert.equal(body.buses[0].busId, "BUS-1");
  assert.equal(body.recentEvents[0].eventId, canonicalEvent.eventId);
});
