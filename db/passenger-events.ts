import { env } from "cloudflare:workers";

import type { PassengerCountEvent } from "@/lib/contracts";

type EventRow = {
  event_id: string;
  schema_version: string;
  observed_at: string;
  received_at: string;
  source: PassengerCountEvent["source"];
  bus_id: string;
  route_id: string;
  stop_id: string;
  door_id: string;
  boardings: number;
  alightings: number;
  occupancy: number;
  capacity: number;
  confidence: number;
  quality_flags: string;
};

type SummaryRow = {
  event_count: number;
  bus_count: number;
  route_count: number;
  flagged_event_count: number;
  stale_bus_count: number;
};

type SourceRow = {
  source: string;
  event_count: number;
  bus_count: number;
  latest_observed_at: string;
};

type ReconciliationRow = {
  route_id: string;
  sensor_boardings: number;
  payment_boardings: number;
  payment_event_count: number;
  latest_observed_at: string;
};

type BusRow = {
  bus_id: string;
  route_id: string;
  occupancy: number;
  capacity: number;
  source: string;
  observed_at: string;
};

const createTableSql = `
  CREATE TABLE IF NOT EXISTS passenger_count_events (
    event_id TEXT PRIMARY KEY NOT NULL,
    schema_version TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    source TEXT NOT NULL,
    bus_id TEXT NOT NULL,
    route_id TEXT NOT NULL,
    stop_id TEXT NOT NULL,
    door_id TEXT NOT NULL,
    boardings INTEGER NOT NULL,
    alightings INTEGER NOT NULL,
    occupancy INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    confidence REAL NOT NULL,
    quality_flags TEXT NOT NULL
  )
`;

const indexStatements = [
  "CREATE INDEX IF NOT EXISTS passenger_events_observed_idx ON passenger_count_events (observed_at)",
  "CREATE INDEX IF NOT EXISTS passenger_events_bus_observed_idx ON passenger_count_events (bus_id, observed_at)",
  "CREATE INDEX IF NOT EXISTS passenger_events_route_observed_idx ON passenger_count_events (route_id, observed_at)",
  "CREATE INDEX IF NOT EXISTS passenger_events_source_observed_idx ON passenger_count_events (source, observed_at)",
];

export function getD1(): D1Database {
  const database = env.DB;
  if (!database) throw new Error("TrackBus D1 storage is unavailable");
  return database;
}

export function getIngestKey(): string | undefined {
  return env.TRACKBUS_INGEST_KEY;
}

async function ensureSchema(database: D1Database): Promise<void> {
  await database.batch([
    database.prepare(createTableSql),
    ...indexStatements.map((statement) => database.prepare(statement)),
  ]);
}

function flagsFromJson(value: string): string[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return ["invalid_stored_quality_flags"];
  }
}

function eventFromRow(row: EventRow): PassengerCountEvent & { receivedAt: string } {
  return {
    schemaVersion: "1.0",
    eventId: row.event_id,
    observedAt: row.observed_at,
    receivedAt: row.received_at,
    source: row.source,
    busId: row.bus_id,
    routeId: row.route_id,
    stopId: row.stop_id,
    doorId: row.door_id,
    boardings: row.boardings,
    alightings: row.alightings,
    occupancy: row.occupancy,
    capacity: row.capacity,
    confidence: row.confidence,
    qualityFlags: flagsFromJson(row.quality_flags),
  };
}

export async function persistPassengerEvent(event: PassengerCountEvent) {
  const database = getD1();
  await ensureSchema(database);
  const receivedAt = new Date().toISOString();
  const result = await database
    .prepare(
      `INSERT OR IGNORE INTO passenger_count_events (
        event_id, schema_version, observed_at, received_at, source,
        bus_id, route_id, stop_id, door_id, boardings, alightings,
        occupancy, capacity, confidence, quality_flags
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .bind(
      event.eventId,
      event.schemaVersion,
      event.observedAt,
      receivedAt,
      event.source,
      event.busId,
      event.routeId,
      event.stopId,
      event.doorId,
      event.boardings,
      event.alightings,
      event.occupancy,
      event.capacity,
      event.confidence,
      JSON.stringify(event.qualityFlags),
    )
    .run();
  const duplicate = (result.meta.changes ?? 0) === 0;
  return { duplicate, receivedAt };
}

export async function recentPassengerEvents(limit = 25, busId?: string) {
  const database = getD1();
  await ensureSchema(database);
  const boundedLimit = Math.max(1, Math.min(100, Math.trunc(limit)));
  const statement = busId
    ? database
        .prepare("SELECT * FROM passenger_count_events WHERE bus_id = ? ORDER BY observed_at DESC, received_at DESC LIMIT ?")
        .bind(busId, boundedLimit)
    : database
        .prepare("SELECT * FROM passenger_count_events ORDER BY observed_at DESC, received_at DESC LIMIT ?")
        .bind(boundedLimit);
  const result = await statement.all<EventRow>();
  return result.results.map(eventFromRow);
}

export async function pilotSnapshot() {
  const database = getD1();
  await ensureSchema(database);
  const [summaryResult, sourceResult, reconciliationResult, busResult, recentResult] = await database.batch([
    database.prepare(`
      SELECT
        COUNT(*) AS event_count,
        COUNT(DISTINCT bus_id) AS bus_count,
        COUNT(DISTINCT route_id) AS route_count,
        COALESCE(SUM(CASE WHEN quality_flags <> '[]' THEN 1 ELSE 0 END), 0) AS flagged_event_count,
        COALESCE((
          SELECT COUNT(*) FROM (
            SELECT bus_id, MAX(observed_at) AS latest
            FROM passenger_count_events GROUP BY bus_id
          ) WHERE unixepoch(latest) < unixepoch('now') - 120
        ), 0) AS stale_bus_count
      FROM passenger_count_events
    `),
    database.prepare(`
      SELECT source, COUNT(*) AS event_count, COUNT(DISTINCT bus_id) AS bus_count,
             MAX(observed_at) AS latest_observed_at
      FROM passenger_count_events GROUP BY source ORDER BY source
    `),
    database.prepare(`
      SELECT route_id,
             COALESCE(SUM(CASE WHEN source IN ('apc', 'vision') THEN boardings ELSE 0 END), 0) AS sensor_boardings,
             COALESCE(SUM(CASE WHEN source = 'payment' THEN boardings ELSE 0 END), 0) AS payment_boardings,
             COALESCE(SUM(CASE WHEN source = 'payment' THEN 1 ELSE 0 END), 0) AS payment_event_count,
             MAX(observed_at) AS latest_observed_at
      FROM passenger_count_events
      GROUP BY route_id ORDER BY latest_observed_at DESC LIMIT 1
    `),
    database.prepare(`
      SELECT bus_id, route_id, occupancy, capacity, source, observed_at FROM (
        SELECT bus_id, route_id, occupancy, capacity, source, observed_at,
               ROW_NUMBER() OVER (PARTITION BY bus_id ORDER BY observed_at DESC, received_at DESC) AS row_number
        FROM passenger_count_events
      ) WHERE row_number = 1 ORDER BY observed_at DESC LIMIT 20
    `),
    database.prepare("SELECT * FROM passenger_count_events ORDER BY observed_at DESC, received_at DESC LIMIT 20"),
  ]);

  const summary = (summaryResult.results[0] ?? {
    event_count: 0,
    bus_count: 0,
    route_count: 0,
    flagged_event_count: 0,
    stale_bus_count: 0,
  }) as SummaryRow;
  const sources = (sourceResult.results as SourceRow[]).map((source) => {
    const ageSeconds = Math.max(0, (Date.now() - Date.parse(source.latest_observed_at)) / 1000);
    return {
      source: source.source,
      status: ageSeconds <= 120 ? "healthy" : "stale",
      coverage: `${source.bus_count} ${source.bus_count === 1 ? "bus" : "buses"}`,
      eventCount: source.event_count,
      busCount: source.bus_count,
      latestObservedAt: source.latest_observed_at,
    };
  });
  const reconciliationRow = reconciliationResult.results[0] as ReconciliationRow | undefined;
  const paymentBoardings = reconciliationRow?.payment_event_count ? reconciliationRow.payment_boardings : null;
  const boardingGap = reconciliationRow && paymentBoardings !== null
    ? reconciliationRow.sensor_boardings - paymentBoardings
    : null;

  return {
    dataMode: summary.event_count > 0 ? "live" : "live-empty",
    summary: {
      status: summary.event_count > 0 ? "receiving" : "waiting-for-camera",
      eventCount: summary.event_count,
      busCount: summary.bus_count,
      routeCount: summary.route_count,
      flaggedEventCount: summary.flagged_event_count,
      staleBusCount: summary.stale_bus_count,
    },
    sources,
    reconciliation: reconciliationRow
      ? {
          routeId: reconciliationRow.route_id,
          sensorBoardings: reconciliationRow.sensor_boardings,
          paymentBoardings,
          boardingGap,
          status: boardingGap === null ? "awaiting-payment-source" : Math.abs(boardingGap) <= 2 ? "aligned" : "needs-review",
        }
      : null,
    buses: (busResult.results as BusRow[]).map((bus) => ({
      busId: bus.bus_id,
      routeId: bus.route_id,
      occupancy: bus.occupancy,
      capacity: bus.capacity,
      source: bus.source,
      observedAt: bus.observed_at,
    })),
    recentEvents: (recentResult.results as EventRow[]).map(eventFromRow),
    refreshedAt: new Date().toISOString(),
  };
}
