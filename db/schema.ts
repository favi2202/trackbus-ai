import { index, integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const passengerCountEvents = sqliteTable(
  "passenger_count_events",
  {
    eventId: text("event_id").primaryKey(),
    schemaVersion: text("schema_version").notNull(),
    observedAt: text("observed_at").notNull(),
    receivedAt: text("received_at").notNull(),
    source: text("source").notNull(),
    busId: text("bus_id").notNull(),
    routeId: text("route_id").notNull(),
    stopId: text("stop_id").notNull(),
    doorId: text("door_id").notNull(),
    boardings: integer("boardings").notNull(),
    alightings: integer("alightings").notNull(),
    occupancy: integer("occupancy").notNull(),
    capacity: integer("capacity").notNull(),
    confidence: real("confidence").notNull(),
    qualityFlags: text("quality_flags").notNull(),
  },
  (table) => [
    index("passenger_events_observed_idx").on(table.observedAt),
    index("passenger_events_bus_observed_idx").on(table.busId, table.observedAt),
    index("passenger_events_route_observed_idx").on(table.routeId, table.observedAt),
    index("passenger_events_source_observed_idx").on(table.source, table.observedAt),
  ],
);
