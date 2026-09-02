import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

from .domain import PassengerCountEvent


@dataclass(frozen=True)
class InsertResult:
    accepted: bool
    duplicate: bool


class SQLiteEventStore:
    """Durable pilot event store with vendor-neutral operational read models."""

    def __init__(self, path: str | Path = "data/trackbus.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS passenger_count_events (
                    event_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL DEFAULT '1.0',
                    observed_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    bus_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    stop_id TEXT NOT NULL,
                    door_id TEXT NOT NULL DEFAULT 'unknown',
                    boardings INTEGER NOT NULL CHECK (boardings >= 0),
                    alightings INTEGER NOT NULL CHECK (alightings >= 0),
                    occupancy INTEGER NOT NULL CHECK (occupancy >= 0),
                    capacity INTEGER NOT NULL CHECK (capacity > 0),
                    confidence REAL NOT NULL DEFAULT 1 CHECK (confidence BETWEEN 0 AND 1),
                    quality_score REAL NOT NULL DEFAULT 1,
                    quality_flags TEXT NOT NULL DEFAULT '[]',
                    quality_issues TEXT NOT NULL DEFAULT '[]',
                    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_events_bus_time
                    ON passenger_count_events (bus_id, observed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_route_time
                    ON passenger_count_events (route_id, observed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_source_time
                    ON passenger_count_events (source, observed_at DESC);
                """
            )
            existing = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(passenger_count_events)"
                ).fetchall()
            }
            migrations = {
                "schema_version": "TEXT NOT NULL DEFAULT '1.0'",
                "door_id": "TEXT NOT NULL DEFAULT 'unknown'",
                "confidence": "REAL NOT NULL DEFAULT 1",
                "quality_score": "REAL NOT NULL DEFAULT 1",
                "quality_flags": "TEXT NOT NULL DEFAULT '[]'",
            }
            for name, definition in migrations.items():
                if name not in existing:
                    self._connection.execute(
                        f"ALTER TABLE passenger_count_events ADD COLUMN {name} {definition}"
                    )

    def append(
        self,
        event: PassengerCountEvent,
        quality_issues: list[dict[str, str]] | None = None,
    ) -> InsertResult:
        values = event.model_dump(by_alias=False)
        values["observed_at"] = event.observed_at.isoformat()
        values["quality_flags"] = json.dumps(event.quality_flags, separators=(",", ":"))
        values["quality_score"] = event.confidence
        values["quality_issues"] = json.dumps(quality_issues or [], separators=(",", ":"))
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO passenger_count_events (
                    event_id, schema_version, observed_at, source, bus_id, route_id,
                    stop_id, door_id, boardings, alightings, occupancy, capacity,
                    confidence, quality_score, quality_flags, quality_issues
                ) VALUES (
                    :event_id, :schema_version, :observed_at, :source, :bus_id,
                    :route_id, :stop_id, :door_id, :boardings, :alightings,
                    :occupancy, :capacity, :confidence, :quality_score,
                    :quality_flags, :quality_issues
                )
                """,
                values,
            )
        return InsertResult(accepted=True, duplicate=cursor.rowcount != 1)

    def latest_for_bus(self, bus_id: str) -> PassengerCountEvent | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM passenger_count_events WHERE bus_id = ?
                ORDER BY observed_at DESC, inserted_at DESC LIMIT 1""",
                (bus_id,),
            ).fetchone()
        return self._event_from_row(row) if row else None

    def list_recent(
        self,
        *,
        limit: int = 100,
        bus_id: str | None = None,
        route_id: str | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if bus_id:
            clauses.append("bus_id = ?")
            parameters.append(bus_id)
        if route_id:
            clauses.append("route_id = ?")
            parameters.append(route_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 500)))
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT * FROM passenger_count_events {where}
                ORDER BY observed_at DESC, inserted_at DESC LIMIT ?""",
                parameters,
            ).fetchall()
        return [self._dict_from_row(row) for row in rows]

    def operational_summary(
        self,
        *,
        now: datetime | None = None,
        stale_after_minutes: int = 15,
    ) -> dict[str, object]:
        reference = now or datetime.now(UTC)
        cutoff = reference - timedelta(minutes=max(1, min(stale_after_minutes, 1440)))
        with self._lock:
            totals = self._connection.execute(
                """
                SELECT COUNT(*) AS event_count, COUNT(DISTINCT bus_id) AS bus_count,
                    COUNT(DISTINCT route_id) AS route_count,
                    MAX(observed_at) AS latest_observed_at,
                    SUM(CASE WHEN quality_issues != '[]' THEN 1 ELSE 0 END) AS flagged_count
                FROM passenger_count_events
                """
            ).fetchone()
            latest_buses = self._connection.execute(
                """SELECT bus_id, MAX(observed_at) AS latest_observed_at
                FROM passenger_count_events GROUP BY bus_id"""
            ).fetchall()
        stale_buses = sum(
            self._parse_observed(row["latest_observed_at"]) < cutoff for row in latest_buses
        )
        return {
            "eventCount": totals["event_count"],
            "busCount": totals["bus_count"],
            "routeCount": totals["route_count"],
            "flaggedEventCount": totals["flagged_count"] or 0,
            "latestObservedAt": totals["latest_observed_at"],
            "staleBusCount": stale_buses,
            "staleAfterMinutes": max(1, min(stale_after_minutes, 1440)),
            "status": (
                "empty"
                if totals["event_count"] == 0
                else "degraded"
                if stale_buses
                else "operational"
            ),
        }

    def source_health(
        self,
        *,
        now: datetime | None = None,
        stale_after_minutes: int = 15,
    ) -> list[dict[str, object]]:
        reference = now or datetime.now(UTC)
        cutoff = reference - timedelta(minutes=stale_after_minutes)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT source, COUNT(*) AS event_count,
                    COUNT(DISTINCT bus_id) AS bus_count,
                    MAX(observed_at) AS latest_observed_at,
                    SUM(CASE WHEN quality_issues != '[]' THEN 1 ELSE 0 END) AS flagged_count
                FROM passenger_count_events GROUP BY source ORDER BY source
                """
            ).fetchall()
        return [
            {
                "source": row["source"],
                "status": (
                    "stale"
                    if self._parse_observed(row["latest_observed_at"]) < cutoff
                    else "degraded"
                    if row["flagged_count"]
                    else "healthy"
                ),
                "eventCount": row["event_count"],
                "busCount": row["bus_count"],
                "flaggedEventCount": row["flagged_count"] or 0,
                "latestObservedAt": row["latest_observed_at"],
            }
            for row in rows
        ]

    def reconciliation(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT route_id, source, SUM(boardings) AS boardings,
                    SUM(alightings) AS alightings, COUNT(*) AS event_count
                FROM passenger_count_events GROUP BY route_id, source
                ORDER BY route_id, source
                """
            ).fetchall()
        routes: dict[str, dict[str, dict[str, int]]] = {}
        for row in rows:
            routes.setdefault(row["route_id"], {})[row["source"]] = {
                "boardings": row["boardings"] or 0,
                "alightings": row["alightings"] or 0,
                "eventCount": row["event_count"],
            }
        result: list[dict[str, object]] = []
        for route_id, sources in routes.items():
            sensor_boardings = sum(
                sources.get(source, {}).get("boardings", 0) for source in ("apc", "vision")
            )
            payment_boardings = sources.get("payment", {}).get("boardings")
            gap = None if payment_boardings is None else sensor_boardings - payment_boardings
            result.append(
                {
                    "routeId": route_id,
                    "sources": sources,
                    "sensorBoardings": sensor_boardings,
                    "paymentBoardings": payment_boardings,
                    "boardingGap": gap,
                    "status": (
                        "insufficient-evidence"
                        if gap is None
                        else "needs-review"
                        if abs(gap) > 3
                        else "aligned"
                    ),
                }
            )
        return result

    def route_summaries(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY bus_id ORDER BY observed_at DESC, inserted_at DESC
                    ) AS position FROM passenger_count_events
                )
                SELECT route_id, COUNT(*) AS active_buses,
                    ROUND(AVG(occupancy * 100.0 / capacity), 1) AS average_occupancy_percent,
                    MAX(observed_at) AS latest_observed_at,
                    SUM(CASE WHEN quality_issues != '[]' THEN 1 ELSE 0 END) AS flagged_buses
                FROM ranked WHERE position = 1 GROUP BY route_id
                ORDER BY average_occupancy_percent DESC, route_id
                """
            ).fetchall()
        return [
            {
                "routeId": row["route_id"],
                "activeBuses": row["active_buses"],
                "averageOccupancyPercent": row["average_occupancy_percent"],
                "latestObservedAt": row["latest_observed_at"],
                "flaggedBuses": row["flagged_buses"] or 0,
            }
            for row in rows
        ]

    @staticmethod
    def _parse_observed(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> PassengerCountEvent:
        return PassengerCountEvent(
            schemaVersion=row["schema_version"],
            eventId=row["event_id"],
            observedAt=row["observed_at"],
            source=row["source"],
            busId=row["bus_id"],
            routeId=row["route_id"],
            stopId=row["stop_id"],
            doorId=row["door_id"],
            boardings=row["boardings"],
            alightings=row["alightings"],
            occupancy=row["occupancy"],
            capacity=row["capacity"],
            confidence=row["confidence"],
            qualityFlags=json.loads(row["quality_flags"]),
        )

    @classmethod
    def _dict_from_row(cls, row: sqlite3.Row) -> dict[str, object]:
        payload = cls._event_from_row(row).model_dump(by_alias=True, mode="json")
        payload["qualityIssues"] = json.loads(row["quality_issues"])
        payload["insertedAt"] = row["inserted_at"]
        return payload

    def close(self) -> None:
        with self._lock:
            self._connection.close()
