import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .domain import PassengerCountEvent


@dataclass(frozen=True)
class InsertResult:
    accepted: bool
    duplicate: bool


class SQLiteEventStore:
    """Small durable adapter for a pilot; replaceable behind the same service boundary."""

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
                    observed_at TEXT NOT NULL,
                    bus_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    stop_id TEXT NOT NULL,
                    boardings INTEGER NOT NULL CHECK (boardings >= 0),
                    alightings INTEGER NOT NULL CHECK (alightings >= 0),
                    occupancy INTEGER NOT NULL CHECK (occupancy >= 0),
                    capacity INTEGER NOT NULL CHECK (capacity > 0),
                    source TEXT NOT NULL,
                    quality_score REAL NOT NULL CHECK (quality_score BETWEEN 0 AND 1),
                    quality_issues TEXT NOT NULL DEFAULT '[]',
                    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_events_bus_time
                    ON passenger_count_events (bus_id, observed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_route_time
                    ON passenger_count_events (route_id, observed_at DESC);
                """
            )

    def append(
        self,
        event: PassengerCountEvent,
        quality_issues: list[dict[str, str]] | None = None,
    ) -> InsertResult:
        values = event.model_dump(by_alias=False)
        values["observed_at"] = event.observed_at.isoformat()
        values["quality_issues"] = json.dumps(quality_issues or [], separators=(",", ":"))
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO passenger_count_events (
                    event_id, observed_at, bus_id, route_id, stop_id,
                    boardings, alightings, occupancy, capacity, source,
                    quality_score, quality_issues
                ) VALUES (
                    :event_id, :observed_at, :bus_id, :route_id, :stop_id,
                    :boardings, :alightings, :occupancy, :capacity, :source,
                    :quality_score, :quality_issues
                )
                """,
                values,
            )
        inserted = cursor.rowcount == 1
        return InsertResult(accepted=True, duplicate=not inserted)

    def latest_for_bus(self, bus_id: str) -> PassengerCountEvent | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM passenger_count_events
                WHERE bus_id = ? ORDER BY observed_at DESC LIMIT 1
                """,
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
                f"""
                SELECT * FROM passenger_count_events {where}
                ORDER BY observed_at DESC LIMIT ?
                """,  # noqa: S608 - the dynamic fragment contains only fixed clause names
                parameters,
            ).fetchall()
        return [self._dict_from_row(row) for row in rows]

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> PassengerCountEvent:
        return PassengerCountEvent(
            eventId=row["event_id"],
            observedAt=row["observed_at"],
            busId=row["bus_id"],
            routeId=row["route_id"],
            stopId=row["stop_id"],
            boardings=row["boardings"],
            alightings=row["alightings"],
            occupancy=row["occupancy"],
            capacity=row["capacity"],
            source=row["source"],
            qualityScore=row["quality_score"],
        )

    @classmethod
    def _dict_from_row(cls, row: sqlite3.Row) -> dict[str, object]:
        event = cls._event_from_row(row)
        payload = event.model_dump(by_alias=True, mode="json")
        payload["qualityIssues"] = json.loads(row["quality_issues"])
        payload["insertedAt"] = row["inserted_at"]
        return payload

    def close(self) -> None:
        with self._lock:
            self._connection.close()
