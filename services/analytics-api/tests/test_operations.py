import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from trackbus_analytics.domain import PassengerCountEvent
from trackbus_analytics.store import SQLiteEventStore


def event(
    event_id: str,
    *,
    observed_at: str,
    bus_id: str,
    route_id: str,
    occupancy: int,
    source: str = "apc",
    boardings: int = 5,
) -> PassengerCountEvent:
    return PassengerCountEvent(
        eventId=event_id,
        observedAt=observed_at,
        source=source,
        busId=bus_id,
        routeId=route_id,
        stopId="CHORSU",
        doorId="DOOR-1",
        boardings=boardings,
        alightings=2,
        occupancy=occupancy,
        capacity=80,
        confidence=0.96,
    )


class OperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteEventStore(Path(self.temp.name) / "trackbus.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_empty_store_reports_empty_status(self) -> None:
        summary = self.store.operational_summary(now=datetime(2026, 7, 18, 12, tzinfo=UTC))
        self.assertEqual(summary["status"], "empty")
        self.assertEqual(summary["eventCount"], 0)

    def test_summary_detects_stale_bus_and_quality_findings(self) -> None:
        self.store.append(
            event(
                "evt-old",
                observed_at="2026-07-18T10:00:00+00:00",
                bus_id="BUS-1",
                route_id="22",
                occupancy=60,
            ),
            [{"code": "low_source_confidence", "severity": "warning"}],
        )
        self.store.append(
            event(
                "evt-live",
                observed_at="2026-07-18T11:55:00+00:00",
                bus_id="BUS-2",
                route_id="22",
                occupancy=40,
            )
        )
        summary = self.store.operational_summary(
            now=datetime(2026, 7, 18, 12, tzinfo=UTC),
            stale_after_minutes=15,
        )
        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(summary["staleBusCount"], 1)
        self.assertEqual(summary["flaggedEventCount"], 1)

    def test_route_summary_uses_latest_event_per_bus(self) -> None:
        self.store.append(
            event(
                "evt-1",
                observed_at="2026-07-18T11:40:00+00:00",
                bus_id="BUS-1",
                route_id="22",
                occupancy=20,
            )
        )
        self.store.append(
            event(
                "evt-2",
                observed_at="2026-07-18T11:50:00+00:00",
                bus_id="BUS-1",
                route_id="22",
                occupancy=60,
            )
        )
        routes = self.store.route_summaries()
        self.assertEqual(routes[0]["routeId"], "22")
        self.assertEqual(routes[0]["averageOccupancyPercent"], 75.0)

    def test_source_health_separates_source_families(self) -> None:
        self.store.append(
            event(
                "evt-apc",
                observed_at="2026-07-18T11:58:00+00:00",
                bus_id="BUS-1",
                route_id="22",
                occupancy=20,
                source="apc",
            )
        )
        self.store.append(
            event(
                "evt-vision",
                observed_at="2026-07-18T10:00:00+00:00",
                bus_id="BUS-2",
                route_id="22",
                occupancy=20,
                source="vision",
            )
        )
        health = self.store.source_health(
            now=datetime(2026, 7, 18, 12, tzinfo=UTC),
            stale_after_minutes=15,
        )
        self.assertEqual(
            {row["source"]: row["status"] for row in health}, {"apc": "healthy", "vision": "stale"}
        )

    def test_reconciliation_exposes_payment_gap(self) -> None:
        self.store.append(
            event(
                "evt-apc",
                observed_at="2026-07-18T11:58:00+00:00",
                bus_id="BUS-1",
                route_id="22",
                occupancy=20,
                source="apc",
                boardings=10,
            )
        )
        self.store.append(
            event(
                "evt-pay",
                observed_at="2026-07-18T11:59:00+00:00",
                bus_id="BUS-2",
                route_id="22",
                occupancy=20,
                source="payment",
                boardings=4,
            )
        )
        result = self.store.reconciliation()[0]
        self.assertEqual(result["boardingGap"], 6)
        self.assertEqual(result["status"], "needs-review")


if __name__ == "__main__":
    unittest.main()
