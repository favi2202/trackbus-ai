import tempfile
import unittest
from pathlib import Path

from trackbus_analytics.domain import PassengerCountEvent
from trackbus_analytics.store import SQLiteEventStore


def event(event_id: str, occupancy: int = 24) -> PassengerCountEvent:
    return PassengerCountEvent(
        eventId=event_id,
        observedAt="2026-07-18T17:28:12+05:00",
        busId="BUS-2204",
        routeId="22",
        stopId="CHORSU",
        boardings=5,
        alightings=2,
        occupancy=occupancy,
        capacity=72,
        source="simulator",
        qualityScore=0.96,
    )


class SQLiteEventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteEventStore(Path(self.temp.name) / "trackbus.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_insert_is_idempotent(self) -> None:
        first = self.store.append(event("evt-1"))
        duplicate = self.store.append(event("evt-1"))
        self.assertFalse(first.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(len(self.store.list_recent()), 1)

    def test_latest_occupancy_is_queryable(self) -> None:
        self.store.append(event("evt-2", occupancy=41))
        latest = self.store.latest_for_bus("BUS-2204")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.occupancy, 41)

    def test_filters_by_route(self) -> None:
        self.store.append(event("evt-3"), [{"code": "example", "severity": "warning"}])
        self.assertEqual(len(self.store.list_recent(route_id="22")), 1)
        self.assertEqual(len(self.store.list_recent(route_id="67")), 0)
        self.assertEqual(
            self.store.list_recent(route_id="22")[0]["qualityIssues"][0]["code"],
            "example",
        )


if __name__ == "__main__":
    unittest.main()
