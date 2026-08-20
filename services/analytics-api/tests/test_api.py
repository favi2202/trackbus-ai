import os
import tempfile
import unittest
from datetime import datetime

from fastapi.testclient import TestClient

from trackbus_analytics.main import app


def payload(event_id: str, observed_at: str = "2026-07-18T12:00:00Z") -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "eventId": event_id,
        "observedAt": observed_at,
        "source": "vision",
        "busId": "BUS-API-1",
        "routeId": "22",
        "stopId": "CHORSU",
        "doorId": "DOOR-1",
        "boardings": 1,
        "alightings": 0,
        "occupancy": 1,
        "capacity": 72,
        "confidence": 0.94,
        "qualityFlags": [],
    }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        os.environ["TRACKBUS_DATABASE_PATH"] = f"{self.temp.name}/api.sqlite3"
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        os.environ.pop("TRACKBUS_DATABASE_PATH", None)
        self.temp.cleanup()

    def test_ingestion_is_persisted_and_idempotent(self) -> None:
        first = self.client.post("/v1/events/passenger-counts", json=payload("evt-api"))
        duplicate = self.client.post("/v1/events/passenger-counts", json=payload("evt-api"))
        self.assertEqual(first.status_code, 202)
        self.assertTrue(first.json()["persisted"])
        self.assertFalse(first.json()["duplicate"])
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(self.client.get("/v1/events/passenger-counts").json()["count"], 1)

    def test_rejects_unknown_source_family(self) -> None:
        invalid = payload("evt-invalid")
        invalid["source"] = "vendor-camera-x"
        self.assertEqual(
            self.client.post("/v1/events/passenger-counts", json=invalid).status_code, 422
        )

    def test_verified_empty_reset_sorts_after_future_device_time(self) -> None:
        future = "2099-01-01T00:00:00Z"
        self.client.post("/v1/events/passenger-counts", json=payload("evt-future", future))
        reset = self.client.post(
            "/v1/buses/BUS-API-1/verified-empty",
            json={"routeId": "22", "stopId": "DEPOT", "doorId": "DOOR-1", "capacity": 72},
        )
        self.assertEqual(reset.status_code, 202)
        self.assertGreater(
            datetime.fromisoformat(reset.json()["observedAt"]),
            datetime.fromisoformat(future),
        )
        latest = self.client.get("/v1/buses/BUS-API-1/occupancy").json()
        self.assertEqual(latest["occupancy"], 0)
        self.assertEqual(latest["source"], "manual")


if __name__ == "__main__":
    unittest.main()
