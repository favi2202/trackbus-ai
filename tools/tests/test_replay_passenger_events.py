import json
import tempfile
import unittest
from pathlib import Path

from replay_passenger_events import load_events


class ReplayInputTests(unittest.TestCase):
    def test_loads_valid_event_array(self) -> None:
        payload = [
            {
                "schemaVersion": "1.0",
                "eventId": "evt-1",
                "observedAt": "2026-07-18T17:30:00+05:00",
                "source": "import",
                "busId": "BUS-2204",
                "routeId": "22",
                "stopId": "CHORSU",
                "doorId": "DOOR-1",
                "boardings": 1,
                "alightings": 0,
                "occupancy": 12,
                "capacity": 72,
                "confidence": 0.95,
                "qualityFlags": [],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_events(str(path))[0]["eventId"], "evt-1")

    def test_rejects_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text('[{"eventId":"evt-1"}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_events(str(path))


if __name__ == "__main__":
    unittest.main()
