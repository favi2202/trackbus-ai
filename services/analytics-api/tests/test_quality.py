import unittest

from trackbus_analytics.domain import PassengerCountEvent
from trackbus_analytics.quality import assess_event


def event(**overrides) -> PassengerCountEvent:
    values = {
        "eventId": "evt-current",
        "observedAt": "2026-07-18T17:30:00+05:00",
        "busId": "BUS-2204",
        "routeId": "22",
        "stopId": "CHORSU",
        "boardings": 5,
        "alightings": 2,
        "occupancy": 33,
        "capacity": 72,
        "source": "simulator",
        "qualityScore": 0.96,
    }
    values.update(overrides)
    return PassengerCountEvent(**values)


class QualityTests(unittest.TestCase):
    def test_consistent_event_has_no_findings(self) -> None:
        previous = event(eventId="evt-previous", occupancy=30)
        self.assertEqual(assess_event(event(), previous), ())

    def test_low_confidence_and_mismatch_are_explained(self) -> None:
        previous = event(eventId="evt-previous", occupancy=10)
        findings = assess_event(event(qualityScore=0.6, occupancy=50), previous)
        self.assertEqual(
            {finding.code for finding in findings},
            {"low_sensor_confidence", "occupancy_delta_mismatch"},
        )


if __name__ == "__main__":
    unittest.main()
