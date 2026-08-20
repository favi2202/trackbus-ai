import unittest

from trackbus_analytics.domain import PassengerCountEvent
from trackbus_analytics.quality import assess_event


def event(**overrides) -> PassengerCountEvent:
    values = {
        "eventId": "evt-current",
        "observedAt": "2026-07-18T17:30:00+05:00",
        "source": "apc",
        "busId": "BUS-2204",
        "routeId": "22",
        "stopId": "CHORSU",
        "doorId": "DOOR-1",
        "boardings": 5,
        "alightings": 2,
        "occupancy": 33,
        "capacity": 72,
        "confidence": 0.96,
        "qualityFlags": [],
    }
    values.update(overrides)
    return PassengerCountEvent(**values)


class QualityTests(unittest.TestCase):
    def test_consistent_event_has_no_findings(self) -> None:
        previous = event(eventId="evt-previous", occupancy=30)
        self.assertEqual(assess_event(event(), previous), ())

    def test_low_confidence_and_mismatch_are_explained(self) -> None:
        previous = event(eventId="evt-previous", occupancy=10)
        findings = assess_event(event(confidence=0.6, occupancy=50), previous)
        self.assertEqual(
            {finding.code for finding in findings},
            {"low_source_confidence", "occupancy_delta_mismatch"},
        )

    def test_impossible_negative_occupancy_is_an_error(self) -> None:
        previous = event(eventId="evt-previous", occupancy=1, boardings=0, alightings=0)
        findings = assess_event(event(boardings=0, alightings=3, occupancy=0), previous)
        issue = next(item for item in findings if item.code == "impossible_negative_occupancy")
        self.assertEqual(issue.severity, "error")

    def test_verified_empty_reset_skips_delta_mismatch(self) -> None:
        previous = event(eventId="evt-previous", occupancy=60)
        reset = event(
            eventId="evt-reset",
            source="manual",
            boardings=0,
            alightings=0,
            occupancy=0,
            qualityFlags=["verified_empty_reset"],
        )
        self.assertEqual(assess_event(reset, previous), ())


if __name__ == "__main__":
    unittest.main()
