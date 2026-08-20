from datetime import UTC, datetime

import pytest

from trackbus.event_contract import EventSource, PassengerCountEvent, vision_event


def test_source_families_are_vendor_neutral() -> None:
    assert {source.value for source in EventSource} == {
        "apc",
        "vision",
        "payment",
        "manual",
        "import",
    }


def test_payload_uses_canonical_camel_case_fields() -> None:
    event = vision_event(
        direction="IN",
        occupancy=4,
        capacity=40,
        confidence=0.91,
        bus_id="BUS-1",
        route_id="22",
        stop_id="STOP-1",
        door_id="DOOR-1",
        observed_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    payload = event.to_payload()
    assert payload["schemaVersion"] == "1.0"
    assert payload["source"] == "vision"
    assert payload["boardings"] == 1
    assert payload["doorId"] == "DOOR-1"


def test_timestamp_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        PassengerCountEvent(
            event_id="evt",
            observed_at="2026-08-20T12:00:00",
            source=EventSource.VISION,
            bus_id="BUS",
            route_id="22",
            stop_id="STOP",
            door_id="DOOR",
            boardings=1,
            alightings=0,
            occupancy=1,
            capacity=40,
            confidence=0.9,
        )


def test_out_event_maps_to_alighting() -> None:
    event = vision_event(
        direction="OUT",
        occupancy=3,
        capacity=40,
        confidence=0.88,
        bus_id="BUS",
        route_id="22",
        stop_id="STOP",
        door_id="DOOR",
    )
    assert event.boardings == 0
    assert event.alightings == 1
