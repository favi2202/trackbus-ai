import pytest

from trackbus.detection import TrackedDetection
from trackbus.showcase_zones import (
    CrossingStateMachine,
    GateGeometry,
    MonotonicGateCounter,
    Zone,
    ZoneLayout,
)


def feed(
    machine: CrossingStateMachine,
    zones: list[Zone],
    *,
    track_id: int = 7,
    start: int = 1,
):
    events = []
    for frame, zone in enumerate(zones, start=start):
        event = machine.observe(
            track_id=track_id,
            zone=zone,
            frame_number=frame,
            confidence=0.9,
        )
        if event:
            events.append(event)
    return events


def test_default_layout_classifies_three_bands() -> None:
    layout = ZoneLayout.default()
    assert layout.classify((50, 10), (100, 100, 3)) is Zone.OUTSIDE
    assert layout.classify((50, 50), (100, 100, 3)) is Zone.DOOR
    assert layout.classify((50, 90), (100, 100, 3)) is Zone.INSIDE


def test_outside_door_inside_emits_boarding() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.OUTSIDE] * 2 + [Zone.DOOR] * 2 + [Zone.INSIDE] * 2,
    )
    assert len(events) == 1
    assert events[0].direction == "IN"
    assert events[0].occupancy_delta == 1


def test_inside_door_outside_emits_alighting() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.INSIDE] * 2 + [Zone.DOOR] * 2 + [Zone.OUTSIDE] * 2,
    )
    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert events[0].occupancy_delta == -1


def test_incomplete_path_never_counts() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.OUTSIDE] * 2 + [Zone.DOOR] * 2 + [Zone.UNKNOWN] * 4,
    )
    assert events == []


def test_direct_zone_jump_never_counts() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.OUTSIDE] * 2 + [Zone.INSIDE] * 2,
    )
    assert events == []


def test_reversal_to_origin_never_counts() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.OUTSIDE] * 2 + [Zone.DOOR] * 2 + [Zone.OUTSIDE] * 2,
    )
    assert events == []


def test_single_frame_noise_is_debounced() -> None:
    events = feed(
        CrossingStateMachine(minimum_zone_frames=2),
        [Zone.OUTSIDE] * 2 + [Zone.DOOR] + [Zone.OUTSIDE] * 2 + [Zone.INSIDE] * 2,
    )
    assert events == []


def test_long_tracking_gap_resets_journey() -> None:
    machine = CrossingStateMachine(minimum_zone_frames=1, maximum_gap_frames=3)
    feed(machine, [Zone.OUTSIDE, Zone.DOOR], start=1)
    events = feed(machine, [Zone.INSIDE], start=10)
    assert events == []


def test_same_track_can_complete_a_full_reverse_crossing() -> None:
    machine = CrossingStateMachine(minimum_zone_frames=1)
    events = feed(
        machine,
        [
            Zone.OUTSIDE,
            Zone.DOOR,
            Zone.INSIDE,
            Zone.DOOR,
            Zone.OUTSIDE,
        ],
    )
    assert [event.direction for event in events] == ["IN", "OUT"]


def tracked_box(
    top: float,
    bottom: float,
    *,
    track_id: int = 7,
    prediction_only: bool = False,
) -> TrackedDetection:
    return TrackedDetection(
        tracking_id=track_id,
        bounding_box=(40, top, 60, bottom),
        confidence=0.8,
        metadata={"prediction_only": prediction_only},
    )


def feed_gates(
    counter: MonotonicGateCounter,
    boxes: list[TrackedDetection],
    *,
    start: int = 1,
):
    events = []
    for frame_number, track in enumerate(boxes, start=start):
        event = counter.observe(
            track=track,
            frame_number=frame_number,
            frame_shape=(100, 100, 3),
        )
        if event is not None:
            events.append(event)
    return events


def test_default_gate_geometry_is_ordered_on_zone_axis() -> None:
    geometry = GateGeometry.from_layout(ZoneLayout.default())

    assert geometry.outside_gate == 0.25
    assert geometry.inside_gate == 0.75
    assert geometry.progress((50, 20), (101, 101, 3)) == pytest.approx(0)
    assert geometry.progress((50, 80), (101, 101, 3)) == pytest.approx(1)


def test_monotonic_center_crossing_emits_one_boarding() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(45, 55),
            tracked_box(45, 55),
            tracked_box(75, 85),
            tracked_box(75, 85),
        ],
    )

    assert [event.direction for event in events] == ["IN"]
    assert counter.summary["confirmed_events"] == 1
    decisions = counter.drain_decisions()
    assert [decision.action for decision in decisions] == [
        "origin_confirmed",
        "gate_crossed",
        "event_confirmed",
    ]
    assert decisions[-1].reason == "two_gates_monotonic"
    assert decisions[-1].to_payload() == {
        "recordType": "trajectory-decision",
        "frameNumber": 6,
        "trackId": "T7",
        "action": "event_confirmed",
        "reason": "two_gates_monotonic",
        "progress": pytest.approx(1.013468),
        "direction": "IN",
        "origin": "outside",
        "directionConsistency": 1.0,
    }


def test_monotonic_center_crossing_emits_one_alighting() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(75, 85),
            tracked_box(75, 85),
            tracked_box(45, 55),
            tracked_box(45, 55),
            tracked_box(10, 20),
            tracked_box(10, 20),
        ],
    )

    assert [event.direction for event in events] == ["OUT"]


def test_direction_inconsistency_rejects_a_blinking_path() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
        minimum_direction_consistency=0.70,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(45, 55),
            tracked_box(20, 30),
            tracked_box(75, 85),
            tracked_box(75, 85),
        ],
    )

    assert events == []
    assert counter.summary["inconsistent_direction_rejections"] == 1
    rejected = [
        decision
        for decision in counter.drain_decisions()
        if decision.action == "journey_rejected"
    ]
    assert rejected[-1].reason == "inconsistent_direction"
    assert rejected[-1].direction_consistency < 0.70


def test_prediction_only_track_cannot_create_crossing_event() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20, prediction_only=True),
            tracked_box(45, 55, prediction_only=True),
            tracked_box(75, 85, prediction_only=True),
        ],
    )

    assert events == []
    assert counter.summary["prediction_observations_ignored"] == 3


def test_real_endpoint_jump_can_count_without_fabricated_door_observation() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
        maximum_gap_frames=8,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(75, 85),
            tracked_box(75, 85),
        ],
    )

    assert [event.direction for event in events] == ["IN"]
    reasons = [decision.reason for decision in counter.drain_decisions()]
    assert "two_gates_monotonic" in reasons
    assert "geometric_door_bridge" not in reasons


def test_return_to_origin_is_rejected_and_audited() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(45, 55),
            tracked_box(10, 20),
            tracked_box(10, 20),
        ],
    )

    assert events == []
    assert counter.summary["reversed_journeys"] == 1
    assert "returned_to_origin" in {
        decision.reason for decision in counter.drain_decisions()
    }


def test_cooldown_blocks_an_implausibly_fast_reverse_event() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
        event_cooldown_frames=90,
    )
    events = feed_gates(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(45, 55),
            tracked_box(75, 85),
            tracked_box(75, 85),
            tracked_box(45, 55),
            tracked_box(10, 20),
            tracked_box(10, 20),
        ],
    )

    assert [event.direction for event in events] == ["IN"]
    assert counter.summary["event_cooldown_suppressed"] == 1
    assert "event_cooldown" in {
        decision.reason for decision in counter.drain_decisions()
    }


def test_tracking_gap_resets_gate_journey() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
        maximum_gap_frames=3,
    )
    feed_gates(
        counter,
        [tracked_box(10, 20), tracked_box(45, 55)],
        start=1,
    )
    events = feed_gates(counter, [tracked_box(75, 85)], start=10)

    assert events == []
    assert counter.summary["gap_resets"] == 1
    assert "tracking_gap" in {
        decision.reason for decision in counter.drain_decisions()
    }


def test_expired_track_is_recorded_before_id_state_is_removed() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
        maximum_gap_frames=3,
    )
    feed_gates(counter, [tracked_box(10, 20)], start=1)

    counter.expire(5)

    assert counter.summary["gap_resets"] == 1
    assert "track_expired" in {
        decision.reason for decision in counter.drain_decisions()
    }


def test_tall_static_box_does_not_create_two_anchor_false_event() -> None:
    counter = MonotonicGateCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
    )
    events = feed_gates(
        counter,
        [tracked_box(10, 90) for _ in range(8)],
    )

    assert events == []
    assert counter.summary["confirmed_events"] == 0
