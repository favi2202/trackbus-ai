from trackbus.detection import TrackedDetection
from trackbus.showcase_zones import (
    CrossingStateMachine,
    DualAnchorCrossingCounter,
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


def feed_dual(
    counter: DualAnchorCrossingCounter,
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


def test_dual_anchors_share_one_event_latch() -> None:
    counter = DualAnchorCrossingCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
        event_cooldown_frames=45,
    )
    events = feed_dual(
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
    assert counter.summary["suppressed_duplicate_events"] == 1


def test_shared_cooldown_suppresses_fast_same_track_recounts() -> None:
    counter = DualAnchorCrossingCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
        event_cooldown_frames=20,
    )
    events = feed_dual(
        counter,
        [
            tracked_box(10, 30),
            tracked_box(40, 60),
            tracked_box(55, 75),
            tracked_box(75, 90),
        ],
    )

    assert [event.direction for event in events] == ["IN"]
    assert counter.summary["suppressed_duplicate_events"] == 1


def test_shared_cooldown_allows_a_real_opposite_direction_crossing() -> None:
    counter = DualAnchorCrossingCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
        event_cooldown_frames=20,
    )
    events = feed_dual(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(45, 55),
            tracked_box(75, 85),
            tracked_box(45, 55),
            tracked_box(10, 20),
        ],
    )

    assert [event.direction for event in events] == ["IN", "OUT"]


def test_prediction_only_track_cannot_create_crossing_event() -> None:
    counter = DualAnchorCrossingCounter(
        ZoneLayout.default(),
        minimum_zone_frames=1,
    )
    events = feed_dual(
        counter,
        [
            tracked_box(10, 20, prediction_only=True),
            tracked_box(45, 55, prediction_only=True),
            tracked_box(75, 85, prediction_only=True),
        ],
    )

    assert events == []
    assert counter.summary["prediction_observations_ignored"] == 3


def test_real_endpoints_can_bridge_door_during_short_detection_gap() -> None:
    counter = DualAnchorCrossingCounter(
        ZoneLayout.default(),
        minimum_zone_frames=2,
        maximum_gap_frames=8,
    )
    events = feed_dual(
        counter,
        [
            tracked_box(10, 20),
            tracked_box(10, 20),
            tracked_box(75, 85),
            tracked_box(75, 85),
        ],
    )

    assert [event.direction for event in events] == ["IN"]
    assert counter.summary["geometric_door_bridges"] == 2
