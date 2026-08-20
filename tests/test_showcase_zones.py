from trackbus.showcase_zones import CrossingStateMachine, Zone, ZoneLayout


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
