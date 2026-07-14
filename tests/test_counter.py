from trackbus.counter import EventType, MovementState, PassengerCounter
from trackbus.zones import ZoneMembership

OUTSIDE = ZoneMembership.OUTSIDE
INSIDE = ZoneMembership.INSIDE
TRANSITION = ZoneMembership.TRANSITION


def observe_many(
    counter: PassengerCounter,
    tracking_id: int,
    zones: list[ZoneMembership],
):
    return [
        event
        for frame, zone in enumerate(zones)
        if (event := counter.observe(tracking_id, zone, frame)) is not None
    ]


def test_outside_to_inside_counts_exactly_one_entry() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=2)

    events = observe_many(counter, 7, [OUTSIDE, OUTSIDE, TRANSITION, INSIDE, INSIDE])

    assert [event.event_type for event in events] == [EventType.IN]
    assert counter.entered_total == 1
    assert counter.exited_total == 0
    assert counter.current_occupancy == 1


def test_inside_to_outside_counts_exactly_one_exit() -> None:
    counter = PassengerCounter(capacity=40, initial_occupancy=1, minimum_zone_frames=2)

    events = observe_many(counter, 8, [INSIDE, INSIDE, TRANSITION, OUTSIDE, OUTSIDE])

    assert [event.event_type for event in events] == [EventType.OUT]
    assert counter.exited_total == 1
    assert counter.current_occupancy == 0


def test_outside_transition_outside_counts_nothing() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=2)

    events = observe_many(counter, 1, [OUTSIDE, OUTSIDE, TRANSITION, OUTSIDE, OUTSIDE])

    assert events == []
    assert counter.track_state(1) is MovementState.OUTSIDE


def test_inside_transition_inside_counts_nothing() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=2)

    events = observe_many(counter, 1, [INSIDE, INSIDE, TRANSITION, INSIDE, INSIDE])

    assert events == []
    assert counter.track_state(1) is MovementState.INSIDE


def test_standing_in_one_zone_is_not_repeatedly_counted() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=2)

    events = observe_many(counter, 2, [OUTSIDE] * 20)

    assert events == []
    assert counter.entered_total == 0


def test_duplicate_destination_observations_create_one_event() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=2)

    events = observe_many(counter, 3, [OUTSIDE] * 2 + [INSIDE] * 20)

    assert len(events) == 1
    assert counter.entered_total == 1


def test_brief_destination_touch_does_not_count() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=3)

    events = observe_many(
        counter,
        4,
        [OUTSIDE] * 3 + [TRANSITION, INSIDE, TRANSITION] + [OUTSIDE] * 3,
    )

    assert events == []


def test_occupancy_never_becomes_negative() -> None:
    counter = PassengerCounter(capacity=40, minimum_zone_frames=1)

    observe_many(counter, 5, [INSIDE, OUTSIDE, INSIDE, OUTSIDE])

    assert counter.exited_total == 2
    assert counter.current_occupancy == 0


def test_stale_tracking_states_are_removed() -> None:
    counter = PassengerCounter(
        capacity=40, minimum_zone_frames=1, stale_track_timeout=5
    )
    counter.observe(10, OUTSIDE, 1)

    assert counter.remove_stale_tracks(5) == ()
    assert counter.remove_stale_tracks(6) == (10,)
    assert counter.active_track_count == 0


def test_same_track_can_make_one_crossing_in_each_direction() -> None:
    counter = PassengerCounter(capacity=40, initial_occupancy=1, minimum_zone_frames=2)

    events = observe_many(
        counter,
        11,
        [OUTSIDE] * 2 + [INSIDE] * 2 + [OUTSIDE] * 2,
    )

    assert [event.event_type for event in events] == [EventType.IN, EventType.OUT]
