from __future__ import annotations

from collections.abc import Iterable

from trackbus.counter import (
    CounterSnapshot,
    EventType,
    MovementState,
    PassengerCounter,
    SuppressionReason,
)
from trackbus.zones import ZoneMembership

OUTSIDE = ZoneMembership.OUTSIDE
INSIDE = ZoneMembership.INSIDE
TRANSITION = ZoneMembership.TRANSITION


def observe_at(
    counter: PassengerCounter,
    observations: Iterable[tuple[int, ZoneMembership]],
    *,
    tracking_id: int = 39,
) -> tuple[list, dict[int, CounterSnapshot]]:
    events = []
    snapshots: dict[int, CounterSnapshot] = {}
    for frame, zone in observations:
        event = counter.observe(tracking_id, zone, frame)
        snapshot = counter.track_snapshot(tracking_id)
        assert snapshot is not None
        snapshots[frame] = snapshot
        if event is not None:
            events.append(event)
    return events, snapshots


def test_initial_origin_requires_configured_dwell_before_crossing() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=3,
        minimum_destination_zone_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (3, INSIDE),
            (4, INSIDE),
            (5, INSIDE),
        ],
    )

    assert events == []
    assert snapshots[0].origin_dwell_frames == 1
    assert snapshots[1].origin_dwell_frames == 2
    assert snapshots[1].stable_zone is None
    assert snapshots[2].state is MovementState.UNKNOWN
    assert snapshots[5].stable_zone is INSIDE
    assert counter.entered_total == 0


def test_departure_requires_origin_dwell_after_aborted_transition() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=3,
        minimum_destination_zone_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, OUTSIDE),
            (3, TRANSITION),
            (4, OUTSIDE),
            (5, TRANSITION),
            (6, OUTSIDE),
            (7, OUTSIDE),
            (8, OUTSIDE),
            (9, TRANSITION),
            (10, INSIDE),
            (11, INSIDE),
        ],
    )

    assert [event.video_frame for event in events] == [11]
    assert (
        SuppressionReason.INSUFFICIENT_ORIGIN_DWELL in snapshots[5].suppression_reasons
    )
    assert snapshots[5].pending_direction is None


def test_direct_zone_flip_without_neutral_never_counts() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=2,
        minimum_destination_zone_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [(0, OUTSIDE), (1, OUTSIDE), (2, INSIDE), (3, INSIDE)],
    )

    assert events == []
    assert snapshots[3].stable_zone is OUTSIDE
    assert (
        SuppressionReason.MISSING_NEUTRAL_TRANSITION in snapshots[3].suppression_reasons
    )


def test_hysteresis_cannot_manufacture_required_neutral_traversal() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=2,
        minimum_destination_zone_frames=2,
        event_cooldown_frames=0,
    )

    counter.observe(39, OUTSIDE, 0, raw_zone=OUTSIDE)
    counter.observe(39, OUTSIDE, 1, raw_zone=OUTSIDE)
    # A shallow raw INSIDE point is made effectively neutral by polygon
    # hysteresis. It is boundary stabilization, not evidence of traversing the
    # actual neutral region.
    counter.observe(39, TRANSITION, 2, raw_zone=INSIDE)
    counter.observe(39, INSIDE, 3, raw_zone=INSIDE)
    event = counter.observe(39, INSIDE, 4, raw_zone=INSIDE)
    snapshot = counter.track_snapshot(39)

    assert event is None
    assert counter.entered_total == 0
    assert snapshot is not None
    assert snapshot.stable_zone is OUTSIDE
    assert SuppressionReason.BOUNDARY_JITTER in snapshot.suppression_reasons or (
        SuppressionReason.MISSING_NEUTRAL_TRANSITION in snapshot.suppression_reasons
    )


def test_destination_requires_consecutive_confirmation() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=2,
        minimum_destination_zone_frames=3,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (3, INSIDE),
            (4, INSIDE),
            (5, INSIDE),
        ],
    )

    assert [event.video_frame for event in events] == [5]
    assert snapshots[3].destination_confirmation_frames == 1
    assert snapshots[4].destination_confirmation_frames == 2
    assert (
        SuppressionReason.INSUFFICIENT_DESTINATION_CONFIRMATION
        in snapshots[4].suppression_reasons
    )
    assert snapshots[5].emitted_event is EventType.IN


def test_latched_destination_rejects_direct_opposite_jitter() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (3, INSIDE),
            (4, INSIDE),
            (5, OUTSIDE),
            (6, INSIDE),
            (7, OUTSIDE),
            (8, INSIDE),
        ],
    )

    assert [event.event_type for event in events] == [EventType.IN]
    assert snapshots[8].stable_zone is INSIDE
    assert counter.exited_total == 0


def test_boundary_oscillation_resets_destination_confirmation() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=2,
        minimum_destination_zone_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (3, INSIDE),
            (4, TRANSITION),
            (5, INSIDE),
            (6, TRANSITION),
            (7, OUTSIDE),
        ],
    )

    assert events == []
    assert snapshots[3].destination_confirmation_frames == 1
    assert snapshots[4].destination_confirmation_frames == 0
    assert snapshots[7].pending_direction is None
    assert snapshots[7].stable_zone is OUTSIDE


def test_short_detection_gap_preserves_pending_transition() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=2,
        maximum_transition_gap_frames=3,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (5, INSIDE),
            (6, INSIDE),
        ],
    )

    assert [event.video_frame for event in events] == [6]
    assert snapshots[5].detection_gap_frames == 2
    assert snapshots[5].pending_direction is EventType.IN
    assert (
        SuppressionReason.EXCESSIVE_DETECTION_GAP
        not in snapshots[5].suppression_reasons
    )


def test_long_detection_gap_invalidates_pending_transition() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=2,
        maximum_transition_gap_frames=2,
        event_cooldown_frames=0,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (6, INSIDE),
            (7, OUTSIDE),
            (8, OUTSIDE),
            (9, TRANSITION),
            (10, INSIDE),
            (11, INSIDE),
        ],
    )

    assert [event.video_frame for event in events] == [11]
    assert snapshots[6].detection_gap_frames == 3
    assert SuppressionReason.EXCESSIVE_DETECTION_GAP in snapshots[6].suppression_reasons
    assert snapshots[6].pending_direction is None


def test_cooldown_held_reverse_is_cancelled_when_track_returns() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=2,
        event_cooldown_frames=20,
    )

    events, snapshots = observe_at(
        counter,
        [
            (0, OUTSIDE),
            (1, OUTSIDE),
            (2, TRANSITION),
            (3, INSIDE),
            (4, INSIDE),
            (5, INSIDE),
            (6, INSIDE),
            (7, TRANSITION),
            (8, OUTSIDE),
            (9, OUTSIDE),
            (10, INSIDE),
            (11, OUTSIDE),
        ],
    )

    assert [(event.video_frame, event.event_type) for event in events] == [
        (4, EventType.IN)
    ]
    assert SuppressionReason.COOLDOWN in snapshots[9].suppression_reasons
    assert snapshots[10].pending_direction is None
    assert snapshots[11].stable_zone is INSIDE


def test_legitimate_delayed_reverse_crossing_counts_after_cooldown() -> None:
    counter = PassengerCounter(
        capacity=40,
        initial_occupancy=1,
        minimum_zone_frames=2,
        event_cooldown_frames=8,
    )
    observations = [
        (0, OUTSIDE),
        (1, OUTSIDE),
        (2, TRANSITION),
        (3, INSIDE),
        (4, INSIDE),
    ]
    observations.extend((frame, INSIDE) for frame in range(5, 13))
    observations.extend([(13, TRANSITION), (14, OUTSIDE), (15, OUTSIDE)])

    events, snapshots = observe_at(counter, observations)

    assert [(event.video_frame, event.event_type) for event in events] == [
        (4, EventType.IN),
        (15, EventType.OUT),
    ]
    assert snapshots[15].stable_zone is OUTSIDE
    assert counter.current_occupancy == 1


def test_repeated_destination_observations_do_not_duplicate_event() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=2,
        event_cooldown_frames=0,
    )
    observations = [(0, OUTSIDE), (1, OUTSIDE), (2, TRANSITION)]
    observations.extend((frame, INSIDE) for frame in range(3, 30))

    events, _ = observe_at(counter, observations)

    assert [(event.video_frame, event.event_type) for event in events] == [
        (4, EventType.IN)
    ]
    assert counter.entered_total == 1


def test_id39_in_boundary_oscillation_return_emits_only_original_in() -> None:
    counter = PassengerCounter(
        capacity=40,
        minimum_origin_zone_frames=2,
        minimum_destination_zone_frames=2,
        maximum_transition_gap_frames=15,
        event_cooldown_frames=30,
    )

    events, snapshots = observe_at(
        counter,
        [
            (350, OUTSIDE),
            (351, OUTSIDE),
            (352, TRANSITION),
            (353, INSIDE),
            (354, INSIDE),
            (355, INSIDE),
            (356, INSIDE),
            (369, TRANSITION),
            (371, OUTSIDE),
            (372, OUTSIDE),
            (373, TRANSITION),
            (374, OUTSIDE),
            (375, TRANSITION),
            (379, INSIDE),
            (380, INSIDE),
        ],
        tracking_id=39,
    )

    assert [
        (event.video_frame, event.tracking_id, event.event_type) for event in events
    ] == [(354, 39, EventType.IN)]
    assert SuppressionReason.COOLDOWN in snapshots[372].suppression_reasons
    assert snapshots[379].pending_direction is None
    assert snapshots[380].stable_zone is INSIDE
    assert counter.entered_total == 1
    assert counter.exited_total == 0
