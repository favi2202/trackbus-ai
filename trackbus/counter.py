"""Model-independent, latched two-zone passenger counting state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from trackbus.zones import ZoneMembership


class MovementState(StrEnum):
    """Human-readable state retained for each active tracking ID."""

    UNKNOWN = "unknown"
    OUTSIDE = "outside"
    TRANSITIONING_IN = "transitioning_in"
    INSIDE = "inside"
    TRANSITIONING_OUT = "transitioning_out"


class EventType(StrEnum):
    """A completed doorway crossing direction."""

    IN = "IN"
    OUT = "OUT"


class SuppressionReason(StrEnum):
    """Why the current observation did not produce a crossing event."""

    COOLDOWN = "cooldown"
    INSUFFICIENT_ORIGIN_DWELL = "insufficient_origin_dwell"
    INSUFFICIENT_DESTINATION_CONFIRMATION = "insufficient_destination_confirmation"
    MISSING_NEUTRAL_TRANSITION = "missing_neutral_transition"
    BOUNDARY_JITTER = "boundary_jitter"
    EXCESSIVE_DETECTION_GAP = "excessive_detection_gap"


@dataclass(frozen=True)
class CountEvent:
    """Snapshot of totals when one complete crossing is confirmed."""

    video_frame: int
    tracking_id: int
    event_type: EventType
    entered_total: int
    exited_total: int
    current_occupancy: int
    occupancy_percentage: float


@dataclass(frozen=True)
class CounterSnapshot:
    """Per-observation state exported only when diagnostics are enabled."""

    raw_zone: ZoneMembership
    effective_zone: ZoneMembership
    stable_zone: ZoneMembership | None
    state: MovementState
    pending_direction: EventType | None
    origin_dwell_frames: int
    destination_confirmation_frames: int
    cooldown_remaining: int
    suppression_reasons: tuple[SuppressionReason, ...]
    emitted_event: EventType | None
    detection_gap_frames: int


@dataclass
class TrackMemory:
    """Short-lived, latched movement history for one ByteTrack ID."""

    state: MovementState = MovementState.UNKNOWN
    stable_zone: ZoneMembership | None = None
    initial_candidate_zone: ZoneMembership | None = None
    initial_candidate_frames: int = 0
    origin_dwell_frames: int = 0
    pending_direction: EventType | None = None
    destination_confirmation_frames: int = 0
    last_seen_frame: int = 0
    last_event_frame: int | None = None
    last_snapshot: CounterSnapshot | None = None
    suppression_counts: dict[SuppressionReason, int] = field(default_factory=dict)


class PassengerCounter:
    """Count only complete, stable, neutral-mediated zone transitions.

    A new ID first establishes an origin by dwelling in one explicit zone. A
    crossing can then begin only with an observed neutral-region sample. The
    destination must be confirmed before an event is emitted and latched. A
    reverse crossing follows the same complete process; a direct zone flip can
    never rebase the latch.
    """

    def __init__(
        self,
        *,
        capacity: int,
        initial_occupancy: int = 0,
        minimum_zone_frames: int = 3,
        minimum_origin_zone_frames: int | None = None,
        minimum_destination_zone_frames: int | None = None,
        maximum_transition_gap_frames: int = 15,
        event_cooldown_frames: int = 30,
        stale_track_timeout: int = 90,
    ) -> None:
        origin_frames = (
            minimum_zone_frames
            if minimum_origin_zone_frames is None
            else minimum_origin_zone_frames
        )
        destination_frames = (
            minimum_zone_frames
            if minimum_destination_zone_frames is None
            else minimum_destination_zone_frames
        )
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if initial_occupancy < 0:
            raise ValueError("initial_occupancy cannot be negative")
        if minimum_zone_frames < 1:
            raise ValueError("minimum_zone_frames must be at least 1")
        if origin_frames < 1:
            raise ValueError("minimum_origin_zone_frames must be at least 1")
        if destination_frames < 1:
            raise ValueError("minimum_destination_zone_frames must be at least 1")
        if maximum_transition_gap_frames < 0:
            raise ValueError("maximum_transition_gap_frames cannot be negative")
        if event_cooldown_frames < 0:
            raise ValueError("event_cooldown_frames cannot be negative")
        if stale_track_timeout < 1:
            raise ValueError("stale_track_timeout must be at least 1")

        self.capacity = capacity
        self.initial_occupancy = initial_occupancy
        self.minimum_zone_frames = minimum_zone_frames
        self.minimum_origin_zone_frames = origin_frames
        self.minimum_destination_zone_frames = destination_frames
        self.maximum_transition_gap_frames = maximum_transition_gap_frames
        self.event_cooldown_frames = event_cooldown_frames
        self.stale_track_timeout = stale_track_timeout
        self.entered_total = 0
        self.exited_total = 0
        self._tracks: dict[int, TrackMemory] = {}
        self._suppression_total_counts: dict[SuppressionReason, int] = {}

    @property
    def current_occupancy(self) -> int:
        """Estimated occupancy, clamped because an estimate cannot be negative."""

        return max(0, self.initial_occupancy + self.entered_total - self.exited_total)

    @property
    def occupancy_percentage(self) -> float:
        """Actual capacity percentage; values above 100 indicate over-capacity."""

        return self.current_occupancy / self.capacity * 100.0

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def track_state(self, tracking_id: int) -> MovementState | None:
        """Return a track state for diagnostics and annotation."""

        memory = self._tracks.get(tracking_id)
        return memory.state if memory else None

    def track_stable_zone(self, tracking_id: int) -> ZoneMembership | None:
        memory = self._tracks.get(tracking_id)
        return memory.stable_zone if memory else None

    def track_snapshot(self, tracking_id: int) -> CounterSnapshot | None:
        memory = self._tracks.get(tracking_id)
        return memory.last_snapshot if memory else None

    def suppression_totals(self) -> dict[str, int]:
        """Aggregate suppression observations for the processing summary."""

        return {
            reason.value: self._suppression_total_counts.get(reason, 0)
            for reason in SuppressionReason
        }

    def observe(
        self,
        tracking_id: int,
        zone: ZoneMembership,
        video_frame: int,
        *,
        raw_zone: ZoneMembership | None = None,
    ) -> CountEvent | None:
        """Process one tracked person's effective zone observation."""

        if tracking_id < 0:
            raise ValueError("tracking_id cannot be negative")
        if video_frame < 0:
            raise ValueError("video_frame cannot be negative")

        self.remove_stale_tracks(video_frame)
        memory = self._tracks.get(tracking_id)
        is_new = memory is None
        if memory is None:
            memory = TrackMemory(last_seen_frame=video_frame)
            self._tracks[tracking_id] = memory
        if video_frame < memory.last_seen_frame:
            raise ValueError("video_frame cannot move backwards for a track")
        gap = 0 if is_new else video_frame - memory.last_seen_frame - 1
        reasons: list[SuppressionReason] = []
        observed_raw = raw_zone or zone
        if observed_raw is not zone:
            reasons.append(SuppressionReason.BOUNDARY_JITTER)

        if gap > self.maximum_transition_gap_frames:
            if (
                memory.pending_direction is not None
                or memory.destination_confirmation_frames
                or memory.initial_candidate_frames
                or memory.origin_dwell_frames
            ):
                reasons.append(SuppressionReason.EXCESSIVE_DETECTION_GAP)
            self._cancel_pending(memory)
            memory.initial_candidate_zone = None
            memory.initial_candidate_frames = 0
            memory.origin_dwell_frames = 0
        memory.last_seen_frame = video_frame

        # Boundary hysteresis may turn a shallow explicit-zone hit into an
        # effective neutral sample. It must never manufacture the mandatory
        # evidence that a track traversed the raw neutral region. Before a
        # transition starts, use the raw explicit side (holding the latch when
        # it is the origin and rejecting a direct flip when it is opposite).
        # Once a raw-neutral transition is already pending, a shallow
        # destination hit stays neutral until it is deep enough to confirm.
        state_machine_zone = self._state_machine_zone(memory, zone, observed_raw)

        event: CountEvent | None
        if memory.stable_zone is None:
            event = self._observe_without_origin(memory, state_machine_zone, reasons)
        elif memory.pending_direction is None:
            event = self._observe_latched(memory, state_machine_zone, reasons)
        else:
            event = self._observe_pending(
                memory,
                state_machine_zone,
                video_frame,
                tracking_id,
                reasons,
            )

        self._snapshot(
            memory,
            raw_zone=observed_raw,
            effective_zone=zone,
            gap=gap,
            reasons=reasons,
            event=event,
            video_frame=video_frame,
        )
        return event

    def remove_stale_tracks(self, current_frame: int) -> tuple[int, ...]:
        """Forget IDs not observed for the configured number of frames."""

        stale_ids = tuple(
            tracking_id
            for tracking_id, memory in self._tracks.items()
            if current_frame - memory.last_seen_frame >= self.stale_track_timeout
        )
        for tracking_id in stale_ids:
            del self._tracks[tracking_id]
        return stale_ids

    def _observe_without_origin(
        self,
        memory: TrackMemory,
        zone: ZoneMembership,
        reasons: list[SuppressionReason],
    ) -> None:
        memory.state = MovementState.UNKNOWN
        if zone is ZoneMembership.TRANSITION:
            memory.initial_candidate_zone = None
            memory.initial_candidate_frames = 0
            return None
        if zone is memory.initial_candidate_zone:
            memory.initial_candidate_frames += 1
        else:
            memory.initial_candidate_zone = zone
            memory.initial_candidate_frames = 1
        if memory.initial_candidate_frames < self.minimum_origin_zone_frames:
            reasons.append(SuppressionReason.INSUFFICIENT_ORIGIN_DWELL)
            return None
        memory.stable_zone = zone
        memory.origin_dwell_frames = memory.initial_candidate_frames
        memory.initial_candidate_zone = None
        memory.initial_candidate_frames = 0
        memory.state = self._stable_state(zone)
        return None

    def _observe_latched(
        self,
        memory: TrackMemory,
        zone: ZoneMembership,
        reasons: list[SuppressionReason],
    ) -> None:
        stable_zone = memory.stable_zone
        if stable_zone is None:  # pragma: no cover - caller invariant
            return None
        if zone is stable_zone:
            memory.origin_dwell_frames += 1
            memory.state = self._stable_state(stable_zone)
            return None
        if zone is ZoneMembership.TRANSITION:
            if memory.origin_dwell_frames < self.minimum_origin_zone_frames:
                reasons.append(SuppressionReason.INSUFFICIENT_ORIGIN_DWELL)
                memory.origin_dwell_frames = 0
                memory.state = self._stable_state(stable_zone)
                return None
            memory.pending_direction = self._direction_from(stable_zone)
            memory.destination_confirmation_frames = 0
            memory.state = self._transition_state(memory.pending_direction)
            return None

        reasons.append(SuppressionReason.MISSING_NEUTRAL_TRANSITION)
        memory.origin_dwell_frames = 0
        memory.state = self._stable_state(stable_zone)
        return None

    def _observe_pending(
        self,
        memory: TrackMemory,
        zone: ZoneMembership,
        video_frame: int,
        tracking_id: int,
        reasons: list[SuppressionReason],
    ) -> CountEvent | None:
        stable_zone = memory.stable_zone
        direction = memory.pending_direction
        if stable_zone is None or direction is None:  # pragma: no cover
            return None
        if zone is stable_zone:
            self._cancel_pending(memory)
            memory.origin_dwell_frames = 1
            memory.state = self._stable_state(stable_zone)
            return None
        if zone is ZoneMembership.TRANSITION:
            memory.destination_confirmation_frames = 0
            memory.state = self._transition_state(direction)
            return None

        memory.destination_confirmation_frames += 1
        memory.state = self._transition_state(direction)
        if (
            memory.destination_confirmation_frames
            < self.minimum_destination_zone_frames
        ):
            reasons.append(SuppressionReason.INSUFFICIENT_DESTINATION_CONFIRMATION)
            return None

        cooldown = self._cooldown_remaining(memory, video_frame)
        if cooldown > 0:
            reasons.append(SuppressionReason.COOLDOWN)
            return None

        if direction is EventType.IN:
            self.entered_total += 1
            destination = ZoneMembership.INSIDE
        else:
            self.exited_total += 1
            destination = ZoneMembership.OUTSIDE
        memory.stable_zone = destination
        memory.state = self._stable_state(destination)
        memory.origin_dwell_frames = 0
        memory.pending_direction = None
        memory.destination_confirmation_frames = 0
        memory.last_event_frame = video_frame
        return CountEvent(
            video_frame=video_frame,
            tracking_id=tracking_id,
            event_type=direction,
            entered_total=self.entered_total,
            exited_total=self.exited_total,
            current_occupancy=self.current_occupancy,
            occupancy_percentage=self.occupancy_percentage,
        )

    def _snapshot(
        self,
        memory: TrackMemory,
        *,
        raw_zone: ZoneMembership,
        effective_zone: ZoneMembership,
        gap: int,
        reasons: list[SuppressionReason],
        event: CountEvent | None,
        video_frame: int,
    ) -> None:
        unique_reasons = tuple(dict.fromkeys(reasons))
        for reason in unique_reasons:
            memory.suppression_counts[reason] = (
                memory.suppression_counts.get(reason, 0) + 1
            )
            self._suppression_total_counts[reason] = (
                self._suppression_total_counts.get(reason, 0) + 1
            )
        memory.last_snapshot = CounterSnapshot(
            raw_zone=raw_zone,
            effective_zone=effective_zone,
            stable_zone=memory.stable_zone,
            state=memory.state,
            pending_direction=memory.pending_direction,
            origin_dwell_frames=(
                memory.origin_dwell_frames
                if memory.stable_zone is not None
                else memory.initial_candidate_frames
            ),
            destination_confirmation_frames=(memory.destination_confirmation_frames),
            cooldown_remaining=self._cooldown_remaining(memory, video_frame),
            suppression_reasons=unique_reasons,
            emitted_event=event.event_type if event is not None else None,
            detection_gap_frames=gap,
        )

    def _cooldown_remaining(self, memory: TrackMemory, video_frame: int) -> int:
        if memory.last_event_frame is None:
            return 0
        return max(
            0,
            self.event_cooldown_frames - (video_frame - memory.last_event_frame),
        )

    @staticmethod
    def _state_machine_zone(
        memory: TrackMemory,
        effective_zone: ZoneMembership,
        raw_zone: ZoneMembership,
    ) -> ZoneMembership:
        if raw_zone is effective_zone:
            return effective_zone
        if memory.stable_zone is None:
            return effective_zone
        if memory.pending_direction is None:
            return raw_zone
        if raw_zone is memory.stable_zone:
            return raw_zone
        return effective_zone

    @staticmethod
    def _cancel_pending(memory: TrackMemory) -> None:
        memory.pending_direction = None
        memory.destination_confirmation_frames = 0

    @staticmethod
    def _direction_from(zone: ZoneMembership) -> EventType:
        return EventType.IN if zone is ZoneMembership.OUTSIDE else EventType.OUT

    @staticmethod
    def _stable_state(zone: ZoneMembership) -> MovementState:
        return (
            MovementState.OUTSIDE
            if zone is ZoneMembership.OUTSIDE
            else MovementState.INSIDE
        )

    @staticmethod
    def _transition_state(direction: EventType) -> MovementState:
        return (
            MovementState.TRANSITIONING_IN
            if direction is EventType.IN
            else MovementState.TRANSITIONING_OUT
        )
