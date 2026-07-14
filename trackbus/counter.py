"""Model-independent two-zone passenger counting state machine."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class TrackMemory:
    """Short-lived movement history for one ByteTrack ID."""

    state: MovementState = MovementState.UNKNOWN
    stable_zone: ZoneMembership | None = None
    candidate_zone: ZoneMembership | None = None
    candidate_frames: int = 0
    last_seen_frame: int = 0


class PassengerCounter:
    """Count confirmed transitions without depending on YOLO or video I/O.

    A track must remain in its destination zone for ``minimum_zone_frames``
    consecutive observations. Observations in the neutral transition area never
    count by themselves, and returning to the original zone cancels a crossing.
    """

    def __init__(
        self,
        *,
        capacity: int,
        initial_occupancy: int = 0,
        minimum_zone_frames: int = 3,
        stale_track_timeout: int = 90,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if initial_occupancy < 0:
            raise ValueError("initial_occupancy cannot be negative")
        if minimum_zone_frames < 1:
            raise ValueError("minimum_zone_frames must be at least 1")
        if stale_track_timeout < 1:
            raise ValueError("stale_track_timeout must be at least 1")

        self.capacity = capacity
        self.initial_occupancy = initial_occupancy
        self.minimum_zone_frames = minimum_zone_frames
        self.stale_track_timeout = stale_track_timeout
        self.entered_total = 0
        self.exited_total = 0
        self._tracks: dict[int, TrackMemory] = {}

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

    def observe(
        self,
        tracking_id: int,
        zone: ZoneMembership,
        video_frame: int,
    ) -> CountEvent | None:
        """Process one tracked person's zone observation for one video frame."""

        if tracking_id < 0:
            raise ValueError("tracking_id cannot be negative")
        if video_frame < 0:
            raise ValueError("video_frame cannot be negative")

        self.remove_stale_tracks(video_frame)
        memory = self._tracks.setdefault(
            tracking_id, TrackMemory(last_seen_frame=video_frame)
        )
        if video_frame < memory.last_seen_frame:
            raise ValueError("video_frame cannot move backwards for a track")
        memory.last_seen_frame = video_frame

        if zone is ZoneMembership.TRANSITION:
            memory.candidate_zone = None
            memory.candidate_frames = 0
            if memory.stable_zone is ZoneMembership.OUTSIDE:
                memory.state = MovementState.TRANSITIONING_IN
            elif memory.stable_zone is ZoneMembership.INSIDE:
                memory.state = MovementState.TRANSITIONING_OUT
            return None

        if zone is memory.candidate_zone:
            memory.candidate_frames += 1
        else:
            memory.candidate_zone = zone
            memory.candidate_frames = 1

        if memory.candidate_frames < self.minimum_zone_frames:
            return None

        if memory.stable_zone is None:
            self._settle(memory, zone)
            return None

        if zone is memory.stable_zone:
            self._settle(memory, zone)
            return None

        event_type = (
            EventType.IN
            if memory.stable_zone is ZoneMembership.OUTSIDE
            and zone is ZoneMembership.INSIDE
            else EventType.OUT
        )
        if event_type is EventType.IN:
            self.entered_total += 1
        else:
            self.exited_total += 1
        self._settle(memory, zone)
        return CountEvent(
            video_frame=video_frame,
            tracking_id=tracking_id,
            event_type=event_type,
            entered_total=self.entered_total,
            exited_total=self.exited_total,
            current_occupancy=self.current_occupancy,
            occupancy_percentage=self.occupancy_percentage,
        )

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

    @staticmethod
    def _settle(memory: TrackMemory, zone: ZoneMembership) -> None:
        memory.stable_zone = zone
        memory.state = (
            MovementState.OUTSIDE
            if zone is ZoneMembership.OUTSIDE
            else MovementState.INSIDE
        )
        memory.candidate_zone = zone
        memory.candidate_frames = 0
