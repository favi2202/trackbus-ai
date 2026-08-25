"""Three-zone, temporary-ID crossing logic for the executable showcase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from trackbus.detection import TrackedDetection


class Zone(StrEnum):
    OUTSIDE = "outside"
    DOOR = "door"
    INSIDE = "inside"
    UNKNOWN = "unknown"


NormalizedPolygon = tuple[tuple[float, float], ...]


def _validate_polygon(polygon: NormalizedPolygon, name: str) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{name} must have at least three points")
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in polygon):
        raise ValueError(f"{name} points must be normalized from 0 to 1")


def point_in_polygon(point: tuple[float, float], polygon: NormalizedPolygon) -> bool:
    """Ray-cast point membership, treating a polygon edge as inside."""

    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-9
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return True
        if (y1 > y) != (y2 > y):
            intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= intersection:
                inside = not inside
        previous = current
    return inside


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    """Return true when two closed line segments touch or cross."""

    values = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True

    def on_segment(
        start: tuple[float, float],
        point: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        return (
            abs(_orientation(start, point, end)) < 1e-9
            and min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
            and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
        )

    return (
        on_segment(first_start, second_start, first_end)
        or on_segment(first_start, second_end, first_end)
        or on_segment(second_start, first_start, second_end)
        or on_segment(second_start, first_end, second_end)
    )


@dataclass(frozen=True)
class ZoneLayout:
    outside: NormalizedPolygon
    door: NormalizedPolygon
    inside: NormalizedPolygon

    def __post_init__(self) -> None:
        for name in ("outside", "door", "inside"):
            _validate_polygon(getattr(self, name), name)

    @classmethod
    def default(cls) -> ZoneLayout:
        return cls(
            outside=((0.04, 0.04), (0.96, 0.04), (0.96, 0.36), (0.04, 0.36)),
            door=((0.04, 0.36), (0.96, 0.36), (0.96, 0.64), (0.04, 0.64)),
            inside=((0.04, 0.64), (0.96, 0.64), (0.96, 0.96), (0.04, 0.96)),
        )

    def classify(
        self, point: tuple[float, float], frame_shape: tuple[int, ...]
    ) -> Zone:
        height, width = frame_shape[:2]
        normalized = point[0] / max(1, width - 1), point[1] / max(1, height - 1)
        for zone in (Zone.DOOR, Zone.OUTSIDE, Zone.INSIDE):
            if point_in_polygon(normalized, getattr(self, zone.value)):
                return zone
        return Zone.UNKNOWN

    def pixel_polygons(
        self, frame_shape: tuple[int, ...]
    ) -> dict[Zone, list[tuple[int, int]]]:
        height, width = frame_shape[:2]
        return {
            zone: [
                (round(x * (width - 1)), round(y * (height - 1)))
                for x, y in getattr(self, zone.value)
            ]
            for zone in (Zone.OUTSIDE, Zone.DOOR, Zone.INSIDE)
        }

    def segment_crosses_zone(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        zone: Zone,
        frame_shape: tuple[int, ...],
    ) -> bool:
        """Test whether a real-to-real anchor segment crosses one zone polygon."""

        if zone not in {Zone.OUTSIDE, Zone.DOOR, Zone.INSIDE}:
            return False
        height, width = frame_shape[:2]
        normalized_start = (
            start[0] / max(1, width - 1),
            start[1] / max(1, height - 1),
        )
        normalized_end = (
            end[0] / max(1, width - 1),
            end[1] / max(1, height - 1),
        )
        polygon = getattr(self, zone.value)
        if point_in_polygon(normalized_start, polygon) or point_in_polygon(
            normalized_end, polygon
        ):
            return True
        return any(
            _segments_intersect(
                normalized_start,
                normalized_end,
                polygon[index - 1],
                polygon[index],
            )
            for index in range(len(polygon))
        )


def load_zone_layout(path: str | Path | None) -> ZoneLayout:
    if path is None:
        return ZoneLayout.default()
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("zone file must contain a mapping")
    if isinstance(document.get("zones"), dict):
        document = document["zones"]

    def polygon(name: str) -> NormalizedPolygon:
        value = document.get(name)
        if not isinstance(value, list):
            raise ValueError(f"zone file is missing '{name}'")
        try:
            return tuple((float(point[0]), float(point[1])) for point in value)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"zone '{name}' must contain [x, y] points") from exc

    return ZoneLayout(
        outside=polygon("outside"), door=polygon("door"), inside=polygon("inside")
    )


@dataclass(frozen=True)
class CrossingEvent:
    track_id: int
    direction: str
    frame_number: int
    confidence: float
    dwell_frames: int
    path: tuple[Zone, Zone, Zone]

    @property
    def occupancy_delta(self) -> int:
        return 1 if self.direction == "IN" else -1


@dataclass
class _TrackState:
    candidate: Zone = Zone.UNKNOWN
    candidate_frames: int = 0
    stable: Zone = Zone.UNKNOWN
    origin: Zone = Zone.UNKNOWN
    saw_door: bool = False
    last_seen: int = 0
    confidence_samples: list[float] = field(default_factory=list)
    journey_frames: int = 0


class CrossingStateMachine:
    """Accept only stable OUTSIDE↔DOOR↔INSIDE paths for each temporary ID."""

    def __init__(
        self, *, minimum_zone_frames: int = 2, maximum_gap_frames: int = 45
    ) -> None:
        if minimum_zone_frames < 1 or maximum_gap_frames < 1:
            raise ValueError("zone dwell and maximum gap must be positive")
        self.minimum_zone_frames = minimum_zone_frames
        self.maximum_gap_frames = maximum_gap_frames
        self._tracks: dict[int, _TrackState] = {}

    def observe(
        self,
        *,
        track_id: int,
        zone: Zone,
        frame_number: int,
        confidence: float,
        bridged_door: bool = False,
    ) -> CrossingEvent | None:
        state = self._tracks.setdefault(track_id, _TrackState(last_seen=frame_number))
        if frame_number - state.last_seen > self.maximum_gap_frames:
            state = _TrackState(last_seen=frame_number)
            self._tracks[track_id] = state
        state.last_seen = frame_number
        state.journey_frames += 1
        state.confidence_samples.append(max(0.0, min(1.0, confidence)))

        if (
            bridged_door
            and state.origin in {Zone.OUTSIDE, Zone.INSIDE}
            and zone
            == (Zone.INSIDE if state.origin is Zone.OUTSIDE else Zone.OUTSIDE)
        ):
            # Both segment endpoints are real observations.  The bridge records
            # only geometric evidence that their connecting path crossed DOOR;
            # no predicted-only point can confirm an event.
            state.saw_door = True

        if zone is Zone.UNKNOWN:
            state.candidate = Zone.UNKNOWN
            state.candidate_frames = 0
            return None
        if zone == state.candidate:
            state.candidate_frames += 1
        else:
            state.candidate = zone
            state.candidate_frames = 1
        if state.candidate_frames < self.minimum_zone_frames or zone == state.stable:
            return None

        state.stable = zone
        if state.origin is Zone.UNKNOWN:
            if zone in {Zone.OUTSIDE, Zone.INSIDE}:
                self._start_journey(state, zone, confidence)
            return None
        if zone is Zone.DOOR:
            state.saw_door = True
            return None
        if zone == state.origin:
            state.saw_door = False
            state.confidence_samples = [confidence]
            state.journey_frames = 1
            return None

        destination = Zone.INSIDE if state.origin is Zone.OUTSIDE else Zone.OUTSIDE
        if zone != destination:
            return None
        if not state.saw_door:
            self._start_journey(state, destination, confidence)
            return None

        origin = state.origin
        direction = "IN" if origin is Zone.OUTSIDE else "OUT"
        event = CrossingEvent(
            track_id=track_id,
            direction=direction,
            frame_number=frame_number,
            confidence=sum(state.confidence_samples) / len(state.confidence_samples),
            dwell_frames=state.journey_frames,
            path=(origin, Zone.DOOR, destination),
        )
        self._start_journey(state, destination, confidence)
        return event

    def expire(self, frame_number: int) -> None:
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if frame_number - state.last_seen <= self.maximum_gap_frames
        }

    def reset(self) -> None:
        self._tracks.clear()

    @staticmethod
    def _start_journey(state: _TrackState, origin: Zone, confidence: float) -> None:
        state.origin = origin
        state.saw_door = False
        state.confidence_samples = [confidence]
        state.journey_frames = 1


@dataclass(frozen=True)
class _AnchorObservation:
    point: tuple[float, float]
    zone: Zone
    frame_number: int


class DualAnchorCrossingCounter:
    """Fuse top/bottom box anchors into one deduplicated crossing decision.

    Each anchor has independent zone history, but both share one event latch per
    stable temporary track.  Prediction-only tracks are ignored by construction.
    A short real-observation gap can bridge the DOOR polygon geometrically.
    """

    ANCHORS = ("bottom_center", "top_center")

    def __init__(
        self,
        layout: ZoneLayout,
        *,
        minimum_zone_frames: int = 2,
        maximum_gap_frames: int = 45,
        event_cooldown_frames: int = 45,
    ) -> None:
        if event_cooldown_frames < 0:
            raise ValueError("event cooldown cannot be negative")
        self.layout = layout
        self.maximum_gap_frames = maximum_gap_frames
        self.event_cooldown_frames = event_cooldown_frames
        self._machines = {
            anchor: CrossingStateMachine(
                minimum_zone_frames=minimum_zone_frames,
                maximum_gap_frames=maximum_gap_frames,
            )
            for anchor in self.ANCHORS
        }
        self._last_observations: dict[
            str, dict[int, _AnchorObservation]
        ] = {anchor: {} for anchor in self.ANCHORS}
        self._last_events: dict[int, tuple[int, str]] = {}
        self._suppressed_duplicates = 0
        self._conflicting_anchor_events = 0
        self._geometric_door_bridges = 0
        self._prediction_observations_ignored = 0

    @property
    def summary(self) -> dict[str, int]:
        return {
            "suppressed_duplicate_events": self._suppressed_duplicates,
            "conflicting_anchor_events": self._conflicting_anchor_events,
            "geometric_door_bridges": self._geometric_door_bridges,
            "prediction_observations_ignored": (
                self._prediction_observations_ignored
            ),
        }

    def anchor_zones(
        self, track: TrackedDetection, frame_shape: tuple[int, ...]
    ) -> dict[str, Zone]:
        return {
            anchor: self.layout.classify(track.anchor_for(anchor), frame_shape)
            for anchor in self.ANCHORS
        }

    def display_zone(
        self, track: TrackedDetection, frame_shape: tuple[int, ...]
    ) -> Zone:
        zones = tuple(self.anchor_zones(track, frame_shape).values())
        if Zone.DOOR in zones:
            return Zone.DOOR
        known = tuple(zone for zone in zones if zone is not Zone.UNKNOWN)
        if not known:
            return Zone.UNKNOWN
        if len(set(known)) == 1:
            return known[0]
        # A tall box spanning both endpoints is visually in the transition.
        return Zone.DOOR

    def observe(
        self,
        *,
        track: TrackedDetection,
        frame_number: int,
        frame_shape: tuple[int, ...],
    ) -> CrossingEvent | None:
        if bool(track.metadata.get("prediction_only")):
            self._prediction_observations_ignored += 1
            return None

        candidates: list[CrossingEvent] = []
        for anchor, zone in self.anchor_zones(track, frame_shape).items():
            point = track.anchor_for(anchor)
            previous = self._last_observations[anchor].get(track.tracking_id)
            bridged_door = self._can_bridge_door(
                previous,
                point,
                zone,
                frame_number,
                frame_shape,
            )
            if bridged_door:
                self._geometric_door_bridges += 1
            event = self._machines[anchor].observe(
                track_id=track.tracking_id,
                zone=zone,
                frame_number=frame_number,
                confidence=track.confidence,
                bridged_door=bridged_door,
            )
            self._last_observations[anchor][track.tracking_id] = _AnchorObservation(
                point=point,
                zone=zone,
                frame_number=frame_number,
            )
            if event is not None:
                candidates.append(event)

        if not candidates:
            return None
        if len({event.direction for event in candidates}) > 1:
            self._conflicting_anchor_events += len(candidates)
            return None

        last_event = self._last_events.get(track.tracking_id)
        if (
            last_event is not None
            and candidates[0].direction == last_event[1]
            and frame_number - last_event[0] <= self.event_cooldown_frames
        ):
            self._suppressed_duplicates += len(candidates)
            return None

        chosen = max(
            candidates,
            key=lambda event: (event.confidence, event.dwell_frames),
        )
        self._suppressed_duplicates += max(0, len(candidates) - 1)
        self._last_events[track.tracking_id] = (frame_number, chosen.direction)
        return chosen

    def expire(self, frame_number: int) -> None:
        for machine in self._machines.values():
            machine.expire(frame_number)
        for anchor, observations in self._last_observations.items():
            self._last_observations[anchor] = {
                track_id: observation
                for track_id, observation in observations.items()
                if frame_number - observation.frame_number <= self.maximum_gap_frames
            }
        retention = max(self.maximum_gap_frames, self.event_cooldown_frames) * 2 + 1
        self._last_events = {
            track_id: event
            for track_id, event in self._last_events.items()
            if frame_number - event[0] <= retention
        }

    def reset(self) -> None:
        for machine in self._machines.values():
            machine.reset()
        for observations in self._last_observations.values():
            observations.clear()
        self._last_events.clear()
        self._suppressed_duplicates = 0
        self._conflicting_anchor_events = 0
        self._geometric_door_bridges = 0
        self._prediction_observations_ignored = 0

    def _can_bridge_door(
        self,
        previous: _AnchorObservation | None,
        point: tuple[float, float],
        zone: Zone,
        frame_number: int,
        frame_shape: tuple[int, ...],
    ) -> bool:
        if previous is None:
            return False
        if frame_number - previous.frame_number > self.maximum_gap_frames:
            return False
        if {previous.zone, zone} != {Zone.OUTSIDE, Zone.INSIDE}:
            return False
        return self.layout.segment_crosses_zone(
            previous.point,
            point,
            Zone.DOOR,
            frame_shape,
        )
