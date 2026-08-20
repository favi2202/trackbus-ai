"""Three-zone, temporary-ID crossing logic for the executable showcase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


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
    ) -> CrossingEvent | None:
        state = self._tracks.setdefault(track_id, _TrackState(last_seen=frame_number))
        if frame_number - state.last_seen > self.maximum_gap_frames:
            state = _TrackState(last_seen=frame_number)
            self._tracks[track_id] = state
        state.last_seen = frame_number
        state.journey_frames += 1
        state.confidence_samples.append(max(0.0, min(1.0, confidence)))

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
