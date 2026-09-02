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


def _polygon_center(polygon: NormalizedPolygon) -> tuple[float, float]:
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


@dataclass(frozen=True)
class GateGeometry:
    """One calibrated OUTSIDE-to-INSIDE axis with two ordered gates."""

    origin: tuple[float, float]
    axis: tuple[float, float]
    span: float
    outside_gate: float
    inside_gate: float

    @classmethod
    def from_layout(cls, layout: ZoneLayout) -> GateGeometry:
        outside = _polygon_center(layout.outside)
        door = _polygon_center(layout.door)
        inside = _polygon_center(layout.inside)
        delta = inside[0] - outside[0], inside[1] - outside[1]
        span = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        if span <= 1e-6:
            raise ValueError("outside and inside zone centers must be distinct")
        axis = delta[0] / span, delta[1] / span
        door_progress = (
            (door[0] - outside[0]) * axis[0]
            + (door[1] - outside[1]) * axis[1]
        ) / span
        if not 0.05 < door_progress < 0.95:
            raise ValueError("door zone center must lie between outside and inside")
        return cls(
            origin=outside,
            axis=axis,
            span=span,
            outside_gate=door_progress / 2,
            inside_gate=(door_progress + 1) / 2,
        )

    def progress(
        self, point: tuple[float, float], frame_shape: tuple[int, ...]
    ) -> float:
        height, width = frame_shape[:2]
        normalized = (
            point[0] / max(1, width - 1),
            point[1] / max(1, height - 1),
        )
        return (
            (normalized[0] - self.origin[0]) * self.axis[0]
            + (normalized[1] - self.origin[1]) * self.axis[1]
        ) / self.span


@dataclass(frozen=True)
class TrajectoryDecision:
    frame_number: int
    track_id: int
    action: str
    reason: str
    progress: float | None = None
    direction: str | None = None
    origin: Zone | None = None
    direction_consistency: float | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "recordType": "trajectory-decision",
            "frameNumber": self.frame_number,
            "trackId": f"T{self.track_id}",
            "action": self.action,
            "reason": self.reason,
        }
        if self.progress is not None:
            payload["progress"] = round(self.progress, 6)
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.origin is not None:
            payload["origin"] = self.origin.value
        if self.direction_consistency is not None:
            payload["directionConsistency"] = round(
                self.direction_consistency, 6
            )
        return payload


@dataclass
class _GateTrackState:
    last_seen: int
    candidate_origin: Zone = Zone.UNKNOWN
    candidate_frames: int = 0
    origin: Zone = Zone.UNKNOWN
    origin_frame: int = 0
    first_gate_crossed: bool = False
    destination_frames: int = 0
    origin_return_frames: int = 0
    samples: list[tuple[int, float, float]] = field(default_factory=list)


class MonotonicGateCounter:
    """Count one real center trajectory crossing two calibrated gates in order."""

    def __init__(
        self,
        layout: ZoneLayout,
        *,
        minimum_zone_frames: int = 2,
        maximum_gap_frames: int = 45,
        event_cooldown_frames: int = 90,
        minimum_direction_consistency: float = 0.70,
        gate_hysteresis: float = 0.03,
        minimum_journey_frames: int = 3,
    ) -> None:
        if minimum_zone_frames < 1 or maximum_gap_frames < 1:
            raise ValueError("zone dwell and maximum gap must be positive")
        if event_cooldown_frames < 0:
            raise ValueError("event cooldown cannot be negative")
        if not 0.5 <= minimum_direction_consistency <= 1:
            raise ValueError("direction consistency must be between 0.5 and 1")
        if not 0 <= gate_hysteresis < 0.20:
            raise ValueError("gate hysteresis must be between 0 and 0.2")
        if minimum_journey_frames < 2:
            raise ValueError("minimum journey frames must be at least 2")
        self.layout = layout
        self.geometry = GateGeometry.from_layout(layout)
        if self.geometry.inside_gate - self.geometry.outside_gate <= (
            gate_hysteresis * 2
        ):
            raise ValueError("gate hysteresis leaves no usable transition span")
        self.minimum_zone_frames = minimum_zone_frames
        self.maximum_gap_frames = maximum_gap_frames
        self.event_cooldown_frames = event_cooldown_frames
        self.minimum_direction_consistency = minimum_direction_consistency
        self.gate_hysteresis = gate_hysteresis
        self.minimum_journey_frames = minimum_journey_frames
        self._tracks: dict[int, _GateTrackState] = {}
        self._last_events: dict[int, tuple[int, str]] = {}
        self._decisions: list[TrajectoryDecision] = []
        self._confirmed_events = 0
        self._gap_resets = 0
        self._reversed_journeys = 0
        self._inconsistent_rejections = 0
        self._insufficient_observation_rejections = 0
        self._cooldown_suppressed = 0
        self._prediction_observations_ignored = 0

    @property
    def summary(self) -> dict[str, int]:
        return {
            "confirmed_events": self._confirmed_events,
            "gap_resets": self._gap_resets,
            "reversed_journeys": self._reversed_journeys,
            "inconsistent_direction_rejections": self._inconsistent_rejections,
            "insufficient_observation_rejections": (
                self._insufficient_observation_rejections
            ),
            "event_cooldown_suppressed": self._cooldown_suppressed,
            "prediction_observations_ignored": (
                self._prediction_observations_ignored
            ),
        }

    def display_zone(
        self, track: TrackedDetection, frame_shape: tuple[int, ...]
    ) -> Zone:
        return self.layout.classify(track.center, frame_shape)

    def observe(
        self,
        *,
        track: TrackedDetection,
        frame_number: int,
        frame_shape: tuple[int, ...],
    ) -> CrossingEvent | None:
        if bool(track.metadata.get("prediction_only")):
            self._prediction_observations_ignored += 1
            self._record(
                frame_number,
                track.tracking_id,
                "ignored",
                "prediction_only",
            )
            return None

        track_id = track.tracking_id
        progress = self.geometry.progress(track.center, frame_shape)
        confidence = max(0.0, min(1.0, track.confidence))
        state = self._tracks.get(track_id)
        if state is None or frame_number - state.last_seen > self.maximum_gap_frames:
            if state is not None:
                self._gap_resets += 1
                self._record(
                    frame_number,
                    track_id,
                    "journey_reset",
                    "tracking_gap",
                    progress,
                    origin=state.origin,
                )
            state = _GateTrackState(last_seen=frame_number)
            self._tracks[track_id] = state
        state.last_seen = frame_number
        endpoint = self._endpoint(progress)

        if state.origin is Zone.UNKNOWN:
            self._observe_origin_candidate(
                state,
                endpoint,
                frame_number,
                progress,
                confidence,
                track_id,
            )
            return None

        state.samples.append((frame_number, progress, confidence))
        direction = "IN" if state.origin is Zone.OUTSIDE else "OUT"
        destination = Zone.INSIDE if direction == "IN" else Zone.OUTSIDE
        expected_sign = 1 if direction == "IN" else -1
        first_gate = (
            self.geometry.outside_gate + self.gate_hysteresis
            if direction == "IN"
            else self.geometry.inside_gate - self.gate_hysteresis
        )
        crossed_first = (
            progress >= first_gate if direction == "IN" else progress <= first_gate
        )
        if crossed_first and not state.first_gate_crossed:
            state.first_gate_crossed = True
            self._record(
                frame_number,
                track_id,
                "gate_crossed",
                "first_gate",
                progress,
                direction,
                state.origin,
            )

        if endpoint is state.origin:
            state.origin_return_frames += 1
            state.destination_frames = 0
            if (
                state.first_gate_crossed
                and state.origin_return_frames >= self.minimum_zone_frames
            ):
                self._reversed_journeys += 1
                consistency = self._direction_consistency(
                    state.samples, expected_sign
                )
                self._record(
                    frame_number,
                    track_id,
                    "journey_rejected",
                    "returned_to_origin",
                    progress,
                    direction,
                    state.origin,
                    consistency,
                )
                self._begin_journey(
                    state, state.origin, frame_number, progress, confidence
                )
            elif not state.first_gate_crossed:
                self._begin_journey(
                    state, state.origin, frame_number, progress, confidence
                )
            return None
        state.origin_return_frames = 0

        state.destination_frames = (
            state.destination_frames + 1 if endpoint is destination else 0
        )
        if state.destination_frames < self.minimum_zone_frames:
            return None

        consistency = self._direction_consistency(state.samples, expected_sign)
        journey_frames = frame_number - state.origin_frame + 1
        rejection_reason: str | None = None
        if not state.first_gate_crossed:
            rejection_reason = "gate_sequence_missing"
        elif journey_frames < self.minimum_journey_frames:
            rejection_reason = "insufficient_real_observations"
        elif consistency < self.minimum_direction_consistency:
            rejection_reason = "inconsistent_direction"

        if rejection_reason is not None:
            if rejection_reason == "inconsistent_direction":
                self._inconsistent_rejections += 1
            else:
                self._insufficient_observation_rejections += 1
            self._record(
                frame_number,
                track_id,
                "journey_rejected",
                rejection_reason,
                progress,
                direction,
                state.origin,
                consistency,
            )
            self._begin_journey(
                state, destination, frame_number, progress, confidence
            )
            return None

        previous_event = self._last_events.get(track_id)
        if (
            previous_event is not None
            and frame_number - previous_event[0] <= self.event_cooldown_frames
        ):
            self._cooldown_suppressed += 1
            self._record(
                frame_number,
                track_id,
                "journey_rejected",
                "event_cooldown",
                progress,
                direction,
                state.origin,
                consistency,
            )
            self._begin_journey(
                state, destination, frame_number, progress, confidence
            )
            return None

        origin = state.origin
        event = CrossingEvent(
            track_id=track_id,
            direction=direction,
            frame_number=frame_number,
            confidence=sum(sample[2] for sample in state.samples) / len(state.samples),
            dwell_frames=journey_frames,
            path=(origin, Zone.DOOR, destination),
        )
        self._confirmed_events += 1
        self._last_events[track_id] = (frame_number, direction)
        self._record(
            frame_number,
            track_id,
            "event_confirmed",
            "two_gates_monotonic",
            progress,
            direction,
            origin,
            consistency,
        )
        self._begin_journey(state, destination, frame_number, progress, confidence)
        return event

    def expire(self, frame_number: int) -> None:
        for track_id, state in tuple(self._tracks.items()):
            if frame_number - state.last_seen <= self.maximum_gap_frames:
                continue
            self._gap_resets += 1
            self._record(
                frame_number,
                track_id,
                "journey_reset",
                "track_expired",
                origin=state.origin,
            )
            del self._tracks[track_id]
        retention = max(self.maximum_gap_frames, self.event_cooldown_frames) * 2 + 1
        self._last_events = {
            track_id: event
            for track_id, event in self._last_events.items()
            if frame_number - event[0] <= retention
        }

    def drain_decisions(self) -> tuple[TrajectoryDecision, ...]:
        decisions = tuple(self._decisions)
        self._decisions.clear()
        return decisions

    def reset(self) -> None:
        self._tracks.clear()
        self._last_events.clear()
        self._decisions.clear()
        self._confirmed_events = 0
        self._gap_resets = 0
        self._reversed_journeys = 0
        self._inconsistent_rejections = 0
        self._insufficient_observation_rejections = 0
        self._cooldown_suppressed = 0
        self._prediction_observations_ignored = 0

    def _endpoint(self, progress: float) -> Zone:
        if progress <= self.geometry.outside_gate - self.gate_hysteresis:
            return Zone.OUTSIDE
        if progress >= self.geometry.inside_gate + self.gate_hysteresis:
            return Zone.INSIDE
        return Zone.DOOR

    def _observe_origin_candidate(
        self,
        state: _GateTrackState,
        endpoint: Zone,
        frame_number: int,
        progress: float,
        confidence: float,
        track_id: int,
    ) -> None:
        if endpoint not in {Zone.OUTSIDE, Zone.INSIDE}:
            state.candidate_origin = Zone.UNKNOWN
            state.candidate_frames = 0
            return
        if endpoint is state.candidate_origin:
            state.candidate_frames += 1
        else:
            state.candidate_origin = endpoint
            state.candidate_frames = 1
        if state.candidate_frames < self.minimum_zone_frames:
            return
        self._begin_journey(state, endpoint, frame_number, progress, confidence)
        self._record(
            frame_number,
            track_id,
            "origin_confirmed",
            "real_endpoint_dwell",
            progress,
            origin=endpoint,
        )

    @staticmethod
    def _begin_journey(
        state: _GateTrackState,
        origin: Zone,
        frame_number: int,
        progress: float,
        confidence: float,
    ) -> None:
        state.candidate_origin = origin
        state.candidate_frames = 1
        state.origin = origin
        state.origin_frame = frame_number
        state.first_gate_crossed = False
        state.destination_frames = 0
        state.origin_return_frames = 0
        state.samples = [(frame_number, progress, confidence)]

    def _direction_consistency(
        self,
        samples: list[tuple[int, float, float]],
        expected_sign: int,
    ) -> float:
        threshold = max(1e-4, self.gate_hysteresis / 4)
        deltas = [
            current[1] - previous[1]
            for previous, current in zip(samples, samples[1:], strict=False)
            if abs(current[1] - previous[1]) >= threshold
        ]
        if not deltas:
            return 0.0
        matching = sum(1 for delta in deltas if delta * expected_sign > 0)
        return matching / len(deltas)

    def _record(
        self,
        frame_number: int,
        track_id: int,
        action: str,
        reason: str,
        progress: float | None = None,
        direction: str | None = None,
        origin: Zone | None = None,
        direction_consistency: float | None = None,
    ) -> None:
        self._decisions.append(
            TrajectoryDecision(
                frame_number=frame_number,
                track_id=track_id,
                action=action,
                reason=reason,
                progress=progress,
                direction=direction,
                origin=origin,
                direction_consistency=direction_consistency,
            )
        )
