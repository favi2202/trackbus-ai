"""One explicit, version-isolated Ultralytics ByteTrack adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection, TrackedDetection
from trackbus.detector import PersonDetector

# Backward-compatible public name used by v0.1.x callers and diagnostics.
TrackedPerson = TrackedDetection


class TrackerAdapterError(RuntimeError):
    """Raised when the installed Ultralytics tracker contract is incompatible."""


def validate_confidence_contract(
    detector_floor: float,
    tracker_config: Mapping[str, object],
) -> dict[str, object]:
    """Validate and describe the detector-to-ByteTrack confidence handoff.

    ByteTrack uses predictions below ``track_high_thresh`` only to associate an
    existing track. A new track must require at least ``new_track_thresh``. The
    detector may deliberately run above ``track_low_thresh`` for a legacy camera,
    but the returned warning makes the unavailable low band explicit.
    """

    try:
        low = float(tracker_config["track_low_thresh"])
        high = float(tracker_config["track_high_thresh"])
        new = float(tracker_config["new_track_thresh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackerAdapterError(
            "ByteTrack confidence settings must include finite low, high, and "
            "new-track thresholds."
        ) from exc
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in (low, high, new)
    ):
        raise TrackerAdapterError(
            "ByteTrack low, high, and new-track thresholds must be between 0 and 1."
        )
    if low > high:
        raise TrackerAdapterError(
            "ByteTrack track_low_thresh cannot exceed track_high_thresh."
        )
    if new < high:
        raise TrackerAdapterError(
            "ByteTrack new_track_thresh cannot be below track_high_thresh; weak "
            "detections must not create new tracks."
        )
    if not math.isfinite(detector_floor) or not 0.0 < detector_floor <= 1.0:
        raise TrackerAdapterError(
            "Detector floor must be greater than 0 and at most 1."
        )

    warning = None
    if detector_floor > low:
        warning = (
            f"detector floor {detector_floor:.3f} is above ByteTrack's low "
            f"threshold {low:.3f}; predictions below the detector floor remain "
            "unavailable for track recovery"
        )
    return {
        "detector_floor": detector_floor,
        "track_low_thresh": low,
        "track_high_thresh": high,
        "new_track_thresh": new,
        "full_low_band_available": detector_floor <= low,
        "weak_detections_can_start_tracks": False,
        "warning": warning,
    }


@dataclass
class _ContinuityState:
    """Short-lived, video-local geometry for one stable anonymous track."""

    stable_id: int
    raw_id: int
    last_seen_frame: int
    bounding_box: tuple[float, float, float, float]
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    observations: int = 1


@dataclass(frozen=True)
class _ContinuityCandidate:
    stable_id: int
    raw_id: int
    score: float
    gap_frames: int
    iou: float
    centroid_distance: float


def _trusted_tracker_config(tracker_config: str) -> str:
    """Allow Ultralytics' packaged ByteTrack YAML or an existing local file.

    ``check_yaml`` can download remote inputs. Resolve and validate the value
    before handing it to that version-sensitive helper so a configuration typo
    cannot turn into an implicit network fetch.
    """

    value = str(tracker_config).strip()
    if not value:
        raise TrackerAdapterError("Tracker config cannot be empty.")
    if "://" in value or value.lower().startswith(("http:", "https:", "ul:")):
        raise TrackerAdapterError("Remote tracker configurations are not allowed.")
    if value == "bytetrack.yaml":
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise TrackerAdapterError(
            f"Tracker config must be 'bytetrack.yaml' or an existing local file: {path}"
        )
    return str(path.resolve())


class ByteTrackAdapter:
    """Submit one fused source-coordinate set to one BYTETracker per frame.

    This adapter is tested with Ultralytics 8.4.95. All internal imports, input
    conversion, output-column assumptions, and empty-result handling live here.
    There is deliberately no fallback to ``YOLO.track``.
    """

    COMPATIBLE_ULTRALYTICS_VERSION = "8.4.95"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        tracker_config: str,
        *,
        device_label: str,
        tracker_overrides: Mapping[str, float | int | bool] | None = None,
    ) -> None:
        trusted_tracker_config = _trusted_tracker_config(tracker_config)
        self.tracker_config = trusted_tracker_config
        self.device_label = device_label
        self.update_count = 0
        try:
            import ultralytics
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            loaded = YAML.load(check_yaml(trusted_tracker_config))
            args = IterableSimpleNamespace(**loaded)
            for name, value in (tracker_overrides or {}).items():
                if name not in {
                    "track_buffer",
                    "track_high_thresh",
                    "track_low_thresh",
                    "new_track_thresh",
                    "match_thresh",
                    "fuse_score",
                }:
                    raise TrackerAdapterError(
                        f"Unsupported ByteTrack override '{name}'."
                    )
                setattr(args, name, value)
            if getattr(args, "tracker_type", None) != "bytetrack":
                raise TrackerAdapterError(
                    f"Tracker config '{trusted_tracker_config}' must set "
                    "tracker_type: bytetrack."
                )
            for attribute in (
                "track_buffer",
                "track_high_thresh",
                "track_low_thresh",
                "new_track_thresh",
                "match_thresh",
                "fuse_score",
            ):
                if not hasattr(args, attribute):
                    raise TrackerAdapterError(
                        "Tracker config "
                        f"'{trusted_tracker_config}' is missing '{attribute}'."
                    )
            self.ultralytics_version = ultralytics.__version__
            self.effective_config = {
                attribute: getattr(args, attribute)
                for attribute in (
                    "track_buffer",
                    "track_high_thresh",
                    "track_low_thresh",
                    "new_track_thresh",
                    "match_thresh",
                    "fuse_score",
                )
            }
            validate_confidence_contract(
                float(self.effective_config["track_low_thresh"]),
                self.effective_config,
            )
            self._boxes_type = Boxes
            self._tracker = BYTETracker(args=args)
        except TrackerAdapterError:
            raise
        except Exception as exc:
            raise TrackerAdapterError(
                "Could not initialize the explicit Ultralytics ByteTrack adapter "
                f"from '{trusted_tracker_config}'. Tested with Ultralytics "
                f"{self.COMPATIBLE_ULTRALYTICS_VERSION}: {exc}"
            ) from exc

    def confidence_contract(self, detector_floor: float) -> dict[str, object]:
        """Return the validated confidence handoff for run summaries."""

        return validate_confidence_contract(detector_floor, self.effective_config)

    def update(
        self,
        detections: list[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedDetection]:
        """Advance ByteTrack exactly once, including on empty detection frames."""

        if any(detection.class_id != self.PERSON_CLASS_ID for detection in detections):
            raise TrackerAdapterError(
                "ByteTrackAdapter accepts person-class detections only."
            )
        height, width = frame.shape[:2]
        rows = np.asarray(
            [
                [
                    *detection.bounding_box,
                    detection.confidence,
                    detection.class_id,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)
        try:
            boxes = self._boxes_type(rows, orig_shape=(height, width)).cpu().numpy()
            tracks = np.asarray(self._tracker.update(boxes, frame))
        except Exception as exc:
            raise TrackerAdapterError(
                "Ultralytics BYTETracker.update failed. Its internal API may be "
                f"incompatible (installed version {self.ultralytics_version}): {exc}"
            ) from exc
        self.update_count += 1
        if tracks.size == 0:
            return []
        if tracks.ndim != 2 or tracks.shape[1] != 8:
            raise TrackerAdapterError(
                "Ultralytics ByteTrack returned an unexpected array; expected Nx8 "
                f"and received shape {tracks.shape}."
            )

        tracked: list[TrackedDetection] = []
        for row in tracks:
            source_index = int(row[7])
            if not 0 <= source_index < len(detections):
                raise TrackerAdapterError(
                    "ByteTrack returned an invalid source-detection index "
                    f"{source_index} for {len(detections)} inputs."
                )
            source = detections[source_index]
            metadata = dict(source.metadata)
            metadata["tracker_input_index"] = source_index
            tracked.append(
                TrackedDetection(
                    tracking_id=int(row[4]),
                    bounding_box=tuple(float(value) for value in row[:4]),
                    confidence=float(row[5]),
                    class_id=int(row[6]),
                    source_views=source.source_views,
                    metadata=metadata,
                )
            )
        return tracked

    def reset(self) -> None:
        """Clear all tracker state for a new source video."""

        try:
            self._tracker.reset()
        except Exception as exc:
            raise TrackerAdapterError(f"Could not reset ByteTrack: {exc}") from exc
        self.update_count = 0


class TrackContinuityAdapter:
    """Conservatively reconnect short ByteTrack ID fragments.

    The adapter never fabricates observations for frames where the wrapped tracker
    returns nothing. It only remaps a newly appearing raw ID to a recently missing
    video-local stable ID when geometry, scale, and movement agree. State is bounded
    by ``max_gap_frames`` and is cleared between videos.
    """

    def __init__(
        self,
        tracker: ByteTrackAdapter,
        *,
        max_gap_frames: int = 3,
        max_centroid_distance: float = 0.08,
        minimum_iou: float = 0.05,
        maximum_size_ratio: float = 1.8,
        minimum_direction_cosine: float = -0.25,
        minimum_match_score: float = 0.55,
    ) -> None:
        self._wrapped = tracker
        self.max_gap_frames = max_gap_frames
        self.max_centroid_distance = max_centroid_distance
        self.minimum_iou = minimum_iou
        self.maximum_size_ratio = maximum_size_ratio
        self.minimum_direction_cosine = minimum_direction_cosine
        self.minimum_match_score = minimum_match_score
        self.device_label = tracker.device_label
        self.tracker_config = tracker.tracker_config
        self.ultralytics_version = tracker.ultralytics_version
        self.update_count = 0
        self._frame_number = -1
        self._states: dict[int, _ContinuityState] = {}
        self._raw_to_stable: dict[int, int] = {}
        self._next_stable_id = 0
        self._stitch_attempts = 0
        self._candidate_pairs_evaluated = 0
        self._stitched_track_fragments = 0
        self._new_stable_tracks = 0
        self._expired_memories = 0
        self._maximum_stitched_gap_frames = 0

    @property
    def effective_config(self) -> dict[str, object]:
        """Expose ByteTrack and continuity settings in processing summaries."""

        return {
            **self._wrapped.effective_config,
            "trackbus_continuity": {
                "enabled": True,
                "max_gap_frames": self.max_gap_frames,
                "max_centroid_distance": self.max_centroid_distance,
                "minimum_iou": self.minimum_iou,
                "maximum_size_ratio": self.maximum_size_ratio,
                "minimum_direction_cosine": self.minimum_direction_cosine,
                "minimum_match_score": self.minimum_match_score,
            },
        }

    @property
    def continuity_summary(self) -> dict[str, object]:
        """Return honest run diagnostics; these are not accuracy metrics."""

        return {
            "enabled": True,
            "frames_processed": self.update_count,
            "stitch_attempts": self._stitch_attempts,
            "candidate_pairs_evaluated": self._candidate_pairs_evaluated,
            "stitched_track_fragments": self._stitched_track_fragments,
            "new_stable_tracks": self._new_stable_tracks,
            "expired_memories": self._expired_memories,
            "maximum_stitched_gap_frames": self._maximum_stitched_gap_frames,
            "active_track_memories": len(self._states),
        }

    def update(
        self,
        detections: list[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedDetection]:
        """Advance ByteTrack once and remap only defensible restarted IDs."""

        tracked = self._wrapped.update(detections, frame)
        self.update_count += 1
        self._frame_number += 1
        self._expire_old_states()
        if not tracked:
            return []

        current_raw_ids = {person.tracking_id for person in tracked}
        assigned_stable_ids: set[int] = set()
        resolved: dict[int, tuple[int, _ContinuityCandidate | None]] = {}

        for person in tracked:
            raw_id = person.tracking_id
            stable_id = self._raw_to_stable.get(raw_id)
            if stable_id is None or stable_id not in self._states:
                continue
            resolved[raw_id] = (stable_id, None)
            assigned_stable_ids.add(stable_id)

        unmatched = [person for person in tracked if person.tracking_id not in resolved]
        candidates: list[_ContinuityCandidate] = []
        for person in unmatched:
            raw_id = person.tracking_id
            possible_states = [
                state
                for state in self._states.values()
                if state.stable_id not in assigned_stable_ids
                and state.raw_id not in current_raw_ids
            ]
            if possible_states:
                self._stitch_attempts += 1
            for state in possible_states:
                self._candidate_pairs_evaluated += 1
                candidate = self._candidate(state, person, frame.shape[:2])
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(key=lambda item: (-item.score, item.stable_id, item.raw_id))
        assigned_raw_ids = set(resolved)
        for candidate in candidates:
            if candidate.raw_id in assigned_raw_ids:
                continue
            if candidate.stable_id in assigned_stable_ids:
                continue
            resolved[candidate.raw_id] = (candidate.stable_id, candidate)
            assigned_raw_ids.add(candidate.raw_id)
            assigned_stable_ids.add(candidate.stable_id)

        output: list[TrackedDetection] = []
        for person in tracked:
            raw_id = person.tracking_id
            assignment = resolved.get(raw_id)
            if assignment is None:
                stable_id = self._allocate_stable_id(raw_id)
                candidate = None
                self._new_stable_tracks += 1
            else:
                stable_id, candidate = assignment

            previous_state = self._states.get(stable_id)
            metadata = dict(person.metadata)
            metadata["raw_tracking_id"] = raw_id
            if candidate is not None and previous_state is not None:
                metadata.update(
                    {
                        "continuity_stitched": True,
                        "continuity_previous_raw_tracking_id": (previous_state.raw_id),
                        "continuity_gap_frames": candidate.gap_frames,
                        "continuity_match_score": round(candidate.score, 4),
                        "continuity_iou": round(candidate.iou, 4),
                        "continuity_centroid_distance": round(
                            candidate.centroid_distance, 4
                        ),
                    }
                )
                flags = _quality_flags(metadata.get("quality_flags"))
                metadata["quality_flags"] = tuple(
                    dict.fromkeys((*flags, "track_continuity_stitch"))
                )
                self._stitched_track_fragments += 1
                self._maximum_stitched_gap_frames = max(
                    self._maximum_stitched_gap_frames,
                    candidate.gap_frames,
                )
                self._remove_raw_mappings(stable_id)
            else:
                gap = (
                    self._frame_number - previous_state.last_seen_frame - 1
                    if previous_state is not None
                    else 0
                )
                metadata["continuity_stitched"] = False
                metadata["continuity_gap_frames"] = max(0, gap)

            self._raw_to_stable[raw_id] = stable_id
            self._update_state(stable_id, raw_id, person)
            output.append(replace(person, tracking_id=stable_id, metadata=metadata))
        return output

    def reset(self) -> None:
        """Clear both wrapped ByteTrack and all video-local continuity state."""

        self._wrapped.reset()
        self.update_count = 0
        self._frame_number = -1
        self._states.clear()
        self._raw_to_stable.clear()
        self._next_stable_id = 0
        self._stitch_attempts = 0
        self._candidate_pairs_evaluated = 0
        self._stitched_track_fragments = 0
        self._new_stable_tracks = 0
        self._expired_memories = 0
        self._maximum_stitched_gap_frames = 0

    def _candidate(
        self,
        state: _ContinuityState,
        person: TrackedDetection,
        frame_shape: tuple[int, int],
    ) -> _ContinuityCandidate | None:
        gap_frames = self._frame_number - state.last_seen_frame - 1
        if not 0 <= gap_frames <= self.max_gap_frames:
            return None

        elapsed_frames = max(1, self._frame_number - state.last_seen_frame)
        dx = state.velocity[0] * elapsed_frames
        dy = state.velocity[1] * elapsed_frames
        predicted_box = _shift_box(state.bounding_box, dx, dy)
        predicted_center = (state.center[0] + dx, state.center[1] + dy)
        frame_height, frame_width = frame_shape
        diagonal = max(1.0, math.hypot(frame_width, frame_height))
        distance = math.dist(predicted_center, person.center) / diagonal
        if distance > self.max_centroid_distance:
            return None

        size_ratio = _box_size_ratio(state.bounding_box, person.bounding_box)
        if size_ratio > self.maximum_size_ratio:
            return None

        displacement = (
            person.center[0] - state.center[0],
            person.center[1] - state.center[1],
        )
        direction_cosine = _direction_cosine(state.velocity, displacement)
        if (
            direction_cosine is not None
            and direction_cosine < self.minimum_direction_cosine
        ):
            return None

        iou = _box_iou(predicted_box, person.bounding_box)
        if iou < self.minimum_iou and distance > self.max_centroid_distance * 0.5:
            return None

        center_score = 1.0 - distance / self.max_centroid_distance
        size_score = _ratio_score(size_ratio, self.maximum_size_ratio)
        direction_score = (
            (direction_cosine + 1.0) / 2.0 if direction_cosine is not None else 0.5
        )
        score = (
            0.45 * center_score
            + 0.30 * iou
            + 0.15 * size_score
            + 0.10 * direction_score
        )
        if score < self.minimum_match_score:
            return None
        return _ContinuityCandidate(
            stable_id=state.stable_id,
            raw_id=person.tracking_id,
            score=score,
            gap_frames=gap_frames,
            iou=iou,
            centroid_distance=distance,
        )

    def _allocate_stable_id(self, raw_id: int) -> int:
        if raw_id not in self._states:
            stable_id = raw_id
        else:
            stable_id = max(self._next_stable_id, max(self._states, default=-1) + 1)
            while stable_id in self._states:
                stable_id += 1
        self._next_stable_id = max(self._next_stable_id, stable_id + 1)
        return stable_id

    def _update_state(
        self,
        stable_id: int,
        raw_id: int,
        person: TrackedDetection,
    ) -> None:
        previous = self._states.get(stable_id)
        velocity = (0.0, 0.0)
        observations = 1
        if previous is not None:
            elapsed = max(1, self._frame_number - previous.last_seen_frame)
            velocity = (
                (person.center[0] - previous.center[0]) / elapsed,
                (person.center[1] - previous.center[1]) / elapsed,
            )
            observations = previous.observations + 1
        self._states[stable_id] = _ContinuityState(
            stable_id=stable_id,
            raw_id=raw_id,
            last_seen_frame=self._frame_number,
            bounding_box=person.bounding_box,
            center=person.center,
            velocity=velocity,
            observations=observations,
        )

    def _expire_old_states(self) -> None:
        expired = [
            stable_id
            for stable_id, state in self._states.items()
            if self._frame_number - state.last_seen_frame - 1 > self.max_gap_frames
        ]
        for stable_id in expired:
            self._states.pop(stable_id, None)
            self._remove_raw_mappings(stable_id)
            self._expired_memories += 1

    def _remove_raw_mappings(self, stable_id: int) -> None:
        for raw_id, mapped_stable_id in tuple(self._raw_to_stable.items()):
            if mapped_stable_id == stable_id:
                self._raw_to_stable.pop(raw_id, None)


def _shift_box(
    box: tuple[float, float, float, float], dx: float, dy: float
) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    return left + dx, top + dy, right + dx, bottom + dy


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_size_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_area = max(1e-6, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1e-6, (second[2] - second[0]) * (second[3] - second[1]))
    return max(first_area / second_area, second_area / first_area)


def _ratio_score(ratio: float, maximum: float) -> float:
    if maximum <= 1.0:
        return 1.0 if ratio <= 1.0 else 0.0
    return max(0.0, 1.0 - (ratio - 1.0) / (maximum - 1.0))


def _direction_cosine(
    velocity: tuple[float, float], displacement: tuple[float, float]
) -> float | None:
    velocity_length = math.hypot(*velocity)
    displacement_length = math.hypot(*displacement)
    if velocity_length <= 1e-6 or displacement_length <= 1e-6:
        return None
    return (velocity[0] * displacement[0] + velocity[1] * displacement[1]) / (
        velocity_length * displacement_length
    )


def _quality_flags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


class ByteTrackPersonTracker:
    """Explicitly requested v0.1.x combined adapter kept for API compatibility."""

    def __init__(
        self,
        detector: PersonDetector,
        *,
        tracker_config: str,
        device: str | int,
        device_label: str,
    ) -> None:
        self.detector = detector
        self.tracker_config = tracker_config
        self.device = device
        self.device_label = device_label

    def update(self, frame: NDArray[np.uint8]) -> list[TrackedPerson]:
        """Run the deprecated combined path only when this class is instantiated."""

        result = self.detector.infer_with_tracking(
            frame, tracker=self.tracker_config, device=self.device
        )
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().tolist()
        identifiers = boxes.id.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        return [
            TrackedPerson(
                tracking_id=int(identifier),
                bounding_box=tuple(float(value) for value in coordinate),
                confidence=float(confidence),
                class_id=int(class_id),
            )
            for coordinate, identifier, confidence, class_id in zip(
                coordinates, identifiers, confidences, classes, strict=True
            )
        ]


def tracker_config_exists(config: str) -> bool:
    """Return true for local tracker profiles (built-ins resolve in Ultralytics)."""

    return Path(config).is_file()
