"""One explicit, version-isolated Ultralytics ByteTrack adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection, TrackedDetection
from trackbus.detector import PersonDetector

# Backward-compatible public name used by v0.1.x callers and diagnostics.
TrackedPerson = TrackedDetection


class TrackerAdapterError(RuntimeError):
    """Raised when the installed Ultralytics tracker contract is incompatible."""


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
