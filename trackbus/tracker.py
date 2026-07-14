"""ByteTrack adapter that exposes model-independent tracked people."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trackbus.detector import PersonDetector


@dataclass(frozen=True)
class TrackedPerson:
    """One person detection associated with a stable tracker ID."""

    tracking_id: int
    bounding_box: tuple[float, float, float, float]
    confidence: float

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-center point used for doorway zone membership."""

        left, _top, right, bottom = self.bounding_box
        return ((left + right) / 2.0, bottom)


class ByteTrackPersonTracker:
    """Convert Ultralytics ByteTrack results into :class:`TrackedPerson` values."""

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
        result = self.detector.infer_with_tracking(
            frame, tracker=self.tracker_config, device=self.device
        )
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.detach().cpu().tolist()
        identifiers = boxes.id.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        return [
            TrackedPerson(
                tracking_id=int(identifier),
                bounding_box=tuple(float(value) for value in coordinate),
                confidence=float(confidence),
            )
            for coordinate, identifier, confidence in zip(
                coordinates, identifiers, confidences, strict=True
            )
        ]
