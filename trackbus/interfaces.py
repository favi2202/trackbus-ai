"""Small protocols separating prediction, fusion, and tracking."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection, TrackedDetection


class DetectorBackend(Protocol):
    """Return untracked detections for one inference image."""

    def detect(
        self, image: NDArray[np.uint8], *, source_view: str
    ) -> list[Detection]: ...


class DetectionFusion(Protocol):
    """Merge source-coordinate predictions before tracking."""

    def fuse(self, detections: list[Detection]) -> list[Detection]: ...


class TrackerBackend(Protocol):
    """Update one tracker once for an original source frame."""

    device_label: str

    def update(
        self,
        detections: list[Detection],
        frame: NDArray[np.uint8],
    ) -> list[TrackedDetection]: ...
