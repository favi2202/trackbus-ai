"""Inference-view cropping and source-coordinate translation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trackbus.config import InferenceViewConfig, NormalizedRoi
from trackbus.detection import BoundingBox, Detection
from trackbus.interfaces import DetectorBackend


@dataclass(frozen=True)
class PixelInferenceView:
    """One named rectangular view resolved for a source-frame size."""

    name: str
    normalized_bounds: NormalizedRoi
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_config(
        cls, config: InferenceViewConfig, frame_width: int, frame_height: int
    ) -> PixelInferenceView:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        left, top, right, bottom = config.bounds
        resolved = cls(
            name=config.name,
            normalized_bounds=config.bounds,
            left=math.floor(left * frame_width),
            top=math.floor(top * frame_height),
            right=math.ceil(right * frame_width),
            bottom=math.ceil(bottom * frame_height),
        )
        if resolved.width <= 0 or resolved.height <= 0:
            raise ValueError(f"inference view '{config.name}' is empty")
        return resolved

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def is_full_frame(self) -> bool:
        return self.normalized_bounds == (0.0, 0.0, 1.0, 1.0)

    def crop(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return (
            frame
            if self.is_full_frame
            else frame[self.top : self.bottom, self.left : self.right]
        )

    def translate(
        self, detection: Detection, frame_width: int, frame_height: int
    ) -> Detection | None:
        left, top, right, bottom = detection.bounding_box
        source_box = clip_box_to_frame(
            (
                left + self.left,
                top + self.top,
                right + self.left,
                bottom + self.top,
            ),
            frame_width,
            frame_height,
        )
        if not box_has_positive_area(source_box):
            return None
        return detection.with_box(
            source_box,
            coordinate_space="source",
            view_bounds=self.normalized_bounds,
        )


class MultiViewInference:
    """Collect translated predictions from every enabled inference view."""

    def __init__(
        self,
        detector: DetectorBackend,
        views: tuple[PixelInferenceView, ...],
    ) -> None:
        if not views:
            raise ValueError("at least one enabled inference view is required")
        self.detector = detector
        self.views = views

    @classmethod
    def from_configs(
        cls,
        detector: DetectorBackend,
        views: tuple[InferenceViewConfig, ...],
        frame_width: int,
        frame_height: int,
    ) -> MultiViewInference:
        enabled = tuple(view for view in views if view.enabled)
        return cls(
            detector,
            tuple(
                PixelInferenceView.from_config(view, frame_width, frame_height)
                for view in enabled
            ),
        )

    def collect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        frame_height, frame_width = frame.shape[:2]
        detections: list[Detection] = []
        for view in self.views:
            local = self.detector.detect(view.crop(frame), source_view=view.name)
            for detection in local:
                translated = view.translate(detection, frame_width, frame_height)
                if translated is not None:
                    detections.append(translated)
        return detections


def clip_box_to_frame(
    box: BoundingBox, frame_width: int, frame_height: int
) -> BoundingBox:
    """Clip an xyxy box to the source image without changing coordinates."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    left, top, right, bottom = box
    maximum_x = float(frame_width - 1)
    maximum_y = float(frame_height - 1)
    return (
        max(0.0, min(maximum_x, left)),
        max(0.0, min(maximum_y, top)),
        max(0.0, min(maximum_x, right)),
        max(0.0, min(maximum_y, bottom)),
    )


def box_has_positive_area(box: BoundingBox) -> bool:
    left, top, right, bottom = box
    return right > left and bottom > top
