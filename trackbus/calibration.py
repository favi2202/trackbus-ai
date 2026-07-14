"""Camera calibration, ROI translation, doorway lanes, and exclusions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.config import CameraConfig, NormalizedPolygon, NormalizedRoi
from trackbus.tracker import TrackedPerson


class CalibrationError(ValueError):
    """Raised when pixel calibration is inconsistent with the camera geometry."""


class DoorwayLane(StrEnum):
    LEFT = "left_lane"
    CENTER = "center_lane"
    RIGHT = "right_lane"


@dataclass(frozen=True)
class PixelRoi:
    left: int
    top: int
    right: int
    bottom: int
    enabled: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def crop(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not self.enabled:
            return frame
        return frame[self.top : self.bottom, self.left : self.right]

    def translate(self, person: TrackedPerson) -> TrackedPerson:
        if not self.enabled:
            return person
        left, top, right, bottom = person.bounding_box
        return TrackedPerson(
            tracking_id=person.tracking_id,
            bounding_box=(
                left + self.left,
                top + self.top,
                right + self.left,
                bottom + self.top,
            ),
            confidence=person.confidence,
        )


class FrameCalibration:
    """Pixel-space calibration derived once for a decoded video resolution."""

    def __init__(
        self, config: CameraConfig, frame_width: int, frame_height: int
    ) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise CalibrationError("Frame width and height must be positive.")
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi = _pixel_roi(config.detection_roi, frame_width, frame_height)
        lane_configs = {
            DoorwayLane.LEFT: config.left_lane,
            DoorwayLane.CENTER: config.center_lane,
            DoorwayLane.RIGHT: config.right_lane,
        }
        self.lanes = {
            lane: _scale_polygon(polygon, frame_width, frame_height)
            for lane, polygon in lane_configs.items()
            if polygon is not None
        }
        self.exclusions = tuple(
            _scale_polygon(polygon, frame_width, frame_height)
            for polygon in config.exclusion_polygons
        )
        self._lane_masks = {
            lane: _polygon_mask(polygon, frame_width, frame_height)
            for lane, polygon in self.lanes.items()
        }
        self._validate_exclusions_do_not_cover_lanes()

    @property
    def doorway_enabled(self) -> bool:
        return bool(self.lanes)

    @property
    def doorway_width(self) -> int:
        if not self.lanes:
            return self.frame_width
        left = min(int(polygon[:, 0].min()) for polygon in self.lanes.values())
        right = max(int(polygon[:, 0].max()) for polygon in self.lanes.values())
        return max(1, right - left + 1)

    def inference_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return self.roi.crop(frame)

    def to_source(self, people: list[TrackedPerson]) -> list[TrackedPerson]:
        return [self._clip_person(self.roi.translate(person)) for person in people]

    def is_excluded(self, person: TrackedPerson) -> bool:
        """Use the established bottom-center anchor for camera exclusions."""

        return any(
            cv2.pointPolygonTest(polygon, person.anchor, False) >= 0
            for polygon in self.exclusions
        )

    def lane_for(self, person: TrackedPerson) -> DoorwayLane | None:
        point = person.center if self.config.lane_anchor == "center" else person.anchor
        candidates = [
            (cv2.pointPolygonTest(polygon, point, True), lane)
            for lane, polygon in self.lanes.items()
        ]
        inside = [candidate for candidate in candidates if candidate[0] >= 0]
        if not inside:
            return None
        return max(inside, key=lambda candidate: candidate[0])[1]

    def box_lanes(
        self, person: TrackedPerson, minimum_overlap: float
    ) -> tuple[DoorwayLane, ...]:
        left, top, right, bottom = _clipped_box(
            person.bounding_box, self.frame_width, self.frame_height
        )
        box_area = max(1, (right - left) * (bottom - top))
        return tuple(
            lane
            for lane, mask in self._lane_masks.items()
            if np.count_nonzero(mask[top:bottom, left:right]) / box_area
            >= minimum_overlap
        )

    def touches_source_edge(self, person: TrackedPerson, margin: int) -> bool:
        left, top, right, bottom = person.bounding_box
        return (
            left <= margin
            or top <= margin
            or right >= self.frame_width - 1 - margin
            or bottom >= self.frame_height - 1 - margin
        )

    def touches_roi_edge(self, person: TrackedPerson, margin: int) -> bool:
        left, top, right, bottom = person.bounding_box
        roi = self.roi
        return (
            left <= roi.left + margin
            or top <= roi.top + margin
            or right >= roi.right - 1 - margin
            or bottom >= roi.bottom - 1 - margin
        )

    def _validate_exclusions_do_not_cover_lanes(self) -> None:
        for exclusion_index, polygon in enumerate(self.exclusions):
            exclusion_mask = _polygon_mask(polygon, self.frame_width, self.frame_height)
            for lane, lane_mask in self._lane_masks.items():
                if np.any((exclusion_mask != 0) & (lane_mask != 0)):
                    raise CalibrationError(
                        "Exclusion polygon "
                        f"{exclusion_index} overlaps {lane.value}; exclusions must "
                        "cover static structures only."
                    )

    def _clip_person(self, person: TrackedPerson) -> TrackedPerson:
        left, top, right, bottom = person.bounding_box
        return TrackedPerson(
            tracking_id=person.tracking_id,
            bounding_box=(
                max(0.0, min(float(self.frame_width - 1), left)),
                max(0.0, min(float(self.frame_height - 1), top)),
                max(0.0, min(float(self.frame_width - 1), right)),
                max(0.0, min(float(self.frame_height - 1), bottom)),
            ),
            confidence=person.confidence,
        )


def box_iou(first: TrackedPerson, second: TrackedPerson) -> float:
    first_left, first_top, first_right, first_bottom = first.bounding_box
    second_left, second_top, second_right, second_bottom = second.bounding_box
    intersection_width = max(
        0.0, min(first_right, second_right) - max(first_left, second_left)
    )
    intersection_height = max(
        0.0, min(first_bottom, second_bottom) - max(first_top, second_top)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_right - first_left) * max(0.0, first_bottom - first_top)
    second_area = max(0.0, second_right - second_left) * max(
        0.0, second_bottom - second_top
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    first: TrackedPerson, second: TrackedPerson, frame_diagonal: float
) -> float:
    return math.dist(first.center, second.center) / frame_diagonal


def _pixel_roi(
    normalized: NormalizedRoi | None, frame_width: int, frame_height: int
) -> PixelRoi:
    if normalized is None:
        return PixelRoi(0, 0, frame_width, frame_height, False)
    left, top, right, bottom = normalized
    pixel_roi = PixelRoi(
        left=math.floor(left * frame_width),
        top=math.floor(top * frame_height),
        right=math.ceil(right * frame_width),
        bottom=math.ceil(bottom * frame_height),
        enabled=True,
    )
    if pixel_roi.width <= 0 or pixel_roi.height <= 0:
        raise CalibrationError("Detection ROI becomes empty at this resolution.")
    return pixel_roi


def _scale_polygon(
    points: NormalizedPolygon, frame_width: int, frame_height: int
) -> NDArray[np.int32]:
    return np.asarray(
        [
            (round(x * (frame_width - 1)), round(y * (frame_height - 1)))
            for x, y in points
        ],
        dtype=np.int32,
    )


def _polygon_mask(
    polygon: NDArray[np.int32], frame_width: int, frame_height: int
) -> NDArray[np.uint8]:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _clipped_box(
    box: tuple[float, float, float, float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, min(frame_width, math.floor(left))),
        max(0, min(frame_height, math.floor(top))),
        max(0, min(frame_width, math.ceil(right))),
        max(0, min(frame_height, math.ceil(bottom))),
    )
