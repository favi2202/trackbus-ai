"""Class-aware source-coordinate detection fusion."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from trackbus.detection import BoundingBox, Detection


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection-over-union for two xyxy boxes."""

    first_left, first_top, first_right, first_bottom = first
    second_left, second_top, second_right, second_bottom = second
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


class NmsDetectionFusion:
    """Conservative cross-view, class-aware NMS with view provenance.

    Ultralytics has already applied NMS inside each prediction call.  Fusion is
    therefore deliberately limited to predictions from different inference
    views; applying a second NMS pass within one view could erase adjacent
    passengers that the detector intentionally retained.
    """

    method = "nms"

    def __init__(
        self,
        iou_threshold: float = 0.50,
        *,
        confidence_strategy: str = "maximum",
        prefer_full_frame: bool = False,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be greater than 0 and at most 1")
        if confidence_strategy != "maximum":
            raise ValueError("only the 'maximum' confidence strategy is supported")
        self.iou_threshold = iou_threshold
        self.confidence_strategy = confidence_strategy
        self.prefer_full_frame = prefer_full_frame

    def fuse(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        grouped: dict[int, list[tuple[int, Detection]]] = {}
        for index, detection in enumerate(detections):
            grouped.setdefault(detection.class_id, []).append((index, detection))

        fused_with_order: list[tuple[int, Detection]] = []
        for class_detections in grouped.values():
            remaining = sorted(
                class_detections,
                key=lambda item: self._sort_key(item[1], item[0]),
                reverse=True,
            )
            while remaining:
                winner_index, winner = remaining.pop(0)
                overlapping = [
                    candidate
                    for candidate in remaining
                    if self._is_cross_view_duplicate(winner, candidate[1])
                ]
                if self._spans_distinct_candidates(winner, overlapping):
                    # A single large detector box can cover two adjacent people.
                    # If the smaller hypotheses do not overlap one another, the
                    # large box is ambiguous and must not erase both histories.
                    overlapping_indexes = {index for index, _detection in overlapping}
                    remaining = [
                        (
                            index,
                            self._with_ambiguous_suppression(detection, winner),
                        )
                        if index in overlapping_indexes
                        else (index, detection)
                        for index, detection in remaining
                    ]
                    continue

                duplicates: list[tuple[int, Detection]] = []
                survivors: list[tuple[int, Detection]] = []
                claimed_views = set(winner.source_views)
                for candidate in remaining:
                    candidate_views = set(candidate[1].source_views)
                    if not claimed_views.intersection(
                        candidate_views
                    ) and self._is_cross_view_duplicate(winner, candidate[1]):
                        duplicates.append(candidate)
                        claimed_views.update(candidate_views)
                    else:
                        survivors.append(candidate)
                remaining = survivors
                fused_with_order.append(
                    (
                        winner_index,
                        self._with_provenance(winner, duplicates),
                    )
                )

        return [
            detection
            for _index, detection in sorted(fused_with_order, key=lambda item: item[0])
        ]

    def _sort_key(self, detection: Detection, original_index: int) -> tuple[Any, ...]:
        full_preference = (
            int(detection.metadata.get("view_bounds") == (0.0, 0.0, 1.0, 1.0))
            if self.prefer_full_frame
            else 0
        )
        return full_preference, detection.confidence, -original_index

    def _is_cross_view_duplicate(self, winner: Detection, candidate: Detection) -> bool:
        return bool(
            set(winner.source_views).isdisjoint(candidate.source_views)
            and box_iou(winner.bounding_box, candidate.bounding_box)
            >= self.iou_threshold
        )

    def _spans_distinct_candidates(
        self,
        winner: Detection,
        candidates: list[tuple[int, Detection]],
    ) -> bool:
        """Identify a large box that ambiguously spans separate hypotheses."""

        winner_area = _box_area(winner.bounding_box)
        if winner_area <= 0 or len(candidates) < 2:
            return False
        smaller = [
            detection
            for _index, detection in candidates
            if _box_area(detection.bounding_box) < winner_area
        ]
        return any(
            box_iou(first.bounding_box, second.bounding_box) < self.iou_threshold
            for index, first in enumerate(smaller)
            for second in smaller[index + 1 :]
        )

    def _with_ambiguous_suppression(
        self,
        detection: Detection,
        ambiguous: Detection,
    ) -> Detection:
        metadata = dict(detection.metadata)
        boxes = list(metadata.get("ambiguous_spanning_boxes_suppressed", []))
        boxes.append(list(ambiguous.bounding_box))
        metadata["ambiguous_spanning_boxes_suppressed"] = boxes
        metadata["ambiguous_spanning_suppressed_count"] = len(boxes)
        return replace(detection, metadata=metadata)

    def _with_provenance(
        self,
        winner: Detection,
        duplicates: list[tuple[int, Detection]],
    ) -> Detection:
        contributors = [winner, *(detection for _index, detection in duplicates)]
        views = tuple(
            dict.fromkeys(
                view for detection in contributors for view in detection.source_views
            )
        )
        metadata = dict(winner.metadata)
        contributing_view_bounds = {
            detection.source_view: detection.metadata.get("view_bounds")
            for detection in contributors
            if detection.metadata.get("view_bounds") is not None
        }
        metadata.update(
            {
                "fusion_method": self.method,
                "fusion_iou_threshold": self.iou_threshold,
                "confidence_strategy": self.confidence_strategy,
                "suppressed_count": len(duplicates),
                "source_detection_count": len(contributors),
                "contributing_view_bounds": contributing_view_bounds,
                "suppressed_boxes": [
                    list(detection.bounding_box) for _index, detection in duplicates
                ],
            }
        )
        return replace(
            winner,
            confidence=max(detection.confidence for detection in contributors),
            contributing_views=views,
            metadata=metadata,
        )


def _box_area(box: BoundingBox) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)
