"""Model-independent detection and tracking records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

BoundingBox = tuple[float, float, float, float]


class AnchorMode(StrEnum):
    """Supported box points for counting-zone membership."""

    CENTER = "center"
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"


def anchor_for_box(box: BoundingBox, mode: str | AnchorMode) -> tuple[float, float]:
    """Return a named anchor without changing exclusion-anchor compatibility."""

    try:
        parsed = AnchorMode(mode)
    except ValueError as exc:
        raise ValueError(
            "anchor mode must be center, bottom_center, or top_center"
        ) from exc
    left, top, right, bottom = box
    x = (left + right) / 2.0
    if parsed is AnchorMode.CENTER:
        return x, (top + bottom) / 2.0
    if parsed is AnchorMode.TOP_CENTER:
        return x, top
    return x, bottom


def _validate_box(box: BoundingBox) -> None:
    left, top, right, bottom = box
    if right < left or bottom < top:
        raise ValueError("bounding box must have left <= right and top <= bottom")


@dataclass(frozen=True)
class Detection:
    """One detector prediction expressed in the current coordinate space."""

    bounding_box: BoundingBox
    confidence: float
    class_id: int
    source_view: str
    contributing_views: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_box(self.bounding_box)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")
        if not self.source_view:
            raise ValueError("source_view cannot be empty")

    @property
    def anchor(self) -> tuple[float, float]:
        """Legacy bottom-center anchor used by exclusions and default counting."""

        return anchor_for_box(self.bounding_box, AnchorMode.BOTTOM_CENTER)

    @property
    def center(self) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, AnchorMode.CENTER)

    @property
    def top_center(self) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, AnchorMode.TOP_CENTER)

    def anchor_for(self, mode: str | AnchorMode) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, mode)

    @property
    def source_views(self) -> tuple[str, ...]:
        """All views known to have contributed to this prediction."""

        return self.contributing_views or (self.source_view,)

    def with_box(self, box: BoundingBox, **metadata: Any) -> Detection:
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata)
        return replace(self, bounding_box=box, metadata=merged_metadata)


@dataclass(frozen=True)
class TrackedDetection:
    """One fused detection associated with a temporary video-local track ID."""

    tracking_id: int
    bounding_box: BoundingBox
    confidence: float
    class_id: int = 0
    source_views: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tracking_id < 0:
            raise ValueError("tracking_id cannot be negative")
        _validate_box(self.bounding_box)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")

    @property
    def anchor(self) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, AnchorMode.BOTTOM_CENTER)

    @property
    def center(self) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, AnchorMode.CENTER)

    @property
    def top_center(self) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, AnchorMode.TOP_CENTER)

    def anchor_for(self, mode: str | AnchorMode) -> tuple[float, float]:
        return anchor_for_box(self.bounding_box, mode)

    def with_box(self, box: BoundingBox, **metadata: Any) -> TrackedDetection:
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata)
        return replace(self, bounding_box=box, metadata=merged_metadata)
