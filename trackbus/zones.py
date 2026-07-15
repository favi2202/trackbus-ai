"""Resolution-independent polygon zones used by the passenger counter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.config import NormalizedPoint, ZonesConfig


class ZoneMembership(StrEnum):
    """The doorway region containing a tracked person's anchor point."""

    OUTSIDE = "outside"
    INSIDE = "inside"
    TRANSITION = "transition"


@dataclass(frozen=True)
class ZoneClassification:
    """Raw polygon result plus an optional boundary-stabilized result."""

    raw: ZoneMembership
    effective: ZoneMembership

    @property
    def boundary_jitter(self) -> bool:
        return self.raw is not self.effective


@dataclass(frozen=True)
class PixelZones:
    """Inside and outside polygons scaled to a particular video resolution."""

    outside: NDArray[np.int32]
    inside: NDArray[np.int32]
    frame_width: int
    frame_height: int

    @classmethod
    def from_normalized(
        cls, config: ZonesConfig, frame_width: int, frame_height: int
    ) -> PixelZones:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame width and height must be positive.")
        return cls(
            outside=_scale_polygon(config.outside, frame_width, frame_height),
            inside=_scale_polygon(config.inside, frame_width, frame_height),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def membership(self, point: tuple[float, float]) -> ZoneMembership:
        """Classify a pixel point; polygon boundaries count as inside the zone."""

        outside = cv2.pointPolygonTest(self.outside, point, False) >= 0
        inside = cv2.pointPolygonTest(self.inside, point, False) >= 0
        if outside and inside:
            return ZoneMembership.TRANSITION
        if outside:
            return ZoneMembership.OUTSIDE
        if inside:
            return ZoneMembership.INSIDE
        return ZoneMembership.TRANSITION

    def classify(
        self,
        point: tuple[float, float],
        boundary_hysteresis: float = 0.0,
    ) -> ZoneClassification:
        """Classify a point and treat shallow polygon-edge hits as neutral.

        ``boundary_hysteresis`` is normalized against the smaller source-frame
        dimension. It shrinks the effective zone interiors without changing the
        configured polygons or their raw diagnostic membership.
        """

        if not 0.0 <= boundary_hysteresis <= 0.10:
            raise ValueError("boundary_hysteresis must be between 0 and 0.10")
        raw = self.membership(point)
        if raw is ZoneMembership.TRANSITION or boundary_hysteresis == 0:
            return ZoneClassification(raw=raw, effective=raw)
        polygon = self.outside if raw is ZoneMembership.OUTSIDE else self.inside
        distance = cv2.pointPolygonTest(polygon, point, True)
        margin = boundary_hysteresis * min(self.frame_width, self.frame_height)
        effective = raw if distance >= margin else ZoneMembership.TRANSITION
        return ZoneClassification(raw=raw, effective=effective)


def _scale_polygon(
    points: tuple[NormalizedPoint, ...], frame_width: int, frame_height: int
) -> NDArray[np.int32]:
    # Multiplying by width - 1 keeps a normalized coordinate of 1.0 in-frame.
    return np.asarray(
        [
            (round(x * (frame_width - 1)), round(y * (frame_height - 1)))
            for x, y in points
        ],
        dtype=np.int32,
    )
