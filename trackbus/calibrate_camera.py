"""Interactive, model-free calibration of TrackBus camera geometry.

The GUI deliberately operates only on a decoded source frame.  It does not run
the detector, infer events, or inspect ground-truth annotations.  Configuration
loading and geometry validation are kept separate from the OpenCV event loop so
they can be tested without a graphical desktop.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

NormalizedPoint = tuple[float, float]
NormalizedPolygon = tuple[NormalizedPoint, ...]


class CameraCalibrationError(ValueError):
    """Raised when camera calibration input or geometry is invalid."""


class CalibrationLayer(StrEnum):
    """Editable calibration layers shown by the GUI."""

    INSIDE = "inside"
    OUTSIDE = "outside"
    CORRIDOR = "crossing_corridor"
    EXCLUSION = "exclusion_polygons"


class AnchorPreviewMode(StrEnum):
    """Box anchors available to the TrackBus zone counter."""

    CENTER = "center"
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"


@dataclass(frozen=True)
class CalibrationState:
    """Serializable geometry plus the currently drawn polygon draft."""

    inside: NormalizedPolygon | None = None
    outside: NormalizedPolygon | None = None
    crossing_corridor: NormalizedPolygon | None = None
    exclusion_polygons: tuple[NormalizedPolygon, ...] = ()
    anchor_mode: AnchorPreviewMode = AnchorPreviewMode.BOTTOM_CENTER
    draft: NormalizedPolygon = ()


@dataclass
class CalibrationSession:
    """Small undoable editor independent of OpenCV window management."""

    state: CalibrationState
    original_state: CalibrationState
    active_layer: CalibrationLayer = CalibrationLayer.INSIDE
    selected_vertex: tuple[CalibrationLayer, int, int] | None = None
    _history: list[CalibrationState] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(cls, document: Mapping[str, Any]) -> CalibrationSession:
        state = calibration_state_from_config(document)
        return cls(state=state, original_state=state)

    def select_layer(self, layer: CalibrationLayer | str) -> bool:
        """Select a layer, refusing to silently reassign an unfinished draft."""

        parsed = CalibrationLayer(layer)
        if self.state.draft and parsed is not self.active_layer:
            return False
        self.active_layer = parsed
        self.selected_vertex = None
        return True

    def add_point(self, point: NormalizedPoint) -> None:
        point = _normalized_point(point, "draft point")
        self._remember()
        self.state = replace(self.state, draft=(*self.state.draft, point))
        self.selected_vertex = None

    def finish_polygon(self) -> None:
        """Commit the draft to the active layer after local validation."""

        issues = validate_polygon(self.state.draft, self.active_layer.value)
        if issues:
            raise CameraCalibrationError("; ".join(issues))
        self._remember()
        draft = self.state.draft
        if self.active_layer is CalibrationLayer.INSIDE:
            self.state = replace(self.state, inside=draft, draft=())
        elif self.active_layer is CalibrationLayer.OUTSIDE:
            self.state = replace(self.state, outside=draft, draft=())
        elif self.active_layer is CalibrationLayer.CORRIDOR:
            self.state = replace(self.state, crossing_corridor=draft, draft=())
        else:
            self.state = replace(
                self.state,
                exclusion_polygons=(*self.state.exclusion_polygons, draft),
                draft=(),
            )

    def clear_active(self) -> None:
        """Clear the active layer (all polygons for the exclusion layer)."""

        self._remember()
        if self.active_layer is CalibrationLayer.INSIDE:
            self.state = replace(self.state, inside=None, draft=())
        elif self.active_layer is CalibrationLayer.OUTSIDE:
            self.state = replace(self.state, outside=None, draft=())
        elif self.active_layer is CalibrationLayer.CORRIDOR:
            self.state = replace(self.state, crossing_corridor=None, draft=())
        else:
            self.state = replace(self.state, exclusion_polygons=(), draft=())
        self.selected_vertex = None

    def reset(self) -> None:
        """Restore all geometry and the anchor to their startup values."""

        self._remember()
        self.state = self.original_state
        self.selected_vertex = None

    def undo(self) -> bool:
        if not self._history:
            return False
        self.state = self._history.pop()
        self.selected_vertex = None
        return True

    def cycle_anchor(self) -> AnchorPreviewMode:
        modes = tuple(AnchorPreviewMode)
        next_mode = modes[(modes.index(self.state.anchor_mode) + 1) % len(modes)]
        self._remember()
        self.state = replace(self.state, anchor_mode=next_mode)
        return next_mode

    def select_nearest_vertex(
        self, point: NormalizedPoint, *, maximum_distance: float = 0.04
    ) -> bool:
        """Select a vertex from the active layer for the next left-click move."""

        point = _normalized_point(point, "selection point")
        if maximum_distance <= 0 or not math.isfinite(maximum_distance):
            raise ValueError("maximum_distance must be a positive finite number")
        candidates: list[tuple[float, int, int]] = []
        for polygon_index, polygon in enumerate(self._active_polygons()):
            candidates.extend(
                (math.dist(point, vertex), polygon_index, vertex_index)
                for vertex_index, vertex in enumerate(polygon)
            )
        if not candidates:
            self.selected_vertex = None
            return False
        distance, polygon_index, vertex_index = min(candidates)
        if distance > maximum_distance:
            self.selected_vertex = None
            return False
        self.selected_vertex = self.active_layer, polygon_index, vertex_index
        return True

    def move_selected_vertex(self, point: NormalizedPoint) -> bool:
        """Move the selected vertex; final whole-calibration validation is later."""

        if self.selected_vertex is None:
            return False
        point = _normalized_point(point, "replacement point")
        layer, polygon_index, vertex_index = self.selected_vertex
        self._remember()
        polygons = [list(polygon) for polygon in self._polygons_for(layer)]
        polygons[polygon_index][vertex_index] = point
        updated = tuple(tuple(polygon) for polygon in polygons)
        self.state = self._replace_layer_polygons(layer, updated)
        self.selected_vertex = None
        return True

    def _remember(self) -> None:
        self._history.append(self.state)

    def _active_polygons(self) -> tuple[NormalizedPolygon, ...]:
        return self._polygons_for(self.active_layer)

    def _polygons_for(self, layer: CalibrationLayer) -> tuple[NormalizedPolygon, ...]:
        if layer is CalibrationLayer.INSIDE:
            return (self.state.inside,) if self.state.inside else ()
        if layer is CalibrationLayer.OUTSIDE:
            return (self.state.outside,) if self.state.outside else ()
        if layer is CalibrationLayer.CORRIDOR:
            return (
                (self.state.crossing_corridor,) if self.state.crossing_corridor else ()
            )
        return self.state.exclusion_polygons

    def _replace_layer_polygons(
        self,
        layer: CalibrationLayer,
        polygons: tuple[NormalizedPolygon, ...],
    ) -> CalibrationState:
        if layer is CalibrationLayer.INSIDE:
            return replace(self.state, inside=polygons[0])
        if layer is CalibrationLayer.OUTSIDE:
            return replace(self.state, outside=polygons[0])
        if layer is CalibrationLayer.CORRIDOR:
            return replace(self.state, crossing_corridor=polygons[0])
        return replace(self.state, exclusion_polygons=polygons)


def _normalized_point(value: object, name: str) -> NormalizedPoint:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise CameraCalibrationError(f"{name} must be an [x, y] point")
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"{name} must contain finite numbers") from exc
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise CameraCalibrationError(f"{name} must contain finite numbers")
    if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
        raise CameraCalibrationError(
            f"{name} coordinates must be normalized from 0.0 to 1.0"
        )
    return point


def _polygon_from_raw(value: object, name: str) -> NormalizedPolygon:
    if not isinstance(value, (tuple, list)):
        raise CameraCalibrationError(f"{name} must be a list of [x, y] points")
    return tuple(
        _normalized_point(point, f"{name}[{index}]")
        for index, point in enumerate(value)
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CameraCalibrationError(f"{name} must be a YAML mapping")
    return value


def calibration_state_from_config(document: Mapping[str, Any]) -> CalibrationState:
    """Extract only editor-owned fields from an otherwise opaque YAML mapping."""

    zones = _mapping(document.get("zones"), "zones")
    camera = _mapping(document.get("camera"), "camera")
    tracking = _mapping(document.get("tracking"), "tracking")

    def optional_polygon(value: object, name: str) -> NormalizedPolygon | None:
        return None if value is None else _polygon_from_raw(value, name)

    raw_exclusions = camera.get("exclusion_polygons", [])
    if raw_exclusions is None:
        raw_exclusions = []
    if not isinstance(raw_exclusions, (tuple, list)):
        raise CameraCalibrationError("camera.exclusion_polygons must be a list")
    exclusions = tuple(
        _polygon_from_raw(polygon, f"camera.exclusion_polygons[{index}]")
        for index, polygon in enumerate(raw_exclusions)
    )
    raw_anchor = tracking.get("zone_anchor", AnchorPreviewMode.BOTTOM_CENTER.value)
    try:
        anchor = AnchorPreviewMode(str(raw_anchor))
    except ValueError as exc:
        raise CameraCalibrationError(
            "tracking.zone_anchor must be center, bottom_center, or top_center"
        ) from exc
    state = CalibrationState(
        inside=optional_polygon(zones.get("inside"), "zones.inside"),
        outside=optional_polygon(zones.get("outside"), "zones.outside"),
        crossing_corridor=optional_polygon(
            camera.get("crossing_corridor"), "camera.crossing_corridor"
        ),
        exclusion_polygons=exclusions,
        anchor_mode=anchor,
    )
    existing_issues = []
    for name, polygon in (
        ("zones.inside", state.inside),
        ("zones.outside", state.outside),
        ("camera.crossing_corridor", state.crossing_corridor),
    ):
        if polygon is not None:
            existing_issues.extend(validate_polygon(polygon, name))
    for index, polygon in enumerate(exclusions):
        existing_issues.extend(
            validate_polygon(polygon, f"camera.exclusion_polygons[{index}]")
        )
    if existing_issues:
        raise CameraCalibrationError("; ".join(existing_issues))
    return state


def load_calibration_config(path: Path) -> dict[str, Any]:
    """Load an existing YAML mapping, or start an empty configuration."""

    if not path.exists():
        return {}
    if not path.is_file():
        raise CameraCalibrationError(f"output config is not a file: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CameraCalibrationError(f"could not read config '{path}': {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise CameraCalibrationError("configuration root must be a YAML mapping")
    return loaded


def validate_polygon(
    polygon: NormalizedPolygon, name: str = "polygon"
) -> tuple[str, ...]:
    """Return deterministic validation issues for one normalized polygon."""

    issues: list[str] = []
    if len(polygon) < 3:
        issues.append(f"{name} must contain at least three points")
        return tuple(issues)
    parsed: list[NormalizedPoint] = []
    for index, point in enumerate(polygon):
        try:
            parsed.append(_normalized_point(point, f"{name}[{index}]"))
        except CameraCalibrationError as exc:
            issues.append(str(exc))
    if issues:
        return tuple(issues)
    twice_area = abs(
        sum(
            parsed[index][0] * parsed[(index + 1) % len(parsed)][1]
            - parsed[(index + 1) % len(parsed)][0] * parsed[index][1]
            for index in range(len(parsed))
        )
    )
    if twice_area <= 1e-9:
        issues.append(f"{name} must have a non-zero area")
    return tuple(issues)


def _polygon_mask(polygon: NormalizedPolygon, size: int = 512) -> np.ndarray:
    points = np.asarray(
        [(round(x * (size - 1)), round(y * (size - 1))) for x, y in polygon],
        dtype=np.int32,
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def _has_usable_neutral_region(
    inside_mask: np.ndarray, outside_mask: np.ndarray
) -> bool:
    neutral = ((inside_mask == 0) & (outside_mask == 0)).astype(np.uint8)
    if np.count_nonzero(neutral) < neutral.size * 0.01:
        return False
    component_count, labels = cv2.connectedComponents(neutral, connectivity=8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    inside_edge = cv2.dilate(inside_mask, kernel) != 0
    outside_edge = cv2.dilate(outside_mask, kernel) != 0
    minimum_area = neutral.size * 0.005
    for label in range(1, component_count):
        component = labels == label
        if np.count_nonzero(component) < minimum_area:
            continue
        if np.any(component & inside_edge) and np.any(component & outside_edge):
            return True
    return False


def validate_calibration(state: CalibrationState) -> tuple[str, ...]:
    """Validate complete geometry using a resolution-independent raster test.

    Overlap is considered substantial when it covers more than ten percent of
    the smaller counting zone.  A neutral connected component must have useful
    area and touch both zones.  When a corridor is configured, it must contain
    positive-area pieces of both zones and of their neutral transition area.
    """

    issues: list[str] = []
    if state.draft:
        issues.append("finish or clear the current polygon draft before saving")
    if state.inside is None:
        issues.append("zones.inside is required")
    else:
        issues.extend(validate_polygon(state.inside, "zones.inside"))
    if state.outside is None:
        issues.append("zones.outside is required")
    else:
        issues.extend(validate_polygon(state.outside, "zones.outside"))
    if state.crossing_corridor is not None:
        issues.extend(
            validate_polygon(state.crossing_corridor, "camera.crossing_corridor")
        )
    for index, polygon in enumerate(state.exclusion_polygons):
        issues.extend(validate_polygon(polygon, f"camera.exclusion_polygons[{index}]"))
    if issues or state.inside is None or state.outside is None:
        return tuple(issues)

    inside_mask = _polygon_mask(state.inside)
    outside_mask = _polygon_mask(state.outside)
    intersection = np.count_nonzero((inside_mask != 0) & (outside_mask != 0))
    smaller_area = min(np.count_nonzero(inside_mask), np.count_nonzero(outside_mask))
    overlap_ratio = intersection / max(1, smaller_area)
    if overlap_ratio > 0.10:
        issues.append(
            "zones.inside and zones.outside substantially overlap "
            f"({overlap_ratio:.1%} of the smaller zone)"
        )
    if not _has_usable_neutral_region(inside_mask, outside_mask):
        issues.append(
            "zones.inside and zones.outside do not leave a usable neutral "
            "transition region connected to both zones"
        )

    corridor_mask = (
        _polygon_mask(state.crossing_corridor)
        if state.crossing_corridor is not None
        else None
    )
    if corridor_mask is not None:
        corridor_area = max(1, np.count_nonzero(corridor_mask))
        inside_area = np.count_nonzero((corridor_mask != 0) & (inside_mask != 0))
        outside_area = np.count_nonzero((corridor_mask != 0) & (outside_mask != 0))
        neutral_area = np.count_nonzero(
            (corridor_mask != 0) & (inside_mask == 0) & (outside_mask == 0)
        )
        if inside_area == 0 or outside_area == 0:
            issues.append(
                "camera.crossing_corridor must intersect both zones.inside "
                "and zones.outside"
            )
        if neutral_area < max(4, round(corridor_area * 0.01)):
            issues.append(
                "camera.crossing_corridor must include usable neutral area "
                "between the counting zones"
            )

    passenger_paths = [
        ("zones.inside", inside_mask),
        ("zones.outside", outside_mask),
    ]
    if corridor_mask is not None:
        passenger_paths.append(("camera.crossing_corridor", corridor_mask))
    for exclusion_index, exclusion in enumerate(state.exclusion_polygons):
        exclusion_mask = _polygon_mask(exclusion)
        for pathway_name, pathway_mask in passenger_paths:
            if np.any((exclusion_mask != 0) & (pathway_mask != 0)):
                issues.append(
                    f"camera.exclusion_polygons[{exclusion_index}] overlaps "
                    f"{pathway_name}; exclusions must not cover passenger paths"
                )
    return tuple(issues)


def apply_calibration_to_config(
    document: Mapping[str, Any], state: CalibrationState
) -> dict[str, Any]:
    """Return a deep copy with only camera-calibration fields replaced."""

    issues = validate_calibration(state)
    if issues:
        raise CameraCalibrationError("; ".join(issues))
    assert state.inside is not None  # narrowed by validate_calibration
    assert state.outside is not None
    updated = copy.deepcopy(dict(document))
    raw_zones = updated.get("zones")
    if raw_zones is not None and not isinstance(raw_zones, dict):
        raise CameraCalibrationError("zones must be a YAML mapping")
    zones = raw_zones if isinstance(raw_zones, dict) else {}
    zones["inside"] = _polygon_to_yaml(state.inside)
    zones["outside"] = _polygon_to_yaml(state.outside)
    updated["zones"] = zones

    raw_camera = updated.get("camera")
    if raw_camera is not None and not isinstance(raw_camera, dict):
        raise CameraCalibrationError("camera must be a YAML mapping")
    camera = raw_camera if isinstance(raw_camera, dict) else {}
    if state.crossing_corridor is None:
        camera.pop("crossing_corridor", None)
    else:
        camera["crossing_corridor"] = _polygon_to_yaml(state.crossing_corridor)
    camera["exclusion_polygons"] = [
        _polygon_to_yaml(polygon) for polygon in state.exclusion_polygons
    ]
    updated["camera"] = camera

    raw_tracking = updated.get("tracking")
    if raw_tracking is not None and not isinstance(raw_tracking, dict):
        raise CameraCalibrationError("tracking must be a YAML mapping")
    tracking = raw_tracking if isinstance(raw_tracking, dict) else {}
    tracking["zone_anchor"] = state.anchor_mode.value
    updated["tracking"] = tracking
    return updated


def _polygon_to_yaml(polygon: NormalizedPolygon) -> list[list[float]]:
    return [[round(x, 6), round(y, 6)] for x, y in polygon]


def save_calibration_config(
    output_path: Path,
    document: Mapping[str, Any],
    state: CalibrationState,
) -> None:
    """Validate and atomically save a calibration while preserving other keys."""

    updated = apply_calibration_to_config(document, state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(
        updated,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


_LAYER_COLORS: dict[CalibrationLayer, tuple[int, int, int]] = {
    CalibrationLayer.INSIDE: (60, 210, 60),
    CalibrationLayer.OUTSIDE: (240, 130, 30),
    CalibrationLayer.CORRIDOR: (0, 220, 255),
    CalibrationLayer.EXCLUSION: (40, 40, 230),
}


def _pixel_point(
    point: NormalizedPoint, frame_width: int, frame_height: int
) -> tuple[int, int]:
    return (
        round(point[0] * max(0, frame_width - 1)),
        round(point[1] * max(0, frame_height - 1)),
    )


def _draw_polygon(
    display: np.ndarray,
    polygon: NormalizedPolygon,
    color: tuple[int, int, int],
    *,
    selected: tuple[int, int] | None = None,
) -> None:
    height, width = display.shape[:2]
    points = np.asarray(
        [_pixel_point(point, width, height) for point in polygon], dtype=np.int32
    )
    if len(points) >= 3:
        fill = display.copy()
        cv2.fillPoly(fill, [points], color)
        cv2.addWeighted(fill, 0.16, display, 0.84, 0.0, display)
        cv2.polylines(display, [points], True, color, 2, cv2.LINE_AA)
    elif len(points) >= 2:
        cv2.polylines(display, [points], False, color, 2, cv2.LINE_AA)
    for vertex_index, point in enumerate(points):
        vertex_color = (255, 255, 255)
        radius = 4
        if selected is not None and selected[1] == vertex_index:
            vertex_color = (0, 255, 255)
            radius = 7
        cv2.circle(display, tuple(point), radius, vertex_color, -1, cv2.LINE_AA)


def _anchor_for_preview_box(
    box: tuple[int, int, int, int], mode: AnchorPreviewMode
) -> tuple[int, int]:
    left, top, right, bottom = box
    x = round((left + right) / 2)
    if mode is AnchorPreviewMode.CENTER:
        return x, round((top + bottom) / 2)
    if mode is AnchorPreviewMode.TOP_CENTER:
        return x, top
    return x, bottom


def draw_calibration_overlay(
    frame: np.ndarray,
    session: CalibrationSession,
    *,
    frame_number: int,
    frame_count: int,
    cursor: tuple[int, int] | None = None,
    status: str = "",
) -> np.ndarray:
    """Render polygons, editing instructions, and a detector-free anchor preview."""

    display = frame.copy()
    height, width = display.shape[:2]
    layers = (
        (
            CalibrationLayer.INSIDE,
            (session.state.inside,) if session.state.inside else (),
        ),
        (
            CalibrationLayer.OUTSIDE,
            (session.state.outside,) if session.state.outside else (),
        ),
        (
            CalibrationLayer.CORRIDOR,
            (session.state.crossing_corridor,)
            if session.state.crossing_corridor
            else (),
        ),
        (CalibrationLayer.EXCLUSION, session.state.exclusion_polygons),
    )
    selected = session.selected_vertex
    for layer, polygons in layers:
        for polygon_index, polygon in enumerate(polygons):
            vertex_selection = None
            if selected and selected[0] is layer and selected[1] == polygon_index:
                vertex_selection = selected[1], selected[2]
            _draw_polygon(
                display,
                polygon,
                _LAYER_COLORS[layer],
                selected=vertex_selection,
            )
    if session.state.draft:
        _draw_polygon(display, session.state.draft, _LAYER_COLORS[session.active_layer])

    if cursor is None:
        cursor = round(width * 0.82), round(height * 0.58)
    box_width = max(24, round(width * 0.12))
    box_height = max(36, round(height * 0.24))
    center_x = min(max(cursor[0], box_width // 2), width - 1 - box_width // 2)
    center_y = min(max(cursor[1], box_height // 2), height - 1 - box_height // 2)
    preview_box = (
        center_x - box_width // 2,
        center_y - box_height // 2,
        center_x + box_width // 2,
        center_y + box_height // 2,
    )
    cv2.rectangle(
        display,
        preview_box[:2],
        preview_box[2:],
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    anchor = _anchor_for_preview_box(preview_box, session.state.anchor_mode)
    cv2.circle(display, anchor, 5, (255, 0, 255), -1, cv2.LINE_AA)

    scale = max(0.38, min(width, height) / 850.0)
    line_height = max(15, round(25 * scale))
    instructions = [
        f"Frame {frame_number}/{max(0, frame_count - 1)} | "
        f"Layer: {session.active_layer.value} | "
        f"Anchor: {session.state.anchor_mode.value}",
        "1 inside | 2 outside | 3 corridor | 4 exclusions | M anchor",
        "Left: add/move | Right: select vertex | Enter: finish | U: undo | C: clear",
        "[ / ]: frame | J / K: 1 second | R: reset | S: validate+save | Q/Esc: cancel",
    ]
    if status:
        instructions.append(status)
    panel_height = min(height, line_height * len(instructions) + 8)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0.0, display)
    for row, text in enumerate(instructions, start=1):
        cv2.putText(
            display,
            text,
            (7, row * line_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return display


def calibrate_camera(
    input_path: Path,
    output_config: Path,
    *,
    frame_number: int = 0,
) -> bool:
    """Run the interactive editor; return true only when a config was saved."""

    if input_path.expanduser().resolve() == output_config.expanduser().resolve():
        raise CameraCalibrationError("input video and output config must differ")
    if frame_number < 0:
        raise CameraCalibrationError("representative frame cannot be negative")

    document = load_calibration_config(output_config)
    session = CalibrationSession.from_config(document)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise CameraCalibrationError(f"could not open input video: {input_path}")
    window_name = f"TrackBus camera calibration - {input_path.name}"
    current_frame = frame_number
    cursor: tuple[int, int] | None = None
    status = ""
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise CameraCalibrationError("input video contains no readable frames")
        if not math.isfinite(fps) or fps <= 0:
            raise CameraCalibrationError("input video reports an invalid FPS")
        if current_frame >= frame_count:
            raise CameraCalibrationError(
                f"representative frame {current_frame} is outside the video "
                f"(0-{frame_count - 1})"
            )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(
            event: int, x: int, y: int, _flags: int, _parameter: object
        ) -> None:
            nonlocal cursor, status
            cursor = x, y
            width = max(1, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(1, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            point = (
                min(1.0, max(0.0, x / max(1, width - 1))),
                min(1.0, max(0.0, y / max(1, height - 1))),
            )
            if event == cv2.EVENT_RBUTTONDOWN:
                selected = session.select_nearest_vertex(point)
                status = (
                    "Vertex selected; left-click its new position."
                    if selected
                    else "No nearby vertex."
                )
            elif event == cv2.EVENT_LBUTTONDOWN:
                if session.move_selected_vertex(point):
                    status = "Vertex moved."
                else:
                    session.add_point(point)
                    status = f"Draft vertices: {len(session.state.draft)}"

        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ok, frame = capture.read()
            if not ok:
                raise CameraCalibrationError(
                    f"could not decode representative frame {current_frame}"
                )
            display = draw_calibration_overlay(
                frame,
                session,
                frame_number=current_frame,
                frame_count=frame_count,
                cursor=cursor,
                status=status,
            )
            cv2.imshow(window_name, display)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            key = cv2.waitKeyEx(0)
            status = ""
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                layer = {
                    ord("1"): CalibrationLayer.INSIDE,
                    ord("2"): CalibrationLayer.OUTSIDE,
                    ord("3"): CalibrationLayer.CORRIDOR,
                    ord("4"): CalibrationLayer.EXCLUSION,
                }[key]
                if not session.select_layer(layer):
                    status = "Finish, undo, or clear the current draft first."
            elif key in (10, 13):
                try:
                    session.finish_polygon()
                    status = "Polygon committed."
                except CameraCalibrationError as exc:
                    status = str(exc)
            elif key in (ord("u"), ord("U")):
                status = "Undone." if session.undo() else "Nothing to undo."
            elif key in (ord("c"), ord("C")):
                session.clear_active()
                status = "Active layer cleared."
            elif key in (ord("r"), ord("R")):
                session.reset()
                status = "Startup calibration restored."
            elif key in (ord("m"), ord("M")):
                status = f"Anchor: {session.cycle_anchor().value}"
            elif key in (ord("s"), ord("S")):
                issues = validate_calibration(session.state)
                if issues:
                    status = "Cannot save: " + "; ".join(issues)
                else:
                    save_calibration_config(output_config, document, session.state)
                    return True
            elif key in (ord("["), 81, 2424832):
                current_frame = max(0, current_frame - 1)
            elif key in (ord("]"), 83, 2555904):
                current_frame = min(frame_count - 1, current_frame + 1)
            elif key in (ord("j"), ord("J")):
                current_frame = max(0, current_frame - max(1, round(fps)))
            elif key in (ord("k"), ord("K")):
                current_frame = min(frame_count - 1, current_frame + max(1, round(fps)))
    except cv2.error as exc:
        raise CameraCalibrationError(
            "OpenCV could not create or update the calibration window; "
            "run this command in a graphical desktop session"
        ) from exc
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.calibrate_camera",
        description=(
            "Draw normalized TrackBus counting zones on a representative video frame."
        ),
        epilog=(
            "Controls: 1 inside; 2 outside; 3 corridor; 4 exclusions; left-click "
            "add/move; right-click select vertex; Enter finish; U undo; C clear; "
            "R reset; M cycle anchor; [/] select frame; J/K one second; S validate "
            "and save; Q/Esc cancel without saving."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Local source video")
    parser.add_argument(
        "--output-config",
        required=True,
        type=Path,
        help="YAML config to create or update atomically",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Initial representative frame number (default: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_config = args.output_config.expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input video does not exist: {input_path}", file=sys.stderr)
        return 2
    if input_path == output_config:
        print("error: input video and output config must differ", file=sys.stderr)
        return 2
    try:
        saved = calibrate_camera(
            input_path,
            output_config,
            frame_number=args.frame,
        )
    except (CameraCalibrationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if saved:
        print(f"Saved normalized camera calibration to {output_config}")
    else:
        print("Calibration cancelled; no configuration changes were written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
