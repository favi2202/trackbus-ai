"""Configuration loading and validation for TrackBus."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""


NormalizedPoint = tuple[float, float]
NormalizedPolygon = tuple[NormalizedPoint, ...]
NormalizedRoi = tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    path: str = "yolo11n.pt"
    confidence: float = 0.35
    detector_floor: float | None = None
    imgsz: int = 640
    precision: str = "fp32"
    half: bool = False
    legacy_half_configured: bool = False

    @property
    def effective_detector_floor(self) -> float:
        """Return the actual YOLO inference floor with legacy compatibility."""

        return self.confidence if self.detector_floor is None else self.detector_floor


@dataclass(frozen=True)
class ContinuityConfig:
    """Optional, bounded TrackBus-level anonymous track continuity."""

    enabled: bool = False
    max_gap_frames: int = 3
    max_centroid_distance: float = 0.08
    minimum_iou: float = 0.05
    maximum_size_ratio: float = 1.8
    minimum_direction_cosine: float = -0.25
    minimum_match_score: float = 0.55


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    minimum_zone_frames: int = 3
    minimum_origin_zone_frames: int | None = None
    minimum_destination_zone_frames: int | None = None
    maximum_transition_gap_frames: int = 15
    event_cooldown_frames: int = 30
    zone_anchor: str = "bottom_center"
    zone_boundary_hysteresis: float = 0.0
    stale_track_timeout: int = 90
    track_high_thresh: float | None = None
    track_low_thresh: float | None = None
    new_track_thresh: float | None = None
    track_buffer: int | None = None
    match_thresh: float | None = None
    fuse_score: bool | None = None
    continuity: ContinuityConfig = ContinuityConfig()

    @property
    def effective_minimum_origin_zone_frames(self) -> int:
        return (
            self.minimum_zone_frames
            if self.minimum_origin_zone_frames is None
            else self.minimum_origin_zone_frames
        )

    @property
    def effective_minimum_destination_zone_frames(self) -> int:
        return (
            self.minimum_zone_frames
            if self.minimum_destination_zone_frames is None
            else self.minimum_destination_zone_frames
        )

    def bytetrack_overrides(self) -> dict[str, float | int | bool]:
        """Return only explicitly configured internal ByteTrack values."""

        values = {
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
        }
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True)
class InferenceViewConfig:
    """One normalized rectangular detector input view."""

    name: str
    bounds: NormalizedRoi
    enabled: bool = True


@dataclass(frozen=True)
class DetectionFusionConfig:
    method: str = "nms"
    iou_threshold: float = 0.50
    confidence_strategy: str = "maximum"
    prefer_full_frame: bool = False


@dataclass(frozen=True)
class CameraConfig:
    detection_roi: NormalizedRoi | None = None
    inference_views: tuple[InferenceViewConfig, ...] = ()
    left_lane: NormalizedPolygon | None = None
    center_lane: NormalizedPolygon | None = None
    right_lane: NormalizedPolygon | None = None
    lane_anchor: str = "center"
    crossing_corridor: NormalizedPolygon | None = None
    exclusion_polygons: tuple[NormalizedPolygon, ...] = ()
    debug_calibration_overlay: bool = False


@dataclass(frozen=True)
class FailureMiningConfig:
    """Bounded, local-only evidence collection for difficult frames."""

    enabled: bool = False
    output_jsonl: Path | None = None
    capture_frames: bool = False
    frames_directory: Path | None = None
    maximum_records: int = 500
    maximum_captured_frames: int = 50
    jpeg_quality: int = 85
    low_confidence_threshold: float | None = None
    short_track_maximum_frames: int = 5
    sudden_detection_drop_ratio: float = 0.50


@dataclass(frozen=True)
class DiagnosticsConfig:
    nearby_distance_normalized: float = 0.12
    heavy_overlap_iou: float = 0.50
    id_restart_window_frames: int = 15
    id_restart_distance_normalized: float = 0.08
    edge_margin_pixels: int = 2
    stationary_step_threshold: float = 0.005
    likely_static_minimum_frames: int = 75
    likely_static_minimum_percentage: float = 0.90
    wide_box_aspect_ratio: float = 0.90
    wide_box_doorway_ratio: float = 0.45
    multi_lane_minimum_box_overlap: float = 0.05
    export_detection_csv: bool = False
    debug_visualization: bool = False
    trajectory_length: int = 30
    failure_mining: FailureMiningConfig = FailureMiningConfig()


@dataclass(frozen=True)
class ZonesConfig:
    outside: tuple[NormalizedPoint, ...]
    inside: tuple[NormalizedPoint, ...]


@dataclass(frozen=True)
class OutputConfig:
    video: Path = Path("data/output/result.mp4")
    events_csv: Path | None = None
    summary_json: Path | None = None
    raw_detections_csv: Path | None = None
    fused_detections_csv: Path | None = None
    tracks_csv: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings for a TrackBus processing run."""

    model: ModelConfig
    tracking: TrackingConfig
    zones: ZonesConfig
    outputs: OutputConfig
    camera: CameraConfig = CameraConfig()
    detection_fusion: DetectionFusionConfig = DetectionFusionConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    capacity: int = 40
    initial_occupancy: int = 0
    device: str = "auto"
    logging_level: str = "INFO"

    def with_overrides(
        self,
        *,
        output: Path | None = None,
        capacity: int | None = None,
        initial_occupancy: int | None = None,
        device: str | None = None,
        confidence: float | None = None,
        detector_floor: float | None = None,
        model: str | None = None,
        imgsz: int | None = None,
    ) -> AppConfig:
        """Return a validated copy with command-line values applied."""

        updated_model = replace(
            self.model,
            path=model if model is not None else self.model.path,
            confidence=(
                confidence if confidence is not None else self.model.confidence
            ),
            detector_floor=(
                detector_floor
                if detector_floor is not None
                else self.model.detector_floor
            ),
            imgsz=imgsz if imgsz is not None else self.model.imgsz,
        )
        updated_outputs = replace(
            self.outputs,
            video=output if output is not None else self.outputs.video,
        )
        updated = replace(
            self,
            model=updated_model,
            outputs=updated_outputs,
            capacity=capacity if capacity is not None else self.capacity,
            initial_occupancy=(
                initial_occupancy
                if initial_occupancy is not None
                else self.initial_occupancy
            ),
            device=device if device is not None else self.device,
        )
        validate_config(updated)
        return updated


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a YAML mapping.")
    return value


def _reject_unknown_keys(
    value: dict[str, Any], name: str, allowed_keys: set[str]
) -> None:
    unknown_keys = sorted((key for key in value if key not in allowed_keys), key=str)
    if unknown_keys:
        rendered = ", ".join(repr(key) for key in unknown_keys)
        raise ConfigError(f"'{name}' contains unknown key(s): {rendered}.")


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a finite number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{name}' must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"'{name}' must be a finite number.")
    return parsed


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"'{name}' must be an integer, not a boolean.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ConfigError(f"'{name}' must be an integer.")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"'{name}' must be an integer.") from exc


def _polygon(value: Any, name: str) -> tuple[NormalizedPoint, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ConfigError(f"'{name}' must contain at least three [x, y] points.")

    points: list[NormalizedPoint] = []
    for index, raw_point in enumerate(value):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ConfigError(f"'{name}[{index}]' must be an [x, y] point.")
        point = (
            _float(raw_point[0], f"{name}[{index}][0]"),
            _float(raw_point[1], f"{name}[{index}][1]"),
        )
        if not all(0.0 <= coordinate <= 1.0 for coordinate in point):
            raise ConfigError(
                f"'{name}[{index}]' coordinates must be normalized from 0.0 to 1.0."
            )
        points.append(point)

    twice_area = abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    )
    if twice_area <= 1e-9:
        raise ConfigError(f"'{name}' must describe a polygon with a non-zero area.")
    return tuple(points)


def _optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty path or null.")
    return Path(value)


def _optional_roi(value: Any, name: str) -> NormalizedRoi | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(f"'{name}' must be [left, top, right, bottom] or null.")
    left, top, right, bottom = (
        _float(coordinate, f"{name}[{index}]") for index, coordinate in enumerate(value)
    )
    if not all(0.0 <= coordinate <= 1.0 for coordinate in (left, top, right, bottom)):
        raise ConfigError(f"'{name}' coordinates must be normalized from 0.0 to 1.0.")
    if left >= right or top >= bottom:
        raise ConfigError(f"'{name}' must have left < right and top < bottom.")
    return left, top, right, bottom


def _optional_polygon(value: Any, name: str) -> NormalizedPolygon | None:
    return None if value is None else _polygon(value, name)


def _polygon_collection(value: Any, name: str) -> tuple[NormalizedPolygon, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{name}' must be a list of polygons.")
    return tuple(
        _polygon(polygon, f"{name}[{index}]") for index, polygon in enumerate(value)
    )


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false.")
    return value


def _optional_bool(value: Any, name: str) -> bool | None:
    return None if value is None else _bool(value, name)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _float(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _int(value, name)


def _inference_views(value: Any, name: str) -> tuple[InferenceViewConfig, ...]:
    if value is None:
        raise ConfigError(
            f"'{name}' cannot be null; omit it for the default full-frame view "
            "or provide a non-empty list."
        )
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"'{name}' must be a list of view mappings.")
    if not value:
        raise ConfigError(
            f"'{name}' cannot be empty; omit it for the default full-frame view "
            "or provide at least one view."
        )
    views: list[InferenceViewConfig] = []
    for index, raw_view in enumerate(value):
        view = _mapping(raw_view, f"{name}[{index}]")
        _reject_unknown_keys(view, f"{name}[{index}]", {"name", "bounds", "enabled"})
        raw_name = view.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigError(f"'{name}[{index}].name' must be a non-empty string.")
        bounds = _optional_roi(view.get("bounds"), f"{name}[{index}].bounds")
        if bounds is None:
            raise ConfigError(f"'{name}[{index}].bounds' cannot be null.")
        views.append(
            InferenceViewConfig(
                name=raw_name.strip(),
                bounds=bounds,
                enabled=_bool(view.get("enabled", True), f"{name}[{index}].enabled"),
            )
        )
    return tuple(views)


def load_config(path: Path) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file."""

    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read configuration file '{path}': {exc}") from exc

    root = _mapping(raw, "configuration")
    _reject_unknown_keys(
        root,
        "configuration",
        {
            "model",
            "tracking",
            "zones",
            "outputs",
            "camera",
            "diagnostics",
            "detection_fusion",
            "capacity",
            "initial_occupancy",
            "device",
            "logging_level",
        },
    )
    model_raw = _mapping(root.get("model", {}), "model")
    tracking_raw = _mapping(root.get("tracking", {}), "tracking")
    continuity_raw = _mapping(tracking_raw.get("continuity", {}), "tracking.continuity")
    zones_raw = _mapping(root.get("zones"), "zones")
    outputs_raw = _mapping(root.get("outputs", {}), "outputs")
    camera_raw = _mapping(root.get("camera", {}), "camera")
    lanes_raw = _mapping(camera_raw.get("doorway_lanes", {}), "camera.doorway_lanes")
    diagnostics_raw = _mapping(root.get("diagnostics", {}), "diagnostics")
    failure_mining_raw = _mapping(
        diagnostics_raw.get("failure_mining", {}),
        "diagnostics.failure_mining",
    )
    fusion_raw = _mapping(root.get("detection_fusion", {}), "detection_fusion")

    _reject_unknown_keys(
        model_raw,
        "model",
        {"path", "confidence", "detector_floor", "imgsz", "precision", "half"},
    )
    if "precision" in model_raw and "half" in model_raw:
        raise ConfigError(
            "'model.precision' cannot be combined with deprecated 'model.half'."
        )
    _reject_unknown_keys(
        tracking_raw,
        "tracking",
        {
            "tracker",
            "minimum_zone_frames",
            "minimum_origin_zone_frames",
            "minimum_destination_zone_frames",
            "maximum_transition_gap_frames",
            "event_cooldown_frames",
            "zone_anchor",
            "zone_boundary_hysteresis",
            "stale_track_timeout",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "continuity",
        },
    )
    _reject_unknown_keys(
        continuity_raw,
        "tracking.continuity",
        {
            "enabled",
            "max_gap_frames",
            "max_centroid_distance",
            "minimum_iou",
            "maximum_size_ratio",
            "minimum_direction_cosine",
            "minimum_match_score",
        },
    )
    _reject_unknown_keys(zones_raw, "zones", {"outside", "inside"})
    _reject_unknown_keys(
        outputs_raw,
        "outputs",
        {
            "video",
            "events_csv",
            "summary_json",
            "raw_detections_csv",
            "fused_detections_csv",
            "tracks_csv",
        },
    )
    _reject_unknown_keys(
        camera_raw,
        "camera",
        {
            "detection_roi",
            "inference_views",
            "doorway_lanes",
            "crossing_corridor",
            "exclusion_polygons",
            "debug_calibration_overlay",
        },
    )
    _reject_unknown_keys(
        lanes_raw,
        "camera.doorway_lanes",
        {"anchor", "left_lane", "center_lane", "right_lane"},
    )
    _reject_unknown_keys(
        diagnostics_raw,
        "diagnostics",
        {
            "nearby_distance_normalized",
            "heavy_overlap_iou",
            "id_restart_window_frames",
            "id_restart_distance_normalized",
            "edge_margin_pixels",
            "stationary_step_threshold",
            "likely_static_minimum_frames",
            "likely_static_minimum_percentage",
            "wide_box_aspect_ratio",
            "wide_box_doorway_ratio",
            "multi_lane_minimum_box_overlap",
            "export_detection_csv",
            "debug_visualization",
            "trajectory_length",
            "failure_mining",
        },
    )
    _reject_unknown_keys(
        failure_mining_raw,
        "diagnostics.failure_mining",
        {
            "enabled",
            "output_jsonl",
            "capture_frames",
            "frames_directory",
            "maximum_records",
            "maximum_captured_frames",
            "jpeg_quality",
            "low_confidence_threshold",
            "short_track_maximum_frames",
            "sudden_detection_drop_ratio",
        },
    )
    _reject_unknown_keys(
        fusion_raw,
        "detection_fusion",
        {"method", "iou_threshold", "confidence_strategy", "prefer_full_frame"},
    )

    inference_views = (
        _inference_views(camera_raw["inference_views"], "camera.inference_views")
        if "inference_views" in camera_raw
        else ()
    )

    legacy_half = (
        _bool(model_raw["half"], "model.half") if "half" in model_raw else None
    )
    model_precision = (
        "fp16"
        if legacy_half is True
        else "fp32"
        if legacy_half is False
        else str(model_raw.get("precision", "fp32")).strip().lower()
    )

    try:
        config = AppConfig(
            model=ModelConfig(
                path=str(model_raw.get("path", "yolo11n.pt")),
                confidence=_float(
                    model_raw.get("confidence", 0.35), "model.confidence"
                ),
                detector_floor=_optional_float(
                    model_raw.get("detector_floor"), "model.detector_floor"
                ),
                imgsz=_int(model_raw.get("imgsz", 640), "model.imgsz"),
                precision=model_precision,
                half=model_precision == "fp16",
                legacy_half_configured=legacy_half is not None,
            ),
            tracking=TrackingConfig(
                tracker=str(tracking_raw.get("tracker", "bytetrack.yaml")),
                minimum_zone_frames=_int(
                    tracking_raw.get("minimum_zone_frames", 3),
                    "tracking.minimum_zone_frames",
                ),
                minimum_origin_zone_frames=_optional_int(
                    tracking_raw.get("minimum_origin_zone_frames"),
                    "tracking.minimum_origin_zone_frames",
                ),
                minimum_destination_zone_frames=_optional_int(
                    tracking_raw.get("minimum_destination_zone_frames"),
                    "tracking.minimum_destination_zone_frames",
                ),
                maximum_transition_gap_frames=_int(
                    tracking_raw.get("maximum_transition_gap_frames", 15),
                    "tracking.maximum_transition_gap_frames",
                ),
                event_cooldown_frames=_int(
                    tracking_raw.get("event_cooldown_frames", 30),
                    "tracking.event_cooldown_frames",
                ),
                zone_anchor=str(
                    tracking_raw.get("zone_anchor", "bottom_center")
                ).strip(),
                zone_boundary_hysteresis=_float(
                    tracking_raw.get("zone_boundary_hysteresis", 0.0),
                    "tracking.zone_boundary_hysteresis",
                ),
                stale_track_timeout=_int(
                    tracking_raw.get("stale_track_timeout", 90),
                    "tracking.stale_track_timeout",
                ),
                track_high_thresh=_optional_float(
                    tracking_raw.get("track_high_thresh"),
                    "tracking.track_high_thresh",
                ),
                track_low_thresh=_optional_float(
                    tracking_raw.get("track_low_thresh"),
                    "tracking.track_low_thresh",
                ),
                new_track_thresh=_optional_float(
                    tracking_raw.get("new_track_thresh"),
                    "tracking.new_track_thresh",
                ),
                track_buffer=_optional_int(
                    tracking_raw.get("track_buffer"), "tracking.track_buffer"
                ),
                match_thresh=_optional_float(
                    tracking_raw.get("match_thresh"), "tracking.match_thresh"
                ),
                fuse_score=_optional_bool(
                    tracking_raw.get("fuse_score"), "tracking.fuse_score"
                ),
                continuity=ContinuityConfig(
                    enabled=_bool(
                        continuity_raw.get("enabled", False),
                        "tracking.continuity.enabled",
                    ),
                    max_gap_frames=_int(
                        continuity_raw.get("max_gap_frames", 3),
                        "tracking.continuity.max_gap_frames",
                    ),
                    max_centroid_distance=_float(
                        continuity_raw.get("max_centroid_distance", 0.08),
                        "tracking.continuity.max_centroid_distance",
                    ),
                    minimum_iou=_float(
                        continuity_raw.get("minimum_iou", 0.05),
                        "tracking.continuity.minimum_iou",
                    ),
                    maximum_size_ratio=_float(
                        continuity_raw.get("maximum_size_ratio", 1.8),
                        "tracking.continuity.maximum_size_ratio",
                    ),
                    minimum_direction_cosine=_float(
                        continuity_raw.get("minimum_direction_cosine", -0.25),
                        "tracking.continuity.minimum_direction_cosine",
                    ),
                    minimum_match_score=_float(
                        continuity_raw.get("minimum_match_score", 0.55),
                        "tracking.continuity.minimum_match_score",
                    ),
                ),
            ),
            zones=ZonesConfig(
                outside=_polygon(zones_raw.get("outside"), "zones.outside"),
                inside=_polygon(zones_raw.get("inside"), "zones.inside"),
            ),
            outputs=OutputConfig(
                video=Path(outputs_raw.get("video", "data/output/result.mp4")),
                events_csv=_optional_path(
                    outputs_raw.get("events_csv"), "outputs.events_csv"
                ),
                summary_json=_optional_path(
                    outputs_raw.get("summary_json"), "outputs.summary_json"
                ),
                raw_detections_csv=_optional_path(
                    outputs_raw.get("raw_detections_csv"),
                    "outputs.raw_detections_csv",
                ),
                fused_detections_csv=_optional_path(
                    outputs_raw.get("fused_detections_csv"),
                    "outputs.fused_detections_csv",
                ),
                tracks_csv=_optional_path(
                    outputs_raw.get("tracks_csv"), "outputs.tracks_csv"
                ),
            ),
            camera=CameraConfig(
                detection_roi=_optional_roi(
                    camera_raw.get("detection_roi"), "camera.detection_roi"
                ),
                inference_views=inference_views,
                left_lane=_optional_polygon(
                    lanes_raw.get("left_lane"), "camera.doorway_lanes.left_lane"
                ),
                center_lane=_optional_polygon(
                    lanes_raw.get("center_lane"), "camera.doorway_lanes.center_lane"
                ),
                right_lane=_optional_polygon(
                    lanes_raw.get("right_lane"), "camera.doorway_lanes.right_lane"
                ),
                lane_anchor=str(lanes_raw.get("anchor", "center")),
                crossing_corridor=_optional_polygon(
                    camera_raw.get("crossing_corridor"),
                    "camera.crossing_corridor",
                ),
                exclusion_polygons=_polygon_collection(
                    camera_raw.get("exclusion_polygons", []),
                    "camera.exclusion_polygons",
                ),
                debug_calibration_overlay=_bool(
                    camera_raw.get("debug_calibration_overlay", False),
                    "camera.debug_calibration_overlay",
                ),
            ),
            diagnostics=DiagnosticsConfig(
                nearby_distance_normalized=_float(
                    diagnostics_raw.get("nearby_distance_normalized", 0.12),
                    "diagnostics.nearby_distance_normalized",
                ),
                heavy_overlap_iou=_float(
                    diagnostics_raw.get("heavy_overlap_iou", 0.50),
                    "diagnostics.heavy_overlap_iou",
                ),
                id_restart_window_frames=_int(
                    diagnostics_raw.get("id_restart_window_frames", 15),
                    "diagnostics.id_restart_window_frames",
                ),
                id_restart_distance_normalized=_float(
                    diagnostics_raw.get("id_restart_distance_normalized", 0.08),
                    "diagnostics.id_restart_distance_normalized",
                ),
                edge_margin_pixels=_int(
                    diagnostics_raw.get("edge_margin_pixels", 2),
                    "diagnostics.edge_margin_pixels",
                ),
                stationary_step_threshold=_float(
                    diagnostics_raw.get("stationary_step_threshold", 0.005),
                    "diagnostics.stationary_step_threshold",
                ),
                likely_static_minimum_frames=_int(
                    diagnostics_raw.get("likely_static_minimum_frames", 75),
                    "diagnostics.likely_static_minimum_frames",
                ),
                likely_static_minimum_percentage=_float(
                    diagnostics_raw.get("likely_static_minimum_percentage", 0.90),
                    "diagnostics.likely_static_minimum_percentage",
                ),
                wide_box_aspect_ratio=_float(
                    diagnostics_raw.get("wide_box_aspect_ratio", 0.90),
                    "diagnostics.wide_box_aspect_ratio",
                ),
                wide_box_doorway_ratio=_float(
                    diagnostics_raw.get("wide_box_doorway_ratio", 0.45),
                    "diagnostics.wide_box_doorway_ratio",
                ),
                multi_lane_minimum_box_overlap=_float(
                    diagnostics_raw.get("multi_lane_minimum_box_overlap", 0.05),
                    "diagnostics.multi_lane_minimum_box_overlap",
                ),
                export_detection_csv=_bool(
                    diagnostics_raw.get("export_detection_csv", False),
                    "diagnostics.export_detection_csv",
                ),
                debug_visualization=_bool(
                    diagnostics_raw.get("debug_visualization", False),
                    "diagnostics.debug_visualization",
                ),
                trajectory_length=_int(
                    diagnostics_raw.get("trajectory_length", 30),
                    "diagnostics.trajectory_length",
                ),
                failure_mining=FailureMiningConfig(
                    enabled=_bool(
                        failure_mining_raw.get("enabled", False),
                        "diagnostics.failure_mining.enabled",
                    ),
                    output_jsonl=_optional_path(
                        failure_mining_raw.get("output_jsonl"),
                        "diagnostics.failure_mining.output_jsonl",
                    ),
                    capture_frames=_bool(
                        failure_mining_raw.get("capture_frames", False),
                        "diagnostics.failure_mining.capture_frames",
                    ),
                    frames_directory=_optional_path(
                        failure_mining_raw.get("frames_directory"),
                        "diagnostics.failure_mining.frames_directory",
                    ),
                    maximum_records=_int(
                        failure_mining_raw.get("maximum_records", 500),
                        "diagnostics.failure_mining.maximum_records",
                    ),
                    maximum_captured_frames=_int(
                        failure_mining_raw.get("maximum_captured_frames", 50),
                        "diagnostics.failure_mining.maximum_captured_frames",
                    ),
                    jpeg_quality=_int(
                        failure_mining_raw.get("jpeg_quality", 85),
                        "diagnostics.failure_mining.jpeg_quality",
                    ),
                    low_confidence_threshold=_optional_float(
                        failure_mining_raw.get("low_confidence_threshold"),
                        "diagnostics.failure_mining.low_confidence_threshold",
                    ),
                    short_track_maximum_frames=_int(
                        failure_mining_raw.get("short_track_maximum_frames", 5),
                        "diagnostics.failure_mining.short_track_maximum_frames",
                    ),
                    sudden_detection_drop_ratio=_float(
                        failure_mining_raw.get("sudden_detection_drop_ratio", 0.50),
                        "diagnostics.failure_mining.sudden_detection_drop_ratio",
                    ),
                ),
            ),
            detection_fusion=DetectionFusionConfig(
                method=str(fusion_raw.get("method", "nms")),
                iou_threshold=_float(
                    fusion_raw.get("iou_threshold", 0.50),
                    "detection_fusion.iou_threshold",
                ),
                confidence_strategy=str(
                    fusion_raw.get("confidence_strategy", "maximum")
                ),
                prefer_full_frame=_bool(
                    fusion_raw.get("prefer_full_frame", False),
                    "detection_fusion.prefer_full_frame",
                ),
            ),
            capacity=_int(root.get("capacity", 40), "capacity"),
            initial_occupancy=_int(
                root.get("initial_occupancy", 0), "initial_occupancy"
            ),
            device=str(root.get("device", "auto")),
            logging_level=str(root.get("logging_level", "INFO")).upper(),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Configuration contains an invalid value: {exc}") from exc

    validate_config(config)
    for message in configuration_warnings(config):
        warnings.warn(message, UserWarning, stacklevel=2)
    return config


def validate_config(config: AppConfig) -> None:
    """Raise :class:`ConfigError` if a configuration is unsafe or inconsistent."""

    if not config.model.path.strip():
        raise ConfigError("'model.path' cannot be empty.")
    if not 0.0 < config.model.confidence <= 1.0:
        raise ConfigError("'model.confidence' must be greater than 0 and at most 1.")
    if not 0.0 < config.model.effective_detector_floor <= 1.0:
        raise ConfigError(
            "'model.detector_floor' must be greater than 0 and at most 1."
        )
    if config.model.effective_detector_floor > config.model.confidence:
        raise ConfigError(
            "'model.detector_floor' cannot exceed 'model.confidence'; the floor "
            "must preserve the configured high-confidence band."
        )
    if config.model.imgsz < 32:
        raise ConfigError("'model.imgsz' must be at least 32 pixels.")
    if config.model.precision not in {"fp32", "fp16"}:
        raise ConfigError("'model.precision' must be fp32 or fp16.")
    if config.camera.detection_roi is not None and config.camera.inference_views:
        raise ConfigError(
            "'camera.detection_roi' is deprecated and cannot be combined with "
            "'camera.inference_views'; migrate the ROI to one named view."
        )
    view_names = [view.name for view in config.camera.inference_views]
    if len(view_names) != len(set(view_names)):
        raise ConfigError("'camera.inference_views' names must be unique.")
    if config.camera.inference_views and not any(
        view.enabled for view in config.camera.inference_views
    ):
        raise ConfigError("'camera.inference_views' must enable at least one view.")
    if not config.tracking.tracker.strip():
        raise ConfigError("'tracking.tracker' cannot be empty.")
    if config.tracking.minimum_zone_frames < 1:
        raise ConfigError("'tracking.minimum_zone_frames' must be at least 1.")
    if config.tracking.effective_minimum_origin_zone_frames < 1:
        raise ConfigError("'tracking.minimum_origin_zone_frames' must be at least 1.")
    if config.tracking.effective_minimum_destination_zone_frames < 1:
        raise ConfigError(
            "'tracking.minimum_destination_zone_frames' must be at least 1."
        )
    if config.tracking.maximum_transition_gap_frames < 0:
        raise ConfigError(
            "'tracking.maximum_transition_gap_frames' cannot be negative."
        )
    if config.tracking.event_cooldown_frames < 0:
        raise ConfigError("'tracking.event_cooldown_frames' cannot be negative.")
    if config.tracking.zone_anchor not in {
        "center",
        "bottom_center",
        "top_center",
    }:
        raise ConfigError(
            "'tracking.zone_anchor' must be center, bottom_center, or top_center."
        )
    if not 0.0 <= config.tracking.zone_boundary_hysteresis <= 0.10:
        raise ConfigError(
            "'tracking.zone_boundary_hysteresis' must be between 0 and 0.10."
        )
    if config.tracking.stale_track_timeout < 1:
        raise ConfigError("'tracking.stale_track_timeout' must be at least 1.")
    for name, value in (
        ("track_high_thresh", config.tracking.track_high_thresh),
        ("track_low_thresh", config.tracking.track_low_thresh),
        ("new_track_thresh", config.tracking.new_track_thresh),
        ("match_thresh", config.tracking.match_thresh),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ConfigError(f"'tracking.{name}' must be between 0 and 1.")
    if config.tracking.track_buffer is not None and config.tracking.track_buffer < 1:
        raise ConfigError("'tracking.track_buffer' must be at least 1.")
    continuity = config.tracking.continuity
    if continuity.max_gap_frames < 1 or continuity.max_gap_frames > 30:
        raise ConfigError(
            "'tracking.continuity.max_gap_frames' must be between 1 and 30."
        )
    if not 0.0 < continuity.max_centroid_distance <= 0.50:
        raise ConfigError(
            "'tracking.continuity.max_centroid_distance' must be greater than 0 "
            "and at most 0.50."
        )
    if not 0.0 <= continuity.minimum_iou <= 1.0:
        raise ConfigError("'tracking.continuity.minimum_iou' must be between 0 and 1.")
    if not 1.0 <= continuity.maximum_size_ratio <= 10.0:
        raise ConfigError(
            "'tracking.continuity.maximum_size_ratio' must be between 1 and 10."
        )
    if not -1.0 <= continuity.minimum_direction_cosine <= 1.0:
        raise ConfigError(
            "'tracking.continuity.minimum_direction_cosine' must be between -1 and 1."
        )
    if not 0.0 <= continuity.minimum_match_score <= 1.0:
        raise ConfigError(
            "'tracking.continuity.minimum_match_score' must be between 0 and 1."
        )
    fusion = config.detection_fusion
    if fusion.method != "nms":
        raise ConfigError("'detection_fusion.method' must be 'nms'.")
    if not 0.0 < fusion.iou_threshold <= 1.0:
        raise ConfigError(
            "'detection_fusion.iou_threshold' must be greater than 0 and at most 1."
        )
    if fusion.confidence_strategy != "maximum":
        raise ConfigError("'detection_fusion.confidence_strategy' must be 'maximum'.")
    if config.camera.lane_anchor not in {"center", "bottom_center"}:
        raise ConfigError(
            "'camera.doorway_lanes.anchor' must be center or bottom_center."
        )
    diagnostics = config.diagnostics
    for name, value in (
        ("heavy_overlap_iou", diagnostics.heavy_overlap_iou),
        ("nearby_distance_normalized", diagnostics.nearby_distance_normalized),
        ("id_restart_distance_normalized", diagnostics.id_restart_distance_normalized),
        ("stationary_step_threshold", diagnostics.stationary_step_threshold),
        (
            "likely_static_minimum_percentage",
            diagnostics.likely_static_minimum_percentage,
        ),
        ("wide_box_aspect_ratio", diagnostics.wide_box_aspect_ratio),
        ("wide_box_doorway_ratio", diagnostics.wide_box_doorway_ratio),
        ("multi_lane_minimum_box_overlap", diagnostics.multi_lane_minimum_box_overlap),
    ):
        if value < 0:
            raise ConfigError(f"'diagnostics.{name}' cannot be negative.")
    for name, value in (
        ("heavy_overlap_iou", diagnostics.heavy_overlap_iou),
        (
            "likely_static_minimum_percentage",
            diagnostics.likely_static_minimum_percentage,
        ),
        ("multi_lane_minimum_box_overlap", diagnostics.multi_lane_minimum_box_overlap),
    ):
        if value > 1.0:
            raise ConfigError(f"'diagnostics.{name}' must be at most 1.0.")
    if diagnostics.id_restart_window_frames < 1:
        raise ConfigError("'diagnostics.id_restart_window_frames' must be at least 1.")
    if diagnostics.edge_margin_pixels < 0:
        raise ConfigError("'diagnostics.edge_margin_pixels' cannot be negative.")
    if diagnostics.likely_static_minimum_frames < 1:
        raise ConfigError(
            "'diagnostics.likely_static_minimum_frames' must be at least 1."
        )
    if diagnostics.trajectory_length < 1:
        raise ConfigError("'diagnostics.trajectory_length' must be at least 1.")
    failure_mining = diagnostics.failure_mining
    if failure_mining.capture_frames and not failure_mining.enabled:
        raise ConfigError(
            "'diagnostics.failure_mining.capture_frames' requires failure mining "
            "to be enabled."
        )
    if not 1 <= failure_mining.maximum_records <= 10_000:
        raise ConfigError(
            "'diagnostics.failure_mining.maximum_records' must be between 1 and 10000."
        )
    if not 0 <= failure_mining.maximum_captured_frames <= 500:
        raise ConfigError(
            "'diagnostics.failure_mining.maximum_captured_frames' must be between "
            "0 and 500."
        )
    if failure_mining.capture_frames and failure_mining.maximum_captured_frames == 0:
        raise ConfigError(
            "'diagnostics.failure_mining.maximum_captured_frames' must be positive "
            "when frame capture is enabled."
        )
    if not 40 <= failure_mining.jpeg_quality <= 100:
        raise ConfigError(
            "'diagnostics.failure_mining.jpeg_quality' must be between 40 and 100."
        )
    if (
        failure_mining.low_confidence_threshold is not None
        and not 0.0 < failure_mining.low_confidence_threshold <= 1.0
    ):
        raise ConfigError(
            "'diagnostics.failure_mining.low_confidence_threshold' must be greater "
            "than 0 and at most 1."
        )
    if not 1 <= failure_mining.short_track_maximum_frames <= 30:
        raise ConfigError(
            "'diagnostics.failure_mining.short_track_maximum_frames' must be "
            "between 1 and 30."
        )
    if not 0.0 <= failure_mining.sudden_detection_drop_ratio <= 1.0:
        raise ConfigError(
            "'diagnostics.failure_mining.sudden_detection_drop_ratio' must be "
            "between 0 and 1."
        )
    if config.capacity < 1:
        raise ConfigError("'capacity' must be at least 1.")
    if config.initial_occupancy < 0:
        raise ConfigError("'initial_occupancy' cannot be negative.")
    if not config.outputs.video.suffix:
        raise ConfigError("'outputs.video' must include a file extension.")
    if config.logging_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(
            "'logging_level' must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )


def resolve_inference_views(camera: CameraConfig) -> tuple[InferenceViewConfig, ...]:
    """Resolve v0.2 views while preserving the deprecated v0.1.2 ROI shape."""

    if camera.inference_views:
        return camera.inference_views
    if camera.detection_roi is not None:
        return (
            InferenceViewConfig(
                name="legacy_detection_roi",
                bounds=camera.detection_roi,
                enabled=True,
            ),
        )
    return (
        InferenceViewConfig(
            name="full",
            bounds=(0.0, 0.0, 1.0, 1.0),
            enabled=True,
        ),
    )


def configuration_warnings(config: AppConfig) -> tuple[str, ...]:
    """Return actionable, deterministic calibration warnings."""

    messages: list[str] = []
    if config.model.legacy_half_configured:
        messages.append(
            "model.half is deprecated in TrackBus configuration; use "
            f"model.precision: {config.model.precision} instead. The setting was "
            "converted once at startup."
        )
    if config.camera.detection_roi is not None:
        messages.append(
            "camera.detection_roi is deprecated; it is treated as the sole "
            "'legacy_detection_roi' inference view. Migrate to camera.inference_views."
        )

    enabled = tuple(
        view for view in resolve_inference_views(config.camera) if view.enabled
    )
    if len(enabled) == 1:
        view = enabled[0]
        outside_coverage = _view_zone_coverage(view.bounds, config.zones.outside)
        inside_coverage = _view_zone_coverage(view.bounds, config.zones.inside)
        minimum_coverage = 0.50
        if min(outside_coverage, inside_coverage) < minimum_coverage:
            messages.append(
                f"sole inference view '{view.name}' covers only "
                f"{outside_coverage:.0%} of OUTSIDE and {inside_coverage:.0%} of "
                "INSIDE; a sole crop should cover at least 50% of both zones or "
                "complete crossing history may be lost."
            )

    passenger_polygons = (
        *(
            polygon
            for polygon in (
                config.camera.left_lane,
                config.camera.center_lane,
                config.camera.right_lane,
                config.camera.crossing_corridor,
            )
            if polygon is not None
        ),
        config.zones.outside,
        config.zones.inside,
    )
    for index, exclusion in enumerate(config.camera.exclusion_polygons):
        if any(_polygons_overlap(exclusion, pathway) for pathway in passenger_polygons):
            messages.append(
                f"exclusion polygon {index} overlaps a configured lane or counting "
                "zone; verify it covers only static structure and no passenger pathway."
            )
    return tuple(messages)


def _view_zone_coverage(bounds: NormalizedRoi, zone: NormalizedPolygon) -> float:
    zone_area = _polygon_area(zone)
    if zone_area <= 0:
        return 0.0
    return _polygon_area(_clip_polygon_to_rect(zone, bounds)) / zone_area


def _polygons_overlap(first: NormalizedPolygon, second: NormalizedPolygon) -> bool:
    # Configuration polygons are usually rectangles. Checking vertices plus edge
    # intersections covers arbitrary simple polygons without adding a geometry
    # dependency.
    if any(_point_in_polygon(point, second) for point in first):
        return True
    if any(_point_in_polygon(point, first) for point in second):
        return True
    first_edges = tuple(zip(first, first[1:] + first[:1], strict=True))
    second_edges = tuple(zip(second, second[1:] + second[:1], strict=True))
    return any(
        _segments_intersect(a, b, c, d) for a, b in first_edges for c, d in second_edges
    )


def _point_in_polygon(point: NormalizedPoint, polygon: NormalizedPolygon) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= crossing_x:
                inside = not inside
        previous = current
    return inside


def _segments_intersect(
    a: NormalizedPoint,
    b: NormalizedPoint,
    c: NormalizedPoint,
    d: NormalizedPoint,
) -> bool:
    def orientation(
        first: NormalizedPoint, second: NormalizedPoint, third: NormalizedPoint
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    def on_segment(
        first: NormalizedPoint,
        point: NormalizedPoint,
        second: NormalizedPoint,
    ) -> bool:
        epsilon = 1e-12
        return (
            min(first[0], second[0]) - epsilon
            <= point[0]
            <= max(first[0], second[0]) + epsilon
            and min(first[1], second[1]) - epsilon
            <= point[1]
            <= max(first[1], second[1]) + epsilon
        )

    epsilon = 1e-12
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if (o1 > epsilon and o2 < -epsilon or o1 < -epsilon and o2 > epsilon) and (
        o3 > epsilon and o4 < -epsilon or o3 < -epsilon and o4 > epsilon
    ):
        return True
    return (
        abs(o1) <= epsilon
        and on_segment(a, c, b)
        or abs(o2) <= epsilon
        and on_segment(a, d, b)
        or abs(o3) <= epsilon
        and on_segment(c, a, d)
        or abs(o4) <= epsilon
        and on_segment(c, b, d)
    )


def _polygon_area(points: tuple[NormalizedPoint, ...] | list[NormalizedPoint]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            sum(
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )
        )
        / 2.0
    )


def _clip_polygon_to_rect(
    polygon: NormalizedPolygon, bounds: NormalizedRoi
) -> list[NormalizedPoint]:
    left, top, right, bottom = bounds
    output = list(polygon)

    def clip(
        points: list[NormalizedPoint],
        inside: Any,
        intersection: Any,
    ) -> list[NormalizedPoint]:
        if not points:
            return []
        result: list[NormalizedPoint] = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    result.append(intersection(previous, current))
                result.append(current)
            elif previous_inside:
                result.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return result

    def at_x(
        first: NormalizedPoint, second: NormalizedPoint, x: float
    ) -> NormalizedPoint:
        delta = second[0] - first[0]
        ratio = 0.0 if delta == 0 else (x - first[0]) / delta
        return x, first[1] + ratio * (second[1] - first[1])

    def at_y(
        first: NormalizedPoint, second: NormalizedPoint, y: float
    ) -> NormalizedPoint:
        delta = second[1] - first[1]
        ratio = 0.0 if delta == 0 else (y - first[1]) / delta
        return first[0] + ratio * (second[0] - first[0]), y

    output = clip(output, lambda point: point[0] >= left, lambda a, b: at_x(a, b, left))
    output = clip(
        output, lambda point: point[0] <= right, lambda a, b: at_x(a, b, right)
    )
    output = clip(output, lambda point: point[1] >= top, lambda a, b: at_y(a, b, top))
    return clip(
        output, lambda point: point[1] <= bottom, lambda a, b: at_y(a, b, bottom)
    )
