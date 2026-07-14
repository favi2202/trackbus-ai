"""Configuration loading and validation for TrackBus."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""


NormalizedPoint = tuple[float, float]


@dataclass(frozen=True)
class ModelConfig:
    path: str = "yolo11n.pt"
    confidence: float = 0.35


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    minimum_zone_frames: int = 3
    stale_track_timeout: int = 90


@dataclass(frozen=True)
class ZonesConfig:
    outside: tuple[NormalizedPoint, ...]
    inside: tuple[NormalizedPoint, ...]


@dataclass(frozen=True)
class OutputConfig:
    video: Path = Path("data/output/result.mp4")
    events_csv: Path | None = None
    summary_json: Path | None = None


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings for a TrackBus processing run."""

    model: ModelConfig
    tracking: TrackingConfig
    zones: ZonesConfig
    outputs: OutputConfig
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
        model: str | None = None,
    ) -> AppConfig:
        """Return a validated copy with command-line values applied."""

        updated_model = replace(
            self.model,
            path=model if model is not None else self.model.path,
            confidence=(
                confidence if confidence is not None else self.model.confidence
            ),
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


def _polygon(value: Any, name: str) -> tuple[NormalizedPoint, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ConfigError(f"'{name}' must contain at least three [x, y] points.")

    points: list[NormalizedPoint] = []
    for index, raw_point in enumerate(value):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise ConfigError(f"'{name}[{index}]' must be an [x, y] point.")
        try:
            point = (float(raw_point[0]), float(raw_point[1]))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"'{name}[{index}]' coordinates must be numbers."
            ) from exc
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


def load_config(path: Path) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file."""

    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read configuration file '{path}': {exc}") from exc

    root = _mapping(raw, "configuration")
    model_raw = _mapping(root.get("model", {}), "model")
    tracking_raw = _mapping(root.get("tracking", {}), "tracking")
    zones_raw = _mapping(root.get("zones"), "zones")
    outputs_raw = _mapping(root.get("outputs", {}), "outputs")

    try:
        config = AppConfig(
            model=ModelConfig(
                path=str(model_raw.get("path", "yolo11n.pt")),
                confidence=float(model_raw.get("confidence", 0.35)),
            ),
            tracking=TrackingConfig(
                tracker=str(tracking_raw.get("tracker", "bytetrack.yaml")),
                minimum_zone_frames=int(tracking_raw.get("minimum_zone_frames", 3)),
                stale_track_timeout=int(tracking_raw.get("stale_track_timeout", 90)),
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
            ),
            capacity=int(root.get("capacity", 40)),
            initial_occupancy=int(root.get("initial_occupancy", 0)),
            device=str(root.get("device", "auto")),
            logging_level=str(root.get("logging_level", "INFO")).upper(),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Configuration contains an invalid value: {exc}") from exc

    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Raise :class:`ConfigError` if a configuration is unsafe or inconsistent."""

    if not config.model.path.strip():
        raise ConfigError("'model.path' cannot be empty.")
    if not 0.0 < config.model.confidence <= 1.0:
        raise ConfigError("'model.confidence' must be greater than 0 and at most 1.")
    if not config.tracking.tracker.strip():
        raise ConfigError("'tracking.tracker' cannot be empty.")
    if config.tracking.minimum_zone_frames < 1:
        raise ConfigError("'tracking.minimum_zone_frames' must be at least 1.")
    if config.tracking.stale_track_timeout < 1:
        raise ConfigError("'tracking.stale_track_timeout' must be at least 1.")
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
