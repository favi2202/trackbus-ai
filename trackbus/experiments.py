"""Reproducible, bounded experiment matrices for TrackBus.

The module deliberately executes the public ``trackbus.main`` command by default.
Tests and downstream applications can inject an :class:`ExperimentExecutor`, so
ordinary unit tests never require model weights or video inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

if TYPE_CHECKING:
    from trackbus.annotate_events import GroundTruthAnnotations

LOGGER = logging.getLogger("trackbus.experiments")
MATRIX_VERSION = 1
MAX_EXPERIMENT_RUNS = 50
_SAFE_NAME = re.compile(r"[^a-z0-9]+")
_STANDARD_YOLO_WEIGHT = re.compile(r"^yolo(?:v?\d+)[a-z0-9_-]*\.pt$", re.IGNORECASE)

_MATRIX_KEYS = {
    "version",
    "base_config",
    "aggregate_ground_truth",
    "evaluation",
    "views",
    "runs",
}
_RUN_KEYS = {
    "name",
    "description",
    "model",
    "confidence",
    "image_size",
    "tracker",
    "enabled_views",
    "detection_fusion",
    "overrides",
}
_TRACKER_KEYS = {
    "config",
    "minimum_zone_frames",
    "stale_track_timeout",
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_buffer",
    "match_thresh",
    "fuse_score",
}
_FUSION_KEYS = {
    "method",
    "iou_threshold",
    "confidence_strategy",
    "prefer_full_frame",
}
_APP_CONFIG_KEYS: dict[str, set[str] | None] = {
    "model": {"path", "confidence", "imgsz", "half"},
    "tracking": _TRACKER_KEYS - {"config"} | {"tracker"},
    "zones": {"outside", "inside"},
    "outputs": {
        "video",
        "events_csv",
        "summary_json",
        "raw_detections_csv",
        "fused_detections_csv",
        "tracks_csv",
    },
    "camera": {
        "detection_roi",
        "inference_views",
        "doorway_lanes",
        "exclusion_polygons",
        "debug_calibration_overlay",
    },
    "diagnostics": {
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
    },
    "detection_fusion": _FUSION_KEYS,
    "capacity": None,
    "initial_occupancy": None,
    "device": None,
    "logging_level": None,
}


class ExperimentConfigError(ValueError):
    """Raised when an experiment matrix is malformed or unsafe."""


class ExperimentExecutionError(RuntimeError):
    """Raised when one experiment cannot produce a usable run summary."""


@dataclass(frozen=True)
class InferenceViewSpec:
    """A named normalized crop available to experiment runs."""

    name: str
    bounds: tuple[float, float, float, float]

    def to_config(self) -> dict[str, Any]:
        return {"name": self.name, "bounds": list(self.bounds), "enabled": True}


@dataclass(frozen=True)
class ExperimentRun:
    """One explicit experiment configuration from a matrix."""

    name: str
    description: str
    model: str
    confidence: float
    image_size: int
    tracker: Mapping[str, Any]
    enabled_views: tuple[str, ...]
    detection_fusion: Mapping[str, Any]
    overrides: Mapping[str, Any]

    @property
    def slug(self) -> str:
        slug = _SAFE_NAME.sub("_", self.name.casefold()).strip("_")
        return slug or "experiment"

    @property
    def inference_mode(self) -> str:
        return "full_frame" if self.enabled_views == ("full",) else "multiview"

    def concise_configuration(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "tracker": deepcopy(dict(self.tracker)),
            "enabled_views": list(self.enabled_views),
            "detection_fusion": deepcopy(dict(self.detection_fusion)),
            "overrides": deepcopy(dict(self.overrides)),
        }


@dataclass(frozen=True)
class ExperimentMatrix:
    """Validated experiment matrix and its source context."""

    source_path: Path
    base_config_path: Path
    views: Mapping[str, InferenceViewSpec]
    runs: tuple[ExperimentRun, ...]
    expected_entered: int
    expected_exited: int
    tolerance_frames: int | None
    tolerance_seconds: float | None


@dataclass(frozen=True)
class ExperimentArtifacts:
    """All deterministic artifact paths belonging to one run."""

    run_dir: Path
    resolved_config: Path
    output_video: Path
    events_csv: Path
    summary_json: Path
    raw_detections_csv: Path
    fused_detections_csv: Path
    tracks_csv: Path
    process_log: Path
    result_json: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "run_directory": str(self.run_dir),
            "resolved_config": str(self.resolved_config),
            "output_video": str(self.output_video),
            "events_csv": str(self.events_csv),
            "summary_json": str(self.summary_json),
            "raw_detections_csv": str(self.raw_detections_csv),
            "fused_detections_csv": str(self.fused_detections_csv),
            "tracks_csv": str(self.tracks_csv),
            "process_log": str(self.process_log),
            "result_json": str(self.result_json),
        }

    def existing_files(self) -> tuple[Path, ...]:
        """Return known run artifacts that would make a rerun ambiguous."""

        paths = (
            self.resolved_config,
            self.output_video,
            self.events_csv,
            self.summary_json,
            self.raw_detections_csv,
            self.fused_detections_csv,
            self.tracks_csv,
            self.process_log,
            self.result_json,
        )
        return tuple(path for path in paths if path.exists())


class ExperimentExecutor(Protocol):
    """Callable boundary used to process one resolved configuration."""

    def __call__(
        self,
        input_path: Path,
        config_path: Path,
        artifacts: ExperimentArtifacts,
    ) -> Mapping[str, Any]:
        """Process one run and return its machine-readable summary."""


def load_experiment_matrix(path: Path) -> ExperimentMatrix:
    """Load and validate an intentionally explicit YAML experiment matrix."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ExperimentConfigError(f"Experiment matrix does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(f"Could not read matrix '{path}': {exc}") from exc
    root = _require_mapping(raw, "matrix")
    _reject_unknown_keys(root, _MATRIX_KEYS, "matrix")

    version = root.get("version", MATRIX_VERSION)
    if version != MATRIX_VERSION:
        raise ExperimentConfigError(
            f"Unsupported experiment matrix version {version!r}; "
            f"expected {MATRIX_VERSION}."
        )

    base_value = root.get("base_config")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ExperimentConfigError("'base_config' must be a non-empty path.")
    base_config_path = (path.parent / base_value).resolve()
    if not base_config_path.is_file():
        raise ExperimentConfigError(
            f"Base configuration does not exist: {base_config_path}"
        )

    views_raw = _require_mapping(root.get("views"), "views")
    views: dict[str, InferenceViewSpec] = {}
    for name, value in views_raw.items():
        if isinstance(value, dict):
            _reject_unknown_keys(value, {"bounds"}, f"views.{name}")
        views[str(name)] = InferenceViewSpec(
            str(name), _normalized_bounds(value, str(name))
        )
    if not views:
        raise ExperimentConfigError("'views' must define at least one inference view.")

    runs_raw = root.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ExperimentConfigError("'runs' must be a non-empty YAML list.")
    if len(runs_raw) > MAX_EXPERIMENT_RUNS:
        raise ExperimentConfigError(
            f"Matrix has {len(runs_raw)} runs; the safety limit is "
            f"{MAX_EXPERIMENT_RUNS}. Split large sweeps into smaller matrices."
        )
    runs = tuple(
        _parse_run(item, index=index, known_views=views)
        for index, item in enumerate(runs_raw)
    )
    names = [run.name for run in runs]
    if len(names) != len(set(names)):
        raise ExperimentConfigError("Experiment run names must be unique.")
    slugs = [run.slug for run in runs]
    if len(slugs) != len(set(slugs)):
        raise ExperimentConfigError(
            "Experiment run names must remain unique after filesystem sanitization."
        )

    ground_truth = _require_mapping(
        root.get("aggregate_ground_truth"), "aggregate_ground_truth"
    )
    _reject_unknown_keys(ground_truth, {"entered", "exited"}, "aggregate_ground_truth")
    if "entered" not in ground_truth or "exited" not in ground_truth:
        raise ExperimentConfigError(
            "'aggregate_ground_truth' must explicitly define entered and exited; "
            "generic matrices must not inherit test-video totals."
        )
    expected_entered = _non_negative_int(
        ground_truth["entered"], "aggregate_ground_truth.entered"
    )
    expected_exited = _non_negative_int(
        ground_truth["exited"], "aggregate_ground_truth.exited"
    )
    evaluation = _require_mapping(root.get("evaluation", {}), "evaluation")
    _reject_unknown_keys(
        evaluation, {"tolerance_frames", "tolerance_seconds"}, "evaluation"
    )
    tolerance_frames = _optional_non_negative_int(
        evaluation.get("tolerance_frames", 25), "evaluation.tolerance_frames"
    )
    tolerance_seconds = _optional_non_negative_float(
        evaluation.get("tolerance_seconds"), "evaluation.tolerance_seconds"
    )
    if tolerance_frames is None and tolerance_seconds is None:
        raise ExperimentConfigError(
            "At least one event tolerance must be configured in frames or seconds."
        )
    if tolerance_frames is not None and tolerance_seconds is not None:
        raise ExperimentConfigError(
            "Configure event tolerance in frames or seconds, not both."
        )

    return ExperimentMatrix(
        source_path=path,
        base_config_path=base_config_path,
        views=views,
        runs=runs,
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        tolerance_frames=tolerance_frames,
        tolerance_seconds=tolerance_seconds,
    )


def _parse_run(
    value: object,
    *,
    index: int,
    known_views: Mapping[str, InferenceViewSpec],
) -> ExperimentRun:
    name_prefix = f"runs[{index}]"
    raw = _require_mapping(value, name_prefix)
    _reject_unknown_keys(raw, _RUN_KEYS, name_prefix)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ExperimentConfigError(f"'{name_prefix}.name' must be non-empty.")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ExperimentConfigError(f"'{name_prefix}.model' must be non-empty.")
    if "://" in model:
        raise ExperimentConfigError(
            f"'{name_prefix}.model' must be a standard Ultralytics weight name "
            "or a local path, not a remote URL."
        )
    confidence = _bounded_float(raw.get("confidence"), f"{name_prefix}.confidence")
    image_size = _positive_int(raw.get("image_size"), f"{name_prefix}.image_size")
    if image_size < 32:
        raise ExperimentConfigError(
            f"'{name_prefix}.image_size' must be at least 32 pixels."
        )

    tracker = _require_mapping(raw.get("tracker"), f"{name_prefix}.tracker")
    _reject_unknown_keys(tracker, _TRACKER_KEYS, f"{name_prefix}.tracker")
    tracker_config = tracker.get("config")
    if not isinstance(tracker_config, str) or not tracker_config.strip():
        raise ExperimentConfigError(
            f"'{name_prefix}.tracker.config' must be a non-empty path."
        )
    enabled_raw = raw.get("enabled_views")
    if not isinstance(enabled_raw, list) or not enabled_raw:
        raise ExperimentConfigError(
            f"'{name_prefix}.enabled_views' must be a non-empty list."
        )
    if not all(isinstance(item, str) and item for item in enabled_raw):
        raise ExperimentConfigError(
            f"'{name_prefix}.enabled_views' entries must be view names."
        )
    enabled_views = tuple(enabled_raw)
    if len(enabled_views) != len(set(enabled_views)):
        raise ExperimentConfigError(
            f"'{name_prefix}.enabled_views' cannot contain duplicates."
        )
    unknown = sorted(set(enabled_views) - known_views.keys())
    if unknown:
        raise ExperimentConfigError(
            f"'{name_prefix}.enabled_views' references unknown views: "
            f"{', '.join(unknown)}."
        )

    fusion = _require_mapping(
        raw.get("detection_fusion", {}), f"{name_prefix}.detection_fusion"
    )
    _reject_unknown_keys(fusion, _FUSION_KEYS, f"{name_prefix}.detection_fusion")
    method = fusion.get("method", "nms")
    if method != "nms":
        raise ExperimentConfigError(
            f"'{name_prefix}.detection_fusion.method' currently supports only 'nms'."
        )
    iou = _bounded_float(
        fusion.get("iou_threshold", 0.5),
        f"{name_prefix}.detection_fusion.iou_threshold",
    )
    confidence_strategy = fusion.get("confidence_strategy", "maximum")
    if confidence_strategy != "maximum":
        raise ExperimentConfigError(
            f"'{name_prefix}.detection_fusion.confidence_strategy' currently "
            "supports only 'maximum'."
        )
    prefer_full_frame = fusion.get("prefer_full_frame", False)
    if not isinstance(prefer_full_frame, bool):
        raise ExperimentConfigError(
            f"'{name_prefix}.detection_fusion.prefer_full_frame' must be boolean."
        )
    overrides = _require_mapping(raw.get("overrides", {}), f"{name_prefix}.overrides")
    _validate_application_config_keys(overrides, f"{name_prefix}.overrides")
    _reject_protected_overrides(overrides, name_prefix)

    return ExperimentRun(
        name=name.strip(),
        description=str(raw.get("description", "")).strip(),
        model=model.strip(),
        confidence=confidence,
        image_size=image_size,
        tracker=deepcopy(dict(tracker)),
        enabled_views=enabled_views,
        detection_fusion={
            "method": method,
            "iou_threshold": iou,
            "confidence_strategy": confidence_strategy,
            "prefer_full_frame": prefer_full_frame,
        },
        overrides=deepcopy(dict(overrides)),
    )


def resolve_run_configuration(
    matrix: ExperimentMatrix,
    run: ExperimentRun,
    artifacts: ExperimentArtifacts,
) -> dict[str, Any]:
    """Overlay one experiment on its base application configuration."""

    try:
        base = yaml.safe_load(matrix.base_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(
            f"Could not read base configuration '{matrix.base_config_path}': {exc}"
        ) from exc
    config = deepcopy(_require_mapping(base, "base configuration"))
    _validate_application_config_keys(config, "base configuration")
    tracker = dict(run.tracker)
    tracker_path = _resolve_contextual_path(
        str(tracker.pop("config")), matrix, kind="tracker configuration"
    )
    model_path = _resolve_contextual_path(run.model, matrix, kind="model")
    run_overrides: dict[str, Any] = {
        "model": {
            "path": model_path,
            "confidence": run.confidence,
            "imgsz": run.image_size,
        },
        "tracking": {"tracker": tracker_path, **tracker},
        "camera": {
            # Explicit views replace, rather than silently reinterpret, v0.1's ROI.
            "detection_roi": None,
            "inference_views": [
                matrix.views[name].to_config() for name in run.enabled_views
            ],
        },
        "detection_fusion": deepcopy(dict(run.detection_fusion)),
        "outputs": {
            "video": str(artifacts.output_video.resolve()),
            "events_csv": str(artifacts.events_csv.resolve()),
            "summary_json": str(artifacts.summary_json.resolve()),
            "raw_detections_csv": str(artifacts.raw_detections_csv.resolve()),
            "fused_detections_csv": str(artifacts.fused_detections_csv.resolve()),
            "tracks_csv": str(artifacts.tracks_csv.resolve()),
        },
    }
    _deep_merge(config, run.overrides)
    # Matrix dimensions and artifact paths are authoritative. Protected override
    # keys are rejected while parsing, and applying these values last is a second
    # safeguard against an accidental change to what the leaderboard records.
    _deep_merge(config, run_overrides)
    return config


def build_artifacts(output_dir: Path, run: ExperimentRun) -> ExperimentArtifacts:
    """Return stable per-run paths under ``output_dir``."""

    run_dir = output_dir.expanduser().resolve() / run.slug
    return ExperimentArtifacts(
        run_dir=run_dir,
        resolved_config=run_dir / "resolved_config.yaml",
        output_video=run_dir / "annotated.mp4",
        events_csv=run_dir / "events.csv",
        summary_json=run_dir / "summary.json",
        raw_detections_csv=run_dir / "raw_detections.csv",
        fused_detections_csv=run_dir / "fused_detections.csv",
        tracks_csv=run_dir / "tracks.csv",
        process_log=run_dir / "process.log",
        result_json=run_dir / "experiment_result.json",
    )


def subprocess_executor(
    input_path: Path,
    config_path: Path,
    artifacts: ExperimentArtifacts,
) -> Mapping[str, Any]:
    """Execute one run through the supported TrackBus command-line entry point."""

    command = [
        sys.executable,
        "-m",
        "trackbus.main",
        "--input",
        str(input_path.resolve()),
        "--config",
        str(config_path.resolve()),
        "--output",
        str(artifacts.output_video.resolve()),
    ]
    completed = subprocess.run(
        command,
        check=False,
        cwd=_repository_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    artifacts.process_log.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise ExperimentExecutionError(
            f"TrackBus exited with status {completed.returncode}; see "
            f"{artifacts.process_log}."
        )
    if not artifacts.summary_json.is_file():
        raise ExperimentExecutionError(
            f"TrackBus completed without writing {artifacts.summary_json}."
        )
    try:
        summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentExecutionError(
            f"Could not read run summary '{artifacts.summary_json}': {exc}"
        ) from exc
    if not isinstance(summary, dict):
        raise ExperimentExecutionError("Run summary must contain a JSON object.")
    return summary


def run_experiments(
    *,
    input_path: Path,
    matrix: ExperimentMatrix,
    output_dir: Path,
    ground_truth_path: Path | None = None,
    executor: ExperimentExecutor = subprocess_executor,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Execute a matrix and write per-run plus aggregate result artifacts."""

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise ExperimentConfigError(f"Input video does not exist: {input_path}")
    if ground_truth_path is not None:
        ground_truth_path = ground_truth_path.expanduser().resolve()
        if not ground_truth_path.is_file():
            raise ExperimentConfigError(
                f"Frame-level ground truth does not exist: {ground_truth_path}"
            )
    ground_truth = _validated_ground_truth(
        ground_truth_path,
        input_path=input_path,
        expected_entered=matrix.expected_entered,
        expected_exited=matrix.expected_exited,
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_stale_run_artifacts(output_dir, matrix.runs)
    reproducibility = _reproducibility_metadata(input_path)

    results: list[dict[str, Any]] = []
    for index, run in enumerate(matrix.runs, start=1):
        LOGGER.info("Running experiment %d/%d: %s", index, len(matrix.runs), run.name)
        artifacts = build_artifacts(output_dir, run)
        artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        resolved = resolve_run_configuration(matrix, run, artifacts)
        artifacts.resolved_config.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        try:
            summary = dict(executor(input_path, artifacts.resolved_config, artifacts))
            result = _successful_result(
                matrix=matrix,
                run=run,
                artifacts=artifacts,
                summary=summary,
                ground_truth=ground_truth,
                resolved_configuration=resolved,
                reproducibility=reproducibility,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            LOGGER.error("Experiment '%s' failed: %s", run.name, exc)
            result = _failed_result(run, artifacts, resolved, reproducibility, exc)
        artifacts.result_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    if ground_truth is None:
        expected_entered = matrix.expected_entered
        expected_exited = matrix.expected_exited
    else:
        expected_entered, expected_exited = _annotation_totals(ground_truth)
    write_leaderboard(
        results,
        output_dir,
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        has_frame_ground_truth=ground_truth_path is not None,
        reproducibility=reproducibility,
    )
    return rank_results(results, has_frame_ground_truth=ground_truth_path is not None)


def _successful_result(
    *,
    matrix: ExperimentMatrix,
    run: ExperimentRun,
    artifacts: ExperimentArtifacts,
    summary: Mapping[str, Any],
    ground_truth: GroundTruthAnnotations | None,
    resolved_configuration: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    entered = _required_summary_int(summary, "entered_total", "entered")
    exited = _required_summary_int(summary, "exited_total", "exited")
    event_evaluation: dict[str, Any] | None = None

    from trackbus.evaluation import evaluate_aggregate

    expected_entered, expected_exited = (
        (matrix.expected_entered, matrix.expected_exited)
        if ground_truth is None
        else _annotation_totals(ground_truth)
    )
    aggregate_evaluation = evaluate_aggregate(
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        predicted_entered=entered,
        predicted_exited=exited,
    ).to_dict()
    if ground_truth is not None:
        from trackbus.evaluation import EventRecord, evaluate_events, load_event_records

        source_fps = _required_summary_float(summary, "source_fps", positive=True)
        if not math.isclose(
            source_fps, ground_truth.source_fps, rel_tol=1e-4, abs_tol=1e-4
        ):
            raise ExperimentExecutionError(
                "Run source FPS does not match frame-level ground truth "
                f"({source_fps:g} != {ground_truth.source_fps:g})."
            )
        predictions = load_event_records(artifacts.events_csv, fps=source_fps)
        ground_truth_events = [
            EventRecord.from_annotated(event) for event in ground_truth.events
        ]
        event_evaluation = evaluate_events(
            ground_truth_events,
            predictions,
            tolerance_frames=matrix.tolerance_frames,
            tolerance_seconds=matrix.tolerance_seconds,
            fps=source_fps,
        ).to_dict()

    unique_ids = _required_summary_int(summary, "unique_tracking_ids")
    possible_restarts = _required_summary_int(
        summary, "possible_id_restart_count", "possible_id_restarts"
    )
    likely_static = _required_summary_int(summary, "likely_static_track_count")
    disappeared = _required_summary_int(
        summary, "tracks_disappearing_after_heavy_overlap"
    )
    expected_crossings = expected_entered + expected_exited
    fragmentation_risk = max(0, unique_ids - expected_crossings)
    false_event_risk = (
        possible_restarts + likely_static + disappeared + fragmentation_risk
    )

    event_metrics = _event_metric_fields(event_evaluation)
    result: dict[str, Any] = {
        "name": run.name,
        "description": run.description,
        "status": "completed",
        "inference_mode": run.inference_mode,
        "entered": entered,
        "exited": exited,
        "aggregate_count_error": abs(entered - expected_entered)
        + abs(exited - expected_exited),
        "total_crossing_count_error": abs(
            entered + exited - expected_entered - expected_exited
        ),
        **event_metrics,
        "person_detections": _required_summary_int(
            summary,
            "raw_person_detections_total",
            "raw_detections_total",
            "person_detections_total",
        ),
        "fused_detections": _required_summary_int(summary, "fused_detections_total"),
        "unique_tracking_ids": unique_ids,
        "possible_id_restarts": possible_restarts,
        "frames_without_detections": _required_summary_int(
            summary,
            "frames_without_raw_detections",
            "frames_without_person_detections",
            "frames_without_detections",
        ),
        "likely_static_tracks": likely_static,
        "tracks_disappearing_after_heavy_overlap": disappeared,
        "id_fragmentation_risk": fragmentation_risk,
        "false_event_risk_score": false_event_risk,
        "processing_fps": _required_summary_float(
            summary, "average_fps", "processing_fps", positive=True
        ),
        "processing_duration_seconds": _required_summary_float(
            summary,
            "processing_time_seconds",
            "processing_duration_seconds",
        ),
        "configuration": deepcopy(dict(resolved_configuration)),
        "requested_configuration": run.concise_configuration(),
        "configuration_sha256": _sha256_file(artifacts.resolved_config),
        "reproducibility": deepcopy(dict(reproducibility)),
        "output_paths": artifacts.as_dict(),
        "aggregate_evaluation": aggregate_evaluation,
        "event_evaluation": event_evaluation,
        "summary": deepcopy(dict(summary)),
        "error": None,
    }
    return result


def _failed_result(
    run: ExperimentRun,
    artifacts: ExperimentArtifacts,
    resolved_configuration: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": run.name,
        "description": run.description,
        "status": "failed",
        "inference_mode": run.inference_mode,
        "configuration": deepcopy(dict(resolved_configuration)),
        "requested_configuration": run.concise_configuration(),
        "configuration_sha256": _sha256_file(artifacts.resolved_config),
        "reproducibility": deepcopy(dict(reproducibility)),
        "output_paths": artifacts.as_dict(),
        "summary": None,
        "aggregate_evaluation": None,
        "event_evaluation": None,
        "error": f"{type(error).__name__}: {error}",
    }
    for field in _LEADERBOARD_METRIC_FIELDS:
        result[field] = None
    return result


def _event_metric_fields(evaluation: Mapping[str, Any] | None) -> dict[str, Any]:
    fields = {
        "event_true_positives": None,
        "event_false_positives": None,
        "event_false_negatives": None,
        "event_precision": None,
        "event_recall": None,
        "event_f1": None,
        "event_in_precision": None,
        "event_in_recall": None,
        "event_in_f1": None,
        "event_out_precision": None,
        "event_out_recall": None,
        "event_out_f1": None,
        "mean_timing_error": None,
        "mean_absolute_timing_error": None,
        "timing_error_unit": None,
    }
    if evaluation is None:
        return fields
    overall = evaluation.get("overall", evaluation)
    if not isinstance(overall, Mapping):
        overall = evaluation
    aliases = {
        "event_true_positives": ("true_positives", "true_positive_events", "tp"),
        "event_false_positives": (
            "false_positives",
            "false_positive_events",
            "fp",
        ),
        "event_false_negatives": (
            "false_negatives",
            "false_negative_events",
            "fn",
        ),
        "event_precision": ("precision",),
        "event_recall": ("recall",),
        "event_f1": ("f1",),
        "mean_timing_error": ("mean_timing_error", "mean_timing_error_frames"),
        "mean_absolute_timing_error": (
            "mean_absolute_timing_error_frames",
            "mean_absolute_timing_error",
        ),
        "timing_error_unit": ("timing_error_unit",),
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in overall:
                fields[target] = overall[candidate]
                break
            if candidate in evaluation:
                fields[target] = evaluation[candidate]
                break
    by_direction = evaluation.get("by_direction")
    if isinstance(by_direction, Mapping):
        for direction in ("IN", "OUT"):
            metrics = by_direction.get(direction)
            if not isinstance(metrics, Mapping):
                continue
            prefix = f"event_{direction.casefold()}"
            fields[f"{prefix}_precision"] = metrics.get("precision")
            fields[f"{prefix}_recall"] = metrics.get("recall")
            fields[f"{prefix}_f1"] = metrics.get("f1")
    return fields


_LEADERBOARD_METRIC_FIELDS = (
    "entered",
    "exited",
    "aggregate_count_error",
    "total_crossing_count_error",
    "event_true_positives",
    "event_false_positives",
    "event_false_negatives",
    "event_precision",
    "event_recall",
    "event_f1",
    "event_in_precision",
    "event_in_recall",
    "event_in_f1",
    "event_out_precision",
    "event_out_recall",
    "event_out_f1",
    "mean_timing_error",
    "mean_absolute_timing_error",
    "timing_error_unit",
    "person_detections",
    "fused_detections",
    "unique_tracking_ids",
    "possible_id_restarts",
    "frames_without_detections",
    "likely_static_tracks",
    "tracks_disappearing_after_heavy_overlap",
    "id_fragmentation_risk",
    "false_event_risk_score",
    "processing_fps",
    "processing_duration_seconds",
)


def rank_results(
    results: Sequence[Mapping[str, Any]],
    *,
    has_frame_ground_truth: bool,
) -> list[dict[str, Any]]:
    """Rank completed runs without treating aggregate totals as event accuracy."""

    def finite_number(value: object, fallback: float) -> float:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return fallback

    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        failed = item.get("status") != "completed"
        count_error = finite_number(item.get("aggregate_count_error"), math.inf)
        risk = finite_number(item.get("false_event_risk_score"), math.inf)
        fps = finite_number(item.get("processing_fps"), 0.0)
        name = str(item.get("name", ""))
        if has_frame_ground_truth:
            f1 = finite_number(item.get("event_f1"), -1.0)
            return failed, -f1, count_error, risk, -fps, name
        return failed, count_error, risk, -fps, name

    return [deepcopy(dict(item)) for item in sorted(results, key=key)]


def write_leaderboard(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    expected_entered: int,
    expected_exited: int,
    has_frame_ground_truth: bool,
    reproducibility: Mapping[str, Any] | None = None,
) -> None:
    """Write ``leaderboard.csv``, ``leaderboard.json``, and ``report.md``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_results(results, has_frame_ground_truth=has_frame_ground_truth)
    ranking_basis = (
        "direction-aware event F1, count error, false-event risk, then FPS"
        if has_frame_ground_truth
        else "aggregate direction count error, false-event risk, then FPS"
    )
    payload = {
        "ranking_basis": ranking_basis,
        "ground_truth": {
            "mode": "frame_events" if has_frame_ground_truth else "aggregate_only",
            "expected_entered": expected_entered,
            "expected_exited": expected_exited,
            "precision_recall_available": has_frame_ground_truth,
        },
        "reproducibility": deepcopy(dict(reproducibility or {})),
        "runs": ranked,
    }
    (output_dir / "leaderboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fieldnames = (
        "rank",
        "name",
        "status",
        "inference_mode",
        *_LEADERBOARD_METRIC_FIELDS,
        "configuration_sha256",
        "configuration",
        "output_paths",
        "error",
    )
    with (output_dir / "leaderboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, result in enumerate(ranked, start=1):
            row = dict(result)
            row["rank"] = rank
            row["configuration"] = json.dumps(
                result.get("configuration"), sort_keys=True, separators=(",", ":")
            )
            row["output_paths"] = json.dumps(
                result.get("output_paths"), sort_keys=True, separators=(",", ":")
            )
            writer.writerow(row)

    report = _render_report(
        ranked,
        ranking_basis=ranking_basis,
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        has_frame_ground_truth=has_frame_ground_truth,
        reproducibility=reproducibility,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _render_report(
    ranked: Sequence[Mapping[str, Any]],
    *,
    ranking_basis: str,
    expected_entered: int,
    expected_exited: int,
    has_frame_ground_truth: bool,
    reproducibility: Mapping[str, Any] | None,
) -> str:
    lines = [
        "# TrackBus experiment report",
        "",
        f"Ranking basis: {ranking_basis}.",
    ]
    if reproducibility:
        input_video = reproducibility.get("input_video", {})
        runtime = reproducibility.get("runtime", {})
        packages = runtime.get("packages", {}) if isinstance(runtime, Mapping) else {}
        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                f"- Input: `{input_video.get('filename', 'unknown')}`",
                f"- Input SHA-256: `{input_video.get('sha256', 'unknown')}`",
                f"- Input bytes: {input_video.get('size_bytes', 'unknown')}",
                f"- Python: {runtime.get('python_version', 'unknown')}",
                f"- Platform: {runtime.get('platform', 'unknown')}",
                "- Packages: "
                + ", ".join(
                    f"{name}={version}" for name, version in sorted(packages.items())
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "Frame-level, direction-aware ground truth was used for one-to-one "
                "event matching."
                if has_frame_ground_truth
                else (
                    "Only aggregate ground truth was available "
                    f"({expected_entered} IN / {expected_exited} OUT). Aggregate "
                    "totals cannot establish event precision, recall, or F1."
                )
            ),
            "",
            "| Rank | Run | Mode | Status | IN | OUT | Direction error | Event F1 | "
            "Raw det. | Fused det. | IDs | Restarts | Empty frames | FPS | Seconds |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, result in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(rank),
                    _markdown_cell(result.get("name")),
                    _markdown_cell(result.get("inference_mode")),
                    _markdown_cell(result.get("status")),
                    _format_metric(result.get("entered")),
                    _format_metric(result.get("exited")),
                    _format_metric(result.get("aggregate_count_error")),
                    _format_metric(result.get("event_f1"), digits=3),
                    _format_metric(result.get("person_detections")),
                    _format_metric(result.get("fused_detections")),
                    _format_metric(result.get("unique_tracking_ids")),
                    _format_metric(result.get("possible_id_restarts")),
                    _format_metric(result.get("frames_without_detections")),
                    _format_metric(result.get("processing_fps"), digits=2),
                    _format_metric(result.get("processing_duration_seconds"), digits=2),
                )
            )
            + " |"
        )

    completed = [item for item in ranked if item.get("status") == "completed"]
    lines.extend(["", "## Interpretation", ""])
    if completed:
        top = completed[0]
        lines.append(
            f"`{top['name']}` ranks first under the stated criteria. This is a "
            "comparative result for this video, not a general accuracy claim."
        )
        lines.extend(_multiview_comparison(completed, has_frame_ground_truth))
    else:
        lines.append("No experiment completed successfully; inspect per-run logs.")

    failed = [item for item in ranked if item.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed runs", ""])
        lines.extend(
            f"- `{item.get('name')}`: {_markdown_cell(item.get('error'))}"
            for item in failed
        )
    lines.append("")
    return "\n".join(lines)


def _multiview_comparison(
    completed: Sequence[Mapping[str, Any]], has_frame_ground_truth: bool
) -> list[str]:
    full = next(
        (item for item in completed if item.get("inference_mode") == "full_frame"),
        None,
    )
    multiview = next(
        (item for item in completed if item.get("inference_mode") == "multiview"),
        None,
    )
    if full is None or multiview is None:
        return []
    metric = "event F1" if has_frame_ground_truth else "direction count error"
    key = "event_f1" if has_frame_ground_truth else "aggregate_count_error"
    full_value = _format_metric(full.get(key), digits=3)
    multi_value = _format_metric(multiview.get(key), digits=3)
    full_fps = _format_metric(full.get("processing_fps"), digits=2)
    multi_fps = _format_metric(multiview.get("processing_fps"), digits=2)
    return [
        "",
        f"The highest-ranked full-frame run (`{full['name']}`) has {metric} "
        f"{full_value} at {full_fps} FPS; the highest-ranked multi-view run "
        f"(`{multiview['name']}`) has {metric} {multi_value} at {multi_fps} FPS.",
    ]


def _refuse_stale_run_artifacts(
    output_dir: Path, runs: Sequence[ExperimentRun]
) -> None:
    stale: list[Path] = []
    for run in runs:
        stale.extend(build_artifacts(output_dir, run).existing_files())
    if stale:
        preview = ", ".join(str(path) for path in stale[:3])
        remainder = f" (and {len(stale) - 3} more)" if len(stale) > 3 else ""
        raise ExperimentConfigError(
            "Experiment output contains stale run artifacts. Use a new output "
            "directory or explicitly archive/remove them before rerunning: "
            f"{preview}{remainder}."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reproducibility_metadata(input_path: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in (
        "trackbus-ai",
        "ultralytics",
        "torch",
        "numpy",
        "opencv-python",
        "PyYAML",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            if distribution == "trackbus-ai":
                from trackbus import __version__

                packages[distribution] = __version__
            else:
                packages[distribution] = None
    stat = input_path.stat()
    return {
        "input_video": {
            "filename": input_path.name,
            "path": str(input_path),
            "size_bytes": stat.st_size,
            "sha256": _sha256_file(input_path),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def _annotation_totals(document: GroundTruthAnnotations) -> tuple[int, int]:
    from trackbus.annotate_events import EventDirection

    return (
        sum(event.direction is EventDirection.IN for event in document.events),
        sum(event.direction is EventDirection.OUT for event in document.events),
    )


def _validated_ground_truth(
    path: Path | None,
    *,
    input_path: Path,
    expected_entered: int,
    expected_exited: int,
) -> GroundTruthAnnotations | None:
    """Load a complete annotation document tied to the requested source video."""

    if path is None:
        return None
    from trackbus.annotate_events import AnnotationError, load_annotations

    try:
        document = load_annotations(path)
    except (AnnotationError, OSError) as exc:
        raise ExperimentConfigError(
            f"Invalid frame-level ground truth '{path}': {exc}"
        ) from exc
    if document.video_filename != input_path.name:
        raise ExperimentConfigError(
            "Frame-level ground truth belongs to a different video "
            f"({document.video_filename!r} != {input_path.name!r})."
        )
    entered, exited = _annotation_totals(document)
    if (entered, exited) != (expected_entered, expected_exited):
        raise ExperimentConfigError(
            "Frame-level ground truth event totals do not match the matrix's "
            "aggregate ground truth; the annotation may be partial or belong to "
            f"another source (IN/OUT {entered}/{exited} != "
            f"{expected_entered}/{expected_exited})."
        )
    return document


def _resolve_contextual_path(
    value: str,
    matrix: ExperimentMatrix,
    *,
    kind: str,
) -> str:
    """Resolve explicit local paths without rewriting bare Ultralytics names."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise ExperimentConfigError(f"{kind.title()} does not exist: {candidate}")
        return str(candidate.resolve())

    if (
        kind == "model"
        and candidate.parent == Path(".")
        and _STANDARD_YOLO_WEIGHT.fullmatch(candidate.name)
    ):
        return value

    roots = (
        matrix.source_path.parent,
        matrix.base_config_path.parent,
        _repository_root(matrix.source_path),
    )
    checked: list[Path] = []
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return str(resolved)

    # Bare names such as yolo11n.pt and bytetrack.yaml are stable identifiers
    # understood by Ultralytics. A relative path with directories, however, is
    # clearly intended as a local file and must not depend on the caller's cwd.
    if candidate.parent != Path("."):
        locations = ", ".join(str(path) for path in checked)
        raise ExperimentConfigError(
            f"Could not resolve relative {kind} '{value}' from matrix, base-config, "
            f"or repository context (checked: {locations})."
        )
    return value


def _repository_root(start: Path | None = None) -> Path:
    """Return the nearest TrackBus project root for deterministic subprocesses."""

    location = (start or Path(__file__)).resolve()
    if location.is_file():
        location = location.parent
    for directory in (location, *location.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "trackbus"
        ).is_dir():
            return directory
    return Path(__file__).resolve().parents[1]


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], name: str
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ExperimentConfigError(
            f"'{name}' contains unknown key(s): {', '.join(unknown)}."
        )


def _validate_application_config_keys(value: Mapping[str, Any], name: str) -> None:
    _reject_unknown_keys(value, set(_APP_CONFIG_KEYS), name)
    for section, allowed in _APP_CONFIG_KEYS.items():
        if allowed is None or section not in value:
            continue
        raw_section = _require_mapping(value[section], f"{name}.{section}")
        _reject_unknown_keys(raw_section, allowed, f"{name}.{section}")

    camera = value.get("camera")
    if isinstance(camera, dict):
        lanes = camera.get("doorway_lanes")
        if lanes is not None:
            lanes_mapping = _require_mapping(lanes, f"{name}.camera.doorway_lanes")
            _reject_unknown_keys(
                lanes_mapping,
                {"anchor", "left_lane", "center_lane", "right_lane"},
                f"{name}.camera.doorway_lanes",
            )
        views = camera.get("inference_views")
        if views is not None:
            if not isinstance(views, list):
                raise ExperimentConfigError(
                    f"'{name}.camera.inference_views' must be a YAML list."
                )
            for index, view in enumerate(views):
                view_mapping = _require_mapping(
                    view, f"{name}.camera.inference_views[{index}]"
                )
                _reject_unknown_keys(
                    view_mapping,
                    {"name", "bounds", "enabled"},
                    f"{name}.camera.inference_views[{index}]",
                )


def _reject_protected_overrides(overrides: Mapping[str, Any], name: str) -> None:
    protected: list[str] = []
    model = overrides.get("model")
    if isinstance(model, Mapping):
        protected.extend(
            f"model.{key}"
            for key in sorted(set(model) & {"path", "confidence", "imgsz"})
        )
    if "tracking" in overrides:
        protected.append("tracking (use the run's tracker mapping)")
    camera = overrides.get("camera")
    if isinstance(camera, Mapping):
        protected.extend(
            f"camera.{key}"
            for key in sorted(set(camera) & {"detection_roi", "inference_views"})
        )
    if "detection_fusion" in overrides:
        protected.append("detection_fusion")
    if "outputs" in overrides:
        protected.append("outputs")
    if protected:
        raise ExperimentConfigError(
            f"'{name}.overrides' cannot change matrix-controlled setting(s): "
            f"{', '.join(protected)}."
        )


def _deep_merge(target: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"'{name}' must be a YAML mapping.")
    return value


def _normalized_bounds(value: object, name: str) -> tuple[float, float, float, float]:
    raw = value.get("bounds") if isinstance(value, dict) else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ExperimentConfigError(f"'views.{name}' must define four bounds.")
    try:
        left, top, right, bottom = (float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"'views.{name}' bounds must be numbers.") from exc
    if not all(0.0 <= item <= 1.0 for item in (left, top, right, bottom)):
        raise ExperimentConfigError(f"'views.{name}' bounds must be normalized.")
    if left >= right or top >= bottom:
        raise ExperimentConfigError(
            f"'views.{name}' requires left < right and top < bottom."
        )
    return left, top, right, bottom


def _bounded_float(
    value: object,
    name: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"'{name}' must be a number.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ExperimentConfigError(f"'{name}' must be in (0, 1].")
    return parsed


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ExperimentConfigError(f"'{name}' must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentConfigError(f"'{name}' must be an integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ExperimentConfigError(f"'{name}' must be an integer.")
    parsed = int(numeric)
    if parsed < 1:
        raise ExperimentConfigError(f"'{name}' must be at least 1.")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ExperimentConfigError(f"'{name}' must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentConfigError(f"'{name}' must be an integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ExperimentConfigError(f"'{name}' must be an integer.")
    parsed = int(numeric)
    if parsed < 0:
        raise ExperimentConfigError(f"'{name}' cannot be negative.")
    return parsed


def _optional_non_negative_int(value: object, name: str) -> int | None:
    return None if value is None else _non_negative_int(value, name)


def _optional_non_negative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentConfigError(f"'{name}' must be a number or null.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ExperimentConfigError(f"'{name}' cannot be negative.")
    return parsed


def _required_summary_int(summary: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in summary and summary[name] is not None:
            value = summary[name]
            if isinstance(value, bool):
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a non-negative integer."
                )
            try:
                parsed = int(value)
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a non-negative integer."
                ) from exc
            if not math.isfinite(numeric) or numeric != parsed or parsed < 0:
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a non-negative integer."
                )
            return parsed
    aliases = " or ".join(f"'{name}'" for name in names)
    raise ExperimentExecutionError(f"Run summary is missing required field {aliases}.")


def _required_summary_float(
    summary: Mapping[str, Any],
    *names: str,
    positive: bool = False,
) -> float:
    for name in names:
        if name in summary and summary[name] is not None:
            value = summary[name]
            if isinstance(value, bool):
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a finite "
                    f"{'positive' if positive else 'non-negative'} number."
                )
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a finite "
                    f"{'positive' if positive else 'non-negative'} number."
                ) from exc
            valid = parsed > 0 if positive else parsed >= 0
            if not math.isfinite(parsed) or not valid:
                raise ExperimentExecutionError(
                    f"Run summary field '{name}' must be a finite "
                    f"{'positive' if positive else 'non-negative'} number."
                )
            return parsed
    aliases = " or ".join(f"'{name}'" for name in names)
    raise ExperimentExecutionError(f"Run summary is missing required field {aliases}.")


def _format_metric(value: object, *, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_cell(value: object) -> str:
    return str(value if value is not None else "—").replace("|", "\\|")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.experiments",
        description="Run a bounded TrackBus configuration matrix and rank results.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Local input video")
    parser.add_argument(
        "--matrix", required=True, type=Path, help="Experiment matrix YAML"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Experiment artifact directory"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional frame-level event annotation JSON",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed run instead of reporting remaining runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        matrix = load_experiment_matrix(args.matrix)
        results = run_experiments(
            input_path=args.input,
            matrix=matrix,
            output_dir=args.output_dir,
            ground_truth_path=args.ground_truth,
            continue_on_error=not args.fail_fast,
        )
    except (ExperimentConfigError, ExperimentExecutionError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2
    completed = sum(result.get("status") == "completed" for result in results)
    LOGGER.info(
        "Completed %d/%d experiments; leaderboard: %s",
        completed,
        len(results),
        args.output_dir / "leaderboard.csv",
    )
    return 0 if completed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
