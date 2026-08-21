"""Bounded, reproducible detector-only TrackBus experiment sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trackbus.detection_benchmark import (
    DetectionBenchmarkError,
    DetectionGroundTruth,
    benchmark_detector,
    load_detection_ground_truth,
    write_benchmark_json,
    write_frame_metrics_csv,
)
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device
from trackbus.interfaces import DetectorBackend

LOGGER = logging.getLogger("trackbus.detection_sweep")
DEFAULT_MAXIMUM_RUNS = 24


class DetectionSweepError(ValueError):
    """Raised when a sweep matrix is unsafe or invalid."""


@dataclass(frozen=True)
class DetectionSweepRun:
    name: str
    model: str
    image_size: int
    confidence: float
    precision: str = "fp32"
    maximum_frames: int | None = None

    @property
    def slug(self) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return value or "run"


@dataclass(frozen=True)
class DetectionSweepMatrix:
    runs: tuple[DetectionSweepRun, ...]
    device: str = "auto"
    low_confidence_threshold: float = 0.35
    ground_truth_iou_threshold: float = 0.50
    stability_iou_threshold: float = 0.10
    edge_margin_pixels: int = 2


DetectorFactory = Callable[[DetectionSweepRun, str | int, str], DetectorBackend]


def load_detection_sweep(path: Path) -> DetectionSweepMatrix:
    """Load an explicit, bounded detector sweep from YAML."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DetectionSweepError(
            f"Could not read sweep matrix '{path}': {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DetectionSweepError(f"Sweep matrix is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise DetectionSweepError("Sweep matrix root must be a mapping.")
    _reject_unknown(
        document,
        "sweep matrix",
        {
            "version",
            "device",
            "maximum_runs",
            "low_confidence_threshold",
            "ground_truth_iou_threshold",
            "stability_iou_threshold",
            "edge_margin_pixels",
            "runs",
        },
    )
    if document.get("version") != 1:
        raise DetectionSweepError("Sweep matrix version must be 1.")
    maximum_runs = _positive_int(
        document.get("maximum_runs", DEFAULT_MAXIMUM_RUNS), "maximum_runs"
    )
    if maximum_runs > DEFAULT_MAXIMUM_RUNS:
        raise DetectionSweepError(
            f"maximum_runs cannot exceed the safety limit {DEFAULT_MAXIMUM_RUNS}."
        )
    raw_runs = document.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise DetectionSweepError("runs must be a non-empty list.")
    if len(raw_runs) > maximum_runs:
        raise DetectionSweepError(
            f"Sweep defines {len(raw_runs)} runs but maximum_runs is {maximum_runs}."
        )
    runs = tuple(_parse_run(value, index) for index, value in enumerate(raw_runs))
    names = [run.name for run in runs]
    slugs = [run.slug for run in runs]
    if len(set(names)) != len(names):
        raise DetectionSweepError("Sweep run names must be unique.")
    if len(set(slugs)) != len(slugs):
        raise DetectionSweepError("Sweep run names must produce unique file slugs.")
    device = document.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise DetectionSweepError("device must be non-empty text.")
    return DetectionSweepMatrix(
        runs=runs,
        device=device.strip(),
        low_confidence_threshold=_probability(
            document.get("low_confidence_threshold", 0.35),
            "low_confidence_threshold",
        ),
        ground_truth_iou_threshold=_iou(
            document.get("ground_truth_iou_threshold", 0.50),
            "ground_truth_iou_threshold",
        ),
        stability_iou_threshold=_iou(
            document.get("stability_iou_threshold", 0.10),
            "stability_iou_threshold",
        ),
        edge_margin_pixels=_non_negative_int(
            document.get("edge_margin_pixels", 2), "edge_margin_pixels"
        ),
    )


def run_detection_sweep(
    *,
    video_path: Path,
    matrix: DetectionSweepMatrix,
    output_dir: Path,
    ground_truth: DetectionGroundTruth | None = None,
    detector_factory: DetectorFactory | None = None,
    device_resolver: Callable[[str], tuple[str | int, str]] = resolve_device,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Run each explicit detector configuration and persist auditable evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device, device_label = device_resolver(matrix.device)
    factory = detector_factory or _default_detector_factory
    results: list[dict[str, Any]] = []
    for run in matrix.runs:
        run_dir = output_dir / run.slug
        summary_path = run_dir / "benchmark.json"
        frames_path = run_dir / "frames.csv"
        LOGGER.info(
            "Running %s: model=%s imgsz=%d conf=%.3f",
            run.name,
            run.model,
            run.image_size,
            run.confidence,
        )
        try:
            detector = factory(run, device, device_label)
            result = benchmark_detector(
                video_path=video_path,
                detector=detector,
                model_name=run.model,
                image_size=run.image_size,
                confidence_threshold=run.confidence,
                device_label=device_label,
                requested_precision=run.precision,
                effective_precision=getattr(detector, "precision", run.precision),
                ground_truth=ground_truth,
                low_confidence_threshold=matrix.low_confidence_threshold,
                matching_iou_threshold=matrix.ground_truth_iou_threshold,
                stability_iou_threshold=matrix.stability_iou_threshold,
                edge_margin_pixels=matrix.edge_margin_pixels,
                maximum_frames=run.maximum_frames,
            )
            write_benchmark_json(summary_path, result)
            write_frame_metrics_csv(frames_path, result.frames)
            results.append(
                _completed_run(run, result.summary, summary_path, frames_path)
            )
        except (DetectionBenchmarkError, DetectorError, OSError) as exc:
            failed = {
                "name": run.name,
                "status": "failed",
                "model": run.model,
                "image_size": run.image_size,
                "confidence": run.confidence,
                "requested_precision": run.precision,
                "effective_precision": None,
                "error": str(exc),
            }
            results.append(failed)
            LOGGER.error("Detection sweep run '%s' failed: %s", run.name, exc)
            if not continue_on_error:
                break
    _write_sweep_outputs(output_dir, results, ground_truth is not None)
    return results


def _completed_run(
    run: DetectionSweepRun,
    summary: Mapping[str, Any],
    summary_path: Path,
    frames_path: Path,
) -> dict[str, Any]:
    detections = summary["detections"]
    performance = summary["performance"]
    stability = summary["frame_to_frame_box_stability_proxy"]
    truth = summary.get("ground_truth")
    return {
        "name": run.name,
        "status": "completed",
        "metric_status": summary["metric_status"],
        "model": run.model,
        "image_size": run.image_size,
        "confidence": run.confidence,
        "requested_precision": summary["configuration"]["requested_precision"],
        "effective_precision": summary["configuration"]["effective_precision"],
        "processed_frames": summary["video"]["processed_frame_count"],
        "detections": detections["total"],
        "frames_without_detections": detections["frames_without_detections"],
        "maximum_zero_detection_streak_frames": detections[
            "maximum_zero_detection_streak_frames"
        ],
        "stability_match_rate": stability["match_rate"],
        "pipeline_fps": performance["pipeline_fps_including_decode"],
        "mean_inference_latency_ms": performance["inference_latency_ms"]["mean"],
        "precision_score": truth.get("precision") if truth else None,
        "recall": truth.get("recall") if truth else None,
        "f1": truth.get("f1") if truth else None,
        "missed_persons": truth.get("missed_persons") if truth else None,
        "false_positives": truth.get("false_positives") if truth else None,
        "summary_json": str(summary_path),
        "frames_csv": str(frames_path),
    }


def _write_sweep_outputs(
    output_dir: Path, results: Sequence[Mapping[str, Any]], has_ground_truth: bool
) -> None:
    document = {
        "schema_version": 1,
        "benchmark_type": "detector_only_sweep",
        "selection_status": (
            "ground_truth_metrics_available"
            if has_ground_truth
            else "no_accuracy_winner_without_box_ground_truth"
        ),
        "results": list(results),
    }
    (output_dir / "results.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = _sweep_csv_fields()
    with (output_dir / "leaderboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _sweep_csv_fields() -> list[str]:
    return [
        "name",
        "status",
        "metric_status",
        "model",
        "image_size",
        "confidence",
        "requested_precision",
        "effective_precision",
        "processed_frames",
        "detections",
        "frames_without_detections",
        "maximum_zero_detection_streak_frames",
        "stability_match_rate",
        "pipeline_fps",
        "mean_inference_latency_ms",
        "precision_score",
        "recall",
        "f1",
        "missed_persons",
        "false_positives",
        "error",
        "summary_json",
        "frames_csv",
    ]


def _default_detector_factory(
    run: DetectionSweepRun, device: str | int, device_label: str
) -> DetectorBackend:
    return UltralyticsDetector(
        run.model,
        run.confidence,
        run.image_size,
        device=device,
        device_label=device_label,
        precision=run.precision,
    )


def _parse_run(value: object, index: int) -> DetectionSweepRun:
    name = f"runs[{index}]"
    if not isinstance(value, dict):
        raise DetectionSweepError(f"{name} must be a mapping.")
    _reject_unknown(
        value,
        name,
        {"name", "model", "image_size", "confidence", "precision", "maximum_frames"},
    )
    run_name = value.get("name")
    model = value.get("model")
    if not isinstance(run_name, str) or not run_name.strip():
        raise DetectionSweepError(f"{name}.name must be non-empty text.")
    if not isinstance(model, str) or not model.strip():
        raise DetectionSweepError(f"{name}.model must be non-empty text.")
    if "://" in model:
        raise DetectionSweepError(f"{name}.model cannot be a remote URL.")
    precision = value.get("precision", "fp32")
    if precision not in {"fp32", "fp16"}:
        raise DetectionSweepError(f"{name}.precision must be fp32 or fp16.")
    maximum_frames_value = value.get("maximum_frames")
    maximum_frames = (
        _positive_int(maximum_frames_value, f"{name}.maximum_frames")
        if maximum_frames_value is not None
        else None
    )
    image_size = _positive_int(value.get("image_size"), f"{name}.image_size")
    if image_size < 32:
        raise DetectionSweepError(f"{name}.image_size must be at least 32.")
    return DetectionSweepRun(
        name=run_name.strip(),
        model=model.strip(),
        image_size=image_size,
        confidence=_probability(value.get("confidence"), f"{name}.confidence"),
        precision=precision,
        maximum_frames=maximum_frames,
    )


def _reject_unknown(value: Mapping[str, Any], name: str, allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DetectionSweepError(
            f"{name} contains unknown key(s): {', '.join(unknown)}."
        )


def _probability(value: object, name: str) -> float:
    parsed = _finite_float(value, name)
    if not 0.0 < parsed <= 1.0:
        raise DetectionSweepError(f"{name} must be in (0, 1].")
    return parsed


def _iou(value: object, name: str) -> float:
    parsed = _finite_float(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise DetectionSweepError(f"{name} must be in [0, 1].")
    return parsed


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise DetectionSweepError(f"{name} must be a finite number.")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DetectionSweepError(f"{name} must be a finite number.") from exc
    if not (float("-inf") < parsed < float("inf")):
        raise DetectionSweepError(f"{name} must be a finite number.")
    return parsed


def _positive_int(value: object, name: str) -> int:
    parsed = _non_negative_int(value, name)
    if parsed < 1:
        raise DetectionSweepError(f"{name} must be at least 1.")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DetectionSweepError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionSweepError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0 or (isinstance(value, float) and not value.is_integer()):
        raise DetectionSweepError(f"{name} must be a non-negative integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.sweep_detection",
        description=(
            "Run an explicit bounded model/resolution/confidence detector sweep. "
            "This command never invokes tracking or passenger counting."
        ),
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        matrix = load_detection_sweep(args.matrix)
        ground_truth = (
            load_detection_ground_truth(args.ground_truth)
            if args.ground_truth is not None
            else None
        )
        results = run_detection_sweep(
            video_path=args.video,
            matrix=matrix,
            output_dir=args.output_dir,
            ground_truth=ground_truth,
            continue_on_error=not args.fail_fast,
        )
    except (
        DetectionSweepError,
        DetectionBenchmarkError,
        DetectorError,
        OSError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2
    completed = sum(result["status"] == "completed" for result in results)
    LOGGER.info(
        "Completed %d/%d detector runs; results: %s",
        completed,
        len(results),
        args.output_dir / "leaderboard.csv",
    )
    return 0 if completed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
