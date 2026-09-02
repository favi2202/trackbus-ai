"""Command-line entry point for TrackBus v0.2."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from trackbus.config import AppConfig, ConfigError, load_config
from trackbus.counter import PassengerCounter
from trackbus.detection_export import (
    DetectionCsvExporter,
    derive_detection_export_paths,
)
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device
from trackbus.event_logger import EventLogger, derive_artifact_paths
from trackbus.failure_mining import (
    FailureMiner,
    FailureMiningError,
    derive_failure_paths,
)
from trackbus.fusion import NmsDetectionFusion
from trackbus.tracker import (
    ByteTrackAdapter,
    TrackContinuityAdapter,
    TrackerAdapterError,
)
from trackbus.video_processor import VideoProcessingError, VideoProcessor

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
LOGGER = logging.getLogger("trackbus")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _image_size(value: str) -> int:
    parsed = int(value)
    if parsed < 32:
        raise argparse.ArgumentTypeError("must be at least 32 pixels")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.main",
        description="Detect, track, and count bus passengers in a local video.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Local input video")
    parser.add_argument("--output", type=Path, help="Annotated output video")
    parser.add_argument("--capacity", type=_positive_int, help="Nominal bus capacity")
    parser.add_argument(
        "--device",
        help="Inference device: auto, cpu, cuda, cuda:N, or a GPU index",
    )
    parser.add_argument("--confidence", type=_confidence, help="YOLO threshold (0, 1]")
    parser.add_argument(
        "--detector-floor",
        type=_confidence,
        help=(
            "Lowest YOLO prediction retained for ByteTrack; cannot exceed confidence"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=_image_size,
        help="YOLO inference image size in pixels (minimum 32)",
    )
    parser.add_argument("--model", help="Ultralytics model name or local weights path")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML configuration file"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show a live preview window"
    )
    parser.add_argument(
        "--initial-occupancy",
        type=_non_negative_int,
        help="Passengers already inside when the video starts",
    )
    return parser


def _load_runtime_config(args: argparse.Namespace) -> AppConfig:
    """Load YAML and apply command-line values with CLI precedence."""

    return load_config(args.config).with_overrides(
        output=args.output,
        capacity=args.capacity,
        initial_occupancy=args.initial_occupancy,
        device=args.device,
        confidence=args.confidence,
        detector_floor=args.detector_floor,
        model=args.model,
        imgsz=args.imgsz,
    )


def _validated_artifact_paths(
    input_path: Path,
    **artifacts: Path,
) -> dict[str, Path]:
    """Resolve output paths and reject aliases before anything is truncated."""

    resolved_input = input_path.expanduser().resolve()
    resolved = {name: path.expanduser().resolve() for name, path in artifacts.items()}
    by_path: dict[Path, list[str]] = {}
    for name, path in resolved.items():
        if path == resolved_input:
            raise VideoProcessingError(
                f"Output artifact '{name}' must not overwrite the input video."
            )
        by_path.setdefault(path, []).append(name)
    duplicates = [names for names in by_path.values() if len(names) > 1]
    if duplicates:
        joined = "; ".join(", ".join(names) for names in duplicates)
        raise VideoProcessingError(
            f"Output artifact paths must be pairwise distinct (conflict: {joined})."
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _load_runtime_config(args)
        logging.basicConfig(
            level=getattr(logging, config.logging_level),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        input_path = args.input.expanduser().resolve()
        configured_output = config.outputs.video.expanduser().resolve()
        if not input_path.is_file():
            raise VideoProcessingError(f"Input video does not exist: {input_path}")
        csv_path, summary_path = derive_artifact_paths(
            configured_output,
            config.outputs.events_csv,
            config.outputs.summary_json,
        )
        raw_path, fused_path, tracks_path = derive_detection_export_paths(
            configured_output,
            config.outputs.raw_detections_csv,
            config.outputs.fused_detections_csv,
            config.outputs.tracks_csv,
        )
        failure_jsonl, failure_frames = derive_failure_paths(
            configured_output,
            config.diagnostics.failure_mining.output_jsonl,
            config.diagnostics.failure_mining.frames_directory,
        )
        optional_artifacts: dict[str, Path] = {}
        if config.diagnostics.failure_mining.enabled:
            optional_artifacts["failure_jsonl"] = failure_jsonl
        artifacts = _validated_artifact_paths(
            input_path,
            video=configured_output,
            events_csv=csv_path,
            summary_json=summary_path,
            raw_detections_csv=raw_path,
            fused_detections_csv=fused_path,
            tracks_csv=tracks_path,
            **optional_artifacts,
        )
        output_path = artifacts["video"]
        csv_path = artifacts["events_csv"]
        summary_path = artifacts["summary_json"]
        raw_path = artifacts["raw_detections_csv"]
        fused_path = artifacts["fused_detections_csv"]
        tracks_path = artifacts["tracks_csv"]
        if config.diagnostics.failure_mining.enabled:
            failure_jsonl = artifacts["failure_jsonl"]
            resolved_failure_frames = failure_frames.expanduser().resolve()
            if resolved_failure_frames in artifacts.values():
                raise VideoProcessingError(
                    "Failure frame directory must not alias an input or output file."
                )
            failure_frames = resolved_failure_frames

        device, device_label = resolve_device(config.device)
        LOGGER.info(
            "Loading model %s on %s (detector_floor=%.3f, "
            "confidence_reference=%.3f, imgsz=%d, precision=%s)",
            config.model.path,
            device_label,
            config.model.effective_detector_floor,
            config.model.confidence,
            config.model.imgsz,
            config.model.precision,
        )
        detector = UltralyticsDetector(
            config.model.path,
            config.model.confidence,
            config.model.imgsz,
            device=device,
            device_label=device_label,
            detector_floor=config.model.effective_detector_floor,
            precision=config.model.precision,
        )
        base_tracker = ByteTrackAdapter(
            tracker_config=config.tracking.tracker,
            device_label=device_label,
            tracker_overrides=config.tracking.bytetrack_overrides(),
        )
        confidence_contract = base_tracker.confidence_contract(
            config.model.effective_detector_floor
        )
        if warning := confidence_contract.get("warning"):
            LOGGER.warning("%s", warning)
        continuity = config.tracking.continuity
        tracker = (
            TrackContinuityAdapter(
                base_tracker,
                max_gap_frames=continuity.max_gap_frames,
                max_centroid_distance=continuity.max_centroid_distance,
                minimum_iou=continuity.minimum_iou,
                maximum_size_ratio=continuity.maximum_size_ratio,
                minimum_direction_cosine=continuity.minimum_direction_cosine,
                minimum_match_score=continuity.minimum_match_score,
            )
            if continuity.enabled
            else base_tracker
        )
        fusion = NmsDetectionFusion(
            iou_threshold=config.detection_fusion.iou_threshold,
            confidence_strategy=config.detection_fusion.confidence_strategy,
            prefer_full_frame=config.detection_fusion.prefer_full_frame,
        )
        counter = PassengerCounter(
            capacity=config.capacity,
            initial_occupancy=config.initial_occupancy,
            minimum_zone_frames=config.tracking.minimum_zone_frames,
            minimum_origin_zone_frames=(
                config.tracking.effective_minimum_origin_zone_frames
            ),
            minimum_destination_zone_frames=(
                config.tracking.effective_minimum_destination_zone_frames
            ),
            maximum_transition_gap_frames=(
                config.tracking.maximum_transition_gap_frames
            ),
            event_cooldown_frames=config.tracking.event_cooldown_frames,
            stale_track_timeout=config.tracking.stale_track_timeout,
        )
        with (
            EventLogger(csv_path, summary_path) as event_logger,
            DetectionCsvExporter(
                enabled=config.diagnostics.export_detection_csv,
                raw_path=raw_path,
                fused_path=fused_path,
                tracks_path=tracks_path,
            ) as detection_exporter,
            FailureMiner(
                config.diagnostics.failure_mining,
                output_path=failure_jsonl,
                frames_directory=failure_frames,
                low_confidence_threshold=(
                    config.diagnostics.failure_mining.low_confidence_threshold
                    or config.model.confidence
                ),
                edge_margin_pixels=config.diagnostics.edge_margin_pixels,
                heavy_overlap_iou=config.diagnostics.heavy_overlap_iou,
                initial_occupancy=config.initial_occupancy,
            ) as failure_miner,
        ):
            processor = VideoProcessor(
                tracker=tracker,
                detector=detector,
                fusion=fusion,
                counter=counter,
                zones_config=config.zones,
                event_logger=event_logger,
                model_name=config.model.path,
                model_confidence=config.model.confidence,
                detector_floor=config.model.effective_detector_floor,
                confidence_contract=confidence_contract,
                inference_image_size=config.model.imgsz,
                model_precision=detector.precision,
                camera_config=config.camera,
                diagnostics_config=config.diagnostics,
                zone_anchor=config.tracking.zone_anchor,
                zone_boundary_hysteresis=(config.tracking.zone_boundary_hysteresis),
                detection_exporter=detection_exporter,
                failure_miner=failure_miner,
            )
            summary = processor.process(input_path, output_path, show=args.show)

        LOGGER.info(
            "Finished: %d frames, %d entered, %d exited, occupancy %d",
            summary.processed_frames,
            summary.entered_total,
            summary.exited_total,
            summary.final_occupancy,
        )
        LOGGER.info("Annotated video: %s", output_path)
        LOGGER.info("Event log: %s", csv_path)
        LOGGER.info("Summary: %s", summary_path)
        if config.diagnostics.export_detection_csv:
            LOGGER.info("Raw detections: %s", raw_path)
            LOGGER.info("Fused detections: %s", fused_path)
            LOGGER.info("Tracks: %s", tracks_path)
        if config.diagnostics.failure_mining.enabled:
            LOGGER.info("Failure evidence: %s", failure_jsonl)
            if config.diagnostics.failure_mining.capture_frames:
                LOGGER.warning(
                    "Failure frames may contain personal data: %s", failure_frames
                )
        return 0
    except (
        ConfigError,
        DetectorError,
        TrackerAdapterError,
        FailureMiningError,
        VideoProcessingError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
