"""Command-line entry point for TrackBus v0.1."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from trackbus.config import ConfigError, load_config
from trackbus.counter import PassengerCounter
from trackbus.detector import DetectorError, PersonDetector, resolve_device
from trackbus.event_logger import EventLogger, derive_artifact_paths
from trackbus.tracker import ByteTrackPersonTracker
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config).with_overrides(
            output=args.output,
            capacity=args.capacity,
            initial_occupancy=args.initial_occupancy,
            device=args.device,
            confidence=args.confidence,
            model=args.model,
        )
        logging.basicConfig(
            level=getattr(logging, config.logging_level),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        input_path = args.input.expanduser().resolve()
        output_path = config.outputs.video.expanduser().resolve()
        if not input_path.is_file():
            raise VideoProcessingError(f"Input video does not exist: {input_path}")
        if input_path == output_path:
            raise VideoProcessingError(
                "Input and output video paths must be different."
            )

        device, device_label = resolve_device(config.device)
        LOGGER.info("Loading model %s on %s", config.model.path, device_label)
        detector = PersonDetector(config.model.path, config.model.confidence)
        tracker = ByteTrackPersonTracker(
            detector,
            tracker_config=config.tracking.tracker,
            device=device,
            device_label=device_label,
        )
        counter = PassengerCounter(
            capacity=config.capacity,
            initial_occupancy=config.initial_occupancy,
            minimum_zone_frames=config.tracking.minimum_zone_frames,
            stale_track_timeout=config.tracking.stale_track_timeout,
        )
        csv_path, summary_path = derive_artifact_paths(
            output_path, config.outputs.events_csv, config.outputs.summary_json
        )
        with EventLogger(csv_path, summary_path) as event_logger:
            processor = VideoProcessor(
                tracker=tracker,
                counter=counter,
                zones_config=config.zones,
                event_logger=event_logger,
                model_name=config.model.path,
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
        return 0
    except (
        ConfigError,
        DetectorError,
        VideoProcessingError,
        OSError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
