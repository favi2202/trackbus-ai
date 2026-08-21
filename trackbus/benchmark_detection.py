"""Command-line detector-only benchmark for TrackBus."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from trackbus.detection_benchmark import (
    DetectionBenchmarkError,
    benchmark_detector,
    load_detection_ground_truth,
    write_benchmark_json,
    write_frame_metrics_csv,
)
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device

LOGGER = logging.getLogger("trackbus.detection_benchmark")


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _iou(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _image_size(value: str) -> int:
    parsed = int(value)
    if parsed < 32:
        raise argparse.ArgumentTypeError("must be at least 32 pixels")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.benchmark_detection",
        description=(
            "Benchmark YOLO person detections without tracking, zones, or IN/OUT "
            "counting. Precision/recall/F1 are reported only with box ground truth."
        ),
    )
    parser.add_argument("--video", required=True, type=Path, help="Local input video")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights")
    parser.add_argument("--imgsz", type=_image_size, default=640)
    parser.add_argument("--conf", type=_probability, default=0.25)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:N, or GPU index"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--ground-truth", type=Path, help="Optional detector-box annotation JSON"
    )
    parser.add_argument("--ground-truth-iou", type=_iou, default=0.50)
    parser.add_argument("--stability-iou", type=_iou, default=0.10)
    parser.add_argument("--low-confidence", type=_probability, default=0.35)
    parser.add_argument("--edge-margin", type=_non_negative_int, default=2)
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        help="Optional bounded smoke-run frame count",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Summary JSON; defaults beside the video",
    )
    parser.add_argument(
        "--frames-csv",
        type=Path,
        help="Frame evidence CSV; defaults beside the summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    output = args.output or args.video.with_suffix(".detection-benchmark.json")
    frames_csv = args.frames_csv or output.with_suffix(".frames.csv")
    try:
        _validate_artifact_paths(
            video=args.video,
            ground_truth=args.ground_truth,
            output=output,
            frames_csv=frames_csv,
        )
        device, device_label = resolve_device(args.device)
        ground_truth = (
            load_detection_ground_truth(args.ground_truth)
            if args.ground_truth is not None
            else None
        )
        LOGGER.info(
            "Detector-only run: model=%s imgsz=%d conf=%.3f device=%s",
            args.model,
            args.imgsz,
            args.conf,
            device_label,
        )
        detector = UltralyticsDetector(
            args.model,
            args.conf,
            args.imgsz,
            device=device,
            device_label=device_label,
            precision=args.precision,
        )
        result = benchmark_detector(
            video_path=args.video,
            detector=detector,
            model_name=args.model,
            image_size=args.imgsz,
            confidence_threshold=args.conf,
            device_label=device_label,
            requested_precision=args.precision,
            effective_precision=detector.precision,
            ground_truth=ground_truth,
            low_confidence_threshold=args.low_confidence,
            matching_iou_threshold=args.ground_truth_iou,
            stability_iou_threshold=args.stability_iou,
            edge_margin_pixels=args.edge_margin,
            maximum_frames=args.max_frames,
        )
        write_benchmark_json(output, result)
        write_frame_metrics_csv(frames_csv, result.frames)
    except (DetectionBenchmarkError, DetectorError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2
    detections = result.summary["detections"]
    performance = result.summary["performance"]
    LOGGER.info(
        "Processed %d frames, %d detections, %.2f pipeline FPS; results: %s",
        result.summary["video"]["processed_frame_count"],
        detections["total"],
        performance["pipeline_fps_including_decode"],
        output,
    )
    if ground_truth is None:
        LOGGER.warning(
            "No box ground truth supplied: precision, recall, and F1 were not claimed."
        )
    return 0


def _validate_artifact_paths(
    *,
    video: Path,
    ground_truth: Path | None,
    output: Path,
    frames_csv: Path,
) -> None:
    paths = {
        "video": video.expanduser().resolve(),
        "output": output.expanduser().resolve(),
        "frames_csv": frames_csv.expanduser().resolve(),
    }
    if ground_truth is not None:
        paths["ground_truth"] = ground_truth.expanduser().resolve()
    by_path: dict[Path, list[str]] = {}
    for name, path in paths.items():
        by_path.setdefault(path, []).append(name)
    conflicts = [names for names in by_path.values() if len(names) > 1]
    if conflicts:
        rendered = "; ".join(", ".join(names) for names in conflicts)
        raise DetectionBenchmarkError(
            f"Input and output paths must be distinct (conflict: {rendered})."
        )


if __name__ == "__main__":
    sys.exit(main())
