"""Bounded, evidence-first camera suitability diagnostics for TrackBus."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.calibration import FrameCalibration
from trackbus.config import (
    AppConfig,
    CameraConfig,
    ConfigError,
    InferenceViewConfig,
    ZonesConfig,
    load_config,
    resolve_inference_views,
)
from trackbus.detection import Detection
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device
from trackbus.fusion import NmsDetectionFusion
from trackbus.interfaces import DetectionFusion, DetectorBackend
from trackbus.views import MultiViewInference

LOGGER = logging.getLogger("trackbus.camera_quality")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class CameraQualityError(RuntimeError):
    """Raised when a camera-quality run cannot produce reliable diagnostics."""


@dataclass(frozen=True)
class FrameQualitySample:
    """Privacy-preserving numeric evidence from one sampled video frame."""

    frame_number: int
    timestamp_seconds: float
    brightness_mean: float
    dark_pixel_percentage: float
    bright_pixel_percentage: float
    sharpness_laplacian_variance: float
    detection_count: int
    confidences: tuple[float, ...]
    person_height_ratios: tuple[float, ...]
    edge_clipped_detection_count: int


@dataclass(frozen=True)
class CameraQualityReport:
    """Machine-readable suitability report; explicitly not an accuracy score."""

    source: str
    status: str
    diagnostic_not_accuracy: bool
    reasons: tuple[str, ...]
    recommendations: tuple[str, ...]
    video: dict[str, Any]
    sampling: dict[str, Any]
    optical_quality: dict[str, Any]
    detection_diagnostics: dict[str, Any]
    calibration_coverage: dict[str, Any]
    performance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_frame_quality(
    frame: NDArray[np.uint8],
    detections: Sequence[Detection] = (),
    *,
    frame_number: int = 0,
    timestamp_seconds: float = 0.0,
    edge_margin_pixels: int = 2,
) -> FrameQualitySample:
    """Measure optical and detection geometry without retaining the frame."""

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise CameraQualityError("Camera-quality frames must be non-empty BGR images.")
    if edge_margin_pixels < 0:
        raise CameraQualityError("edge_margin_pixels cannot be negative.")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    dark_percentage = float(np.mean(gray <= 15) * 100.0)
    bright_percentage = float(np.mean(gray >= 240) * 100.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height_ratios: list[float] = []
    edge_count = 0
    for detection in detections:
        left, top, right, bottom = detection.bounding_box
        height_ratios.append(max(0.0, bottom - top) / max(1, height))
        if (
            left <= edge_margin_pixels
            or top <= edge_margin_pixels
            or right >= width - 1 - edge_margin_pixels
            or bottom >= height - 1 - edge_margin_pixels
        ):
            edge_count += 1
    return FrameQualitySample(
        frame_number=frame_number,
        timestamp_seconds=round(timestamp_seconds, 6),
        brightness_mean=round(brightness, 3),
        dark_pixel_percentage=round(dark_percentage, 3),
        bright_pixel_percentage=round(bright_percentage, 3),
        sharpness_laplacian_variance=round(sharpness, 3),
        detection_count=len(detections),
        confidences=tuple(detection.confidence for detection in detections),
        person_height_ratios=tuple(height_ratios),
        edge_clipped_detection_count=edge_count,
    )


def calibration_path_coverage(
    camera: CameraConfig,
    zones: ZonesConfig,
) -> dict[str, Any]:
    """Report whether enabled inference views retain the configured crossing path."""

    enabled = tuple(view for view in resolve_inference_views(camera) if view.enabled)
    required_points = (*zones.outside, *zones.inside)
    single_view = any(
        all(_point_in_view(point, view) for point in required_points)
        for view in enabled
    )
    combined = all(
        any(_point_in_view(point, view) for view in enabled)
        for point in required_points
    )
    return {
        "enabled_inference_views": [view.name for view in enabled],
        "single_view_covers_outside_and_inside": single_view,
        "combined_views_cover_outside_and_inside": combined,
        "assessment_basis": "configured_normalized_zones_and_inference_views",
    }


def classify_camera_quality(
    *,
    width: int,
    height: int,
    blurred_frame_percentage: float,
    underexposed_frame_percentage: float,
    overexposed_frame_percentage: float,
    complete_path_coverage: bool,
    detection_enabled: bool,
    detection_total: int,
    median_person_height_ratio: float | None,
    edge_clipped_detection_percentage: float | None,
    sudden_count_drop_percentage: float | None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Classify heuristic suitability with concrete, non-accuracy reasons."""

    reasons: list[str] = []
    recommendations: list[str] = []
    unsuitable = False
    marginal = False

    if width < 320 or height < 240:
        unsuitable = True
        reasons.append(f"resolution {width}x{height} is below 320x240")
        recommendations.append("Use a higher-resolution source before calibration.")
    elif width < 640 or height < 360:
        marginal = True
        reasons.append(f"resolution {width}x{height} leaves limited person detail")
        recommendations.append("Prefer at least 640x360, then re-run this analyzer.")

    if not complete_path_coverage:
        unsuitable = True
        reasons.append("enabled inference views do not cover both configured zones")
        recommendations.append(
            "Reframe or recalibrate without cropping the OUTSIDE-to-INSIDE path."
        )
    if blurred_frame_percentage >= 70.0:
        unsuitable = True
        reasons.append(f"{blurred_frame_percentage:.1f}% of samples appear blurred")
        recommendations.append("Reduce vibration or motion blur and improve focus.")
    elif blurred_frame_percentage >= 25.0:
        marginal = True
        reasons.append(f"{blurred_frame_percentage:.1f}% of samples appear blurred")
        recommendations.append("Improve focus, shutter speed, mounting, or lighting.")

    exposure_max = max(underexposed_frame_percentage, overexposed_frame_percentage)
    if exposure_max >= 70.0:
        unsuitable = True
        reasons.append("most sampled frames have unusable exposure")
        recommendations.append("Correct camera exposure or doorway lighting.")
    elif exposure_max >= 20.0:
        marginal = True
        reasons.append("exposure is unstable in a meaningful share of samples")
        recommendations.append("Stabilize exposure and reduce glare or deep shadows.")

    if not detection_enabled:
        marginal = True
        reasons.append("person-detection geometry was intentionally not evaluated")
        recommendations.append("Re-run with YOLO enabled before approving the camera.")
    elif detection_total == 0:
        marginal = True
        reasons.append("no people were detected in the sampled frames")
        recommendations.append(
            "Use footage containing representative doorway traffic before approval."
        )
    elif median_person_height_ratio is not None:
        if median_person_height_ratio < 0.025:
            unsuitable = True
            reasons.append("detected people are extremely small in the frame")
            recommendations.append(
                "Move the camera closer or use a tighter full-path view."
            )
        elif median_person_height_ratio < 0.08:
            marginal = True
            reasons.append("detected people provide limited pixel detail")
            recommendations.append("Improve camera placement before model fine-tuning.")

    if edge_clipped_detection_percentage is not None:
        if edge_clipped_detection_percentage >= 80.0:
            unsuitable = True
            reasons.append("most detected people are clipped by a frame edge")
            recommendations.append(
                "Widen or reposition the camera view around the door."
            )
        elif edge_clipped_detection_percentage >= 25.0:
            marginal = True
            reasons.append("many detected people touch a frame edge")
            recommendations.append(
                "Reposition the camera to retain complete trajectories."
            )
    if (
        sudden_count_drop_percentage is not None
        and sudden_count_drop_percentage >= 30.0
    ):
        marginal = True
        reasons.append("sampled detection counts frequently drop suddenly")
        recommendations.append(
            "Inspect occlusion, compression, and lighting transitions."
        )

    status = "UNSUITABLE" if unsuitable else "MARGINAL" if marginal else "GOOD"
    if not reasons:
        reasons.append("sampled optical, geometry, and calibration checks passed")
        recommendations.append(
            "Proceed to annotated detection, tracking, and IN/OUT evaluation."
        )
    return status, tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(recommendations))


def analyze_video(
    source: Path,
    *,
    config: AppConfig,
    detector: DetectorBackend | None,
    fusion: DetectionFusion | None = None,
    sample_every: int = 15,
    maximum_samples: int = 120,
    edge_margin_pixels: int | None = None,
) -> CameraQualityReport:
    """Analyze bounded samples from a local video without storing source frames."""

    if sample_every < 1 or maximum_samples < 1:
        raise CameraQualityError("Sampling intervals and limits must be at least 1.")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise CameraQualityError(f"Could not open input video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise CameraQualityError("Input video reports invalid frame dimensions.")
    if fps <= 0 or not math.isfinite(fps):
        fps = 30.0
    view_inference = (
        MultiViewInference.from_configs(
            detector,
            resolve_inference_views(config.camera),
            width,
            height,
        )
        if detector is not None
        else None
    )
    calibration = FrameCalibration(config.camera, width, height)
    samples: list[FrameQualitySample] = []
    frame_number = 0
    started = time.perf_counter()
    margin = (
        config.diagnostics.edge_margin_pixels
        if edge_margin_pixels is None
        else edge_margin_pixels
    )
    try:
        while len(samples) < maximum_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % sample_every == 0:
                detections: list[Detection] = []
                if view_inference is not None:
                    raw = view_inference.collect(frame)
                    included = [
                        detection
                        for detection in raw
                        if not calibration.is_excluded(detection)
                    ]
                    detections = (
                        fusion.fuse(included) if fusion is not None else included
                    )
                samples.append(
                    measure_frame_quality(
                        frame,
                        detections,
                        frame_number=frame_number,
                        timestamp_seconds=frame_number / fps,
                        edge_margin_pixels=margin,
                    )
                )
            frame_number += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not samples:
        raise CameraQualityError("The input video did not yield any readable frames.")
    return _build_report(
        source,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        decoded_frames=frame_number,
        samples=samples,
        sample_every=sample_every,
        maximum_samples=maximum_samples,
        elapsed=elapsed,
        config=config,
        detection_enabled=detector is not None,
        device_label=getattr(detector, "device_label", None),
    )


def _build_report(
    source: Path,
    *,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    decoded_frames: int,
    samples: Sequence[FrameQualitySample],
    sample_every: int,
    maximum_samples: int,
    elapsed: float,
    config: AppConfig,
    detection_enabled: bool,
    device_label: str | None,
) -> CameraQualityReport:
    brightness = [sample.brightness_mean for sample in samples]
    sharpness = [sample.sharpness_laplacian_variance for sample in samples]
    underexposed = sum(
        sample.brightness_mean < 40 or sample.dark_pixel_percentage >= 50
        for sample in samples
    )
    overexposed = sum(
        sample.brightness_mean > 215 or sample.bright_pixel_percentage >= 50
        for sample in samples
    )
    blurred = sum(sample.sharpness_laplacian_variance < 60 for sample in samples)
    confidences = [value for sample in samples for value in sample.confidences]
    height_ratios = [
        value for sample in samples for value in sample.person_height_ratios
    ]
    detection_total = sum(sample.detection_count for sample in samples)
    edge_total = sum(sample.edge_clipped_detection_count for sample in samples)
    count_drop_opportunities = 0
    count_drops = 0
    for previous, current in zip(samples, samples[1:], strict=False):
        if previous.detection_count >= 2:
            count_drop_opportunities += 1
            count_drops += current.detection_count <= previous.detection_count / 2
    sample_count = len(samples)
    under_percentage = underexposed / sample_count * 100
    over_percentage = overexposed / sample_count * 100
    blurred_percentage = blurred / sample_count * 100
    edge_percentage = edge_total / detection_total * 100 if detection_total else None
    drop_percentage = (
        count_drops / count_drop_opportunities * 100
        if count_drop_opportunities
        else None
    )
    coverage = calibration_path_coverage(config.camera, config.zones)
    status, reasons, recommendations = classify_camera_quality(
        width=width,
        height=height,
        blurred_frame_percentage=blurred_percentage,
        underexposed_frame_percentage=under_percentage,
        overexposed_frame_percentage=over_percentage,
        complete_path_coverage=bool(
            coverage["combined_views_cover_outside_and_inside"]
        ),
        detection_enabled=detection_enabled,
        detection_total=detection_total,
        median_person_height_ratio=(
            statistics.median(height_ratios) if height_ratios else None
        ),
        edge_clipped_detection_percentage=edge_percentage,
        sudden_count_drop_percentage=drop_percentage,
    )
    duration = total_frames / fps if total_frames > 0 else decoded_frames / fps
    return CameraQualityReport(
        source=str(source),
        status=status,
        diagnostic_not_accuracy=True,
        reasons=reasons,
        recommendations=recommendations,
        video={
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "reported_frame_count": total_frames,
            "duration_seconds": round(duration, 3),
        },
        sampling={
            "sample_every_frames": sample_every,
            "maximum_samples": maximum_samples,
            "sampled_frames": sample_count,
            "decoded_frames": decoded_frames,
            "frame_numbers": [sample.frame_number for sample in samples],
        },
        optical_quality={
            "brightness_mean": round(statistics.mean(brightness), 3),
            "brightness_median": round(statistics.median(brightness), 3),
            "sharpness_median_laplacian_variance": round(
                statistics.median(sharpness), 3
            ),
            "blurred_frame_percentage": round(blurred_percentage, 2),
            "underexposed_frame_percentage": round(under_percentage, 2),
            "overexposed_frame_percentage": round(over_percentage, 2),
            "heuristic_thresholds": {
                "blur_laplacian_variance_below": 60,
                "underexposed_brightness_below": 40,
                "overexposed_brightness_above": 215,
            },
        },
        detection_diagnostics={
            "enabled": detection_enabled,
            "model": config.model.path if detection_enabled else None,
            "detector_floor": (
                config.model.effective_detector_floor if detection_enabled else None
            ),
            "confidence_reference": (
                config.model.confidence if detection_enabled else None
            ),
            "detection_total": detection_total,
            "frames_with_detections": sum(
                sample.detection_count > 0 for sample in samples
            ),
            "confidence_p10": _percentile(confidences, 10),
            "confidence_median": _percentile(confidences, 50),
            "confidence_p90": _percentile(confidences, 90),
            "median_person_height_ratio": (
                round(statistics.median(height_ratios), 4) if height_ratios else None
            ),
            "edge_clipped_detection_count": edge_total,
            "edge_clipped_detection_percentage": (
                round(edge_percentage, 2) if edge_percentage is not None else None
            ),
            "sudden_count_drop_percentage": (
                round(drop_percentage, 2) if drop_percentage is not None else None
            ),
        },
        calibration_coverage=coverage,
        performance={
            "device": device_label if detection_enabled else "not_used",
            "elapsed_seconds": round(elapsed, 3),
            "sample_pipeline_fps_including_decode": round(sample_count / elapsed, 2)
            if elapsed
            else 0.0,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.camera_quality",
        description=(
            "Sample a local camera/video and report heuristic suitability. This is "
            "diagnostic evidence, not detector or counting accuracy."
        ),
    )
    parser.add_argument("--source", required=True, type=Path, help="Local video")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Override YOLO weights")
    parser.add_argument("--imgsz", type=_image_size)
    parser.add_argument("--confidence", type=_probability)
    parser.add_argument("--detector-floor", type=_probability)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:N, or GPU index")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--sample-every", type=_positive_int, default=15)
    parser.add_argument("--max-samples", type=_positive_int, default=120)
    parser.add_argument("--edge-margin", type=_non_negative_int)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Measure only optical and calibration quality",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        source = args.source.expanduser().resolve()
        if not source.is_file():
            raise CameraQualityError(f"Input video does not exist: {source}")
        output = args.output.expanduser().resolve() if args.output else None
        if output is not None and output == source:
            raise CameraQualityError("Camera-quality JSON cannot overwrite the video.")
        config = load_config(args.config).with_overrides(
            device=args.device,
            confidence=args.confidence,
            detector_floor=args.detector_floor,
            model=args.model,
            imgsz=args.imgsz,
        )
        if args.precision is not None:
            config = replace(
                config,
                model=replace(
                    config.model,
                    precision=args.precision,
                    half=args.precision == "fp16",
                ),
            )
        detector: DetectorBackend | None = None
        fusion: DetectionFusion | None = None
        if not args.no_detection:
            device, device_label = resolve_device(config.device)
            detector = UltralyticsDetector(
                config.model.path,
                config.model.confidence,
                config.model.imgsz,
                detector_floor=config.model.effective_detector_floor,
                device=device,
                device_label=device_label,
                precision=config.model.precision,
            )
            fusion = NmsDetectionFusion(
                iou_threshold=config.detection_fusion.iou_threshold,
                confidence_strategy=config.detection_fusion.confidence_strategy,
                prefer_full_frame=config.detection_fusion.prefer_full_frame,
            )
        report = analyze_video(
            source,
            config=config,
            detector=detector,
            fusion=fusion,
            sample_every=args.sample_every,
            maximum_samples=args.max_samples,
            edge_margin_pixels=args.edge_margin,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        _print_report(report, output)
        return 0
    except (CameraQualityError, ConfigError, DetectorError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


def _print_report(report: CameraQualityReport, output: Path | None) -> None:
    print(f"TrackBus camera quality: {report.status}")
    for reason in report.reasons:
        print(f"  - {reason}")
    print("Recommendations:")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
    print("Note: this report is diagnostic evidence, not an accuracy measurement.")
    if output is not None:
        print(f"JSON report: {output}")


def _point_in_view(point: tuple[float, float], view: InferenceViewConfig) -> bool:
    x, y = point
    left, top, right, bottom = view.bounds
    return left <= x <= right and top <= y <= bottom


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), percentile)), 4)


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


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _image_size(value: str) -> int:
    parsed = int(value)
    if parsed < 32:
        raise argparse.ArgumentTypeError("must be at least 32 pixels")
    return parsed


if __name__ == "__main__":
    sys.exit(main())
