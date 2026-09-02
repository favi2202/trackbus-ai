"""Detector-only measurement for TrackBus person predictions.

This module deliberately does not import or invoke tracking, zones, counting, or
event logic.  It can therefore answer whether a failure starts at person
detection before downstream components are changed.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from trackbus.detection import BoundingBox, Detection
from trackbus.interfaces import DetectorBackend
from trackbus.preprocessing import preprocessing_latency_ms, preprocessing_summary


class DetectionBenchmarkError(ValueError):
    """Raised when benchmark inputs or annotations are invalid."""


@dataclass(frozen=True)
class GroundTruthPerson:
    """One manually labelled person box in an annotated frame."""

    bounding_box: BoundingBox
    annotation_id: str | None = None


@dataclass(frozen=True)
class DetectionGroundTruth:
    """Validated box annotations for a subset or all frames in one video."""

    coordinate_space: str
    frames: Mapping[int, tuple[GroundTruthPerson, ...]]
    video_filename: str | None = None
    source_width: int | None = None
    source_height: int | None = None

    def boxes_for(
        self, frame_number: int, frame_width: int, frame_height: int
    ) -> tuple[GroundTruthPerson, ...] | None:
        people = self.frames.get(frame_number)
        if people is None:
            return None
        if self.coordinate_space == "pixels":
            return people
        scaled: list[GroundTruthPerson] = []
        for person in people:
            left, top, right, bottom = person.bounding_box
            scaled.append(
                GroundTruthPerson(
                    (
                        left * frame_width,
                        top * frame_height,
                        right * frame_width,
                        bottom * frame_height,
                    ),
                    person.annotation_id,
                )
            )
        return tuple(scaled)


@dataclass(frozen=True)
class FrameDetectionMetrics:
    """Detector-only measurements for one decoded source frame."""

    frame: int
    timestamp_seconds: float
    detections: int
    confidence_minimum: float | None
    confidence_mean: float | None
    confidence_maximum: float | None
    low_confidence_detections: int
    edge_clipped_detections: int
    inference_latency_ms: float
    preprocessing_latency_ms: float
    previous_frame_matches: int
    previous_frame_mean_iou: float | None
    ground_truth_persons: int | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "detections": self.detections,
            "confidence_minimum": _rounded(self.confidence_minimum),
            "confidence_mean": _rounded(self.confidence_mean),
            "confidence_maximum": _rounded(self.confidence_maximum),
            "low_confidence_detections": self.low_confidence_detections,
            "edge_clipped_detections": self.edge_clipped_detections,
            "inference_latency_ms": round(self.inference_latency_ms, 4),
            "preprocessing_latency_ms": round(self.preprocessing_latency_ms, 4),
            "previous_frame_matches": self.previous_frame_matches,
            "previous_frame_mean_iou": _rounded(self.previous_frame_mean_iou),
            "ground_truth_persons": self.ground_truth_persons,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True)
class DetectionBenchmarkResult:
    """Summary and frame evidence produced by one detector configuration."""

    summary: Mapping[str, Any]
    frames: tuple[FrameDetectionMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "frames": [frame.to_dict() for frame in self.frames],
        }


def load_detection_ground_truth(path: Path) -> DetectionGroundTruth:
    """Load the TrackBus detector-box annotation schema from JSON."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DetectionBenchmarkError(
            f"Could not read detection ground truth '{path}': {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DetectionBenchmarkError(
            f"Detection ground truth is not valid JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise DetectionBenchmarkError("Detection ground truth root must be an object.")
    if document.get("schema_version") != 1:
        raise DetectionBenchmarkError(
            "Detection ground truth schema_version must be 1."
        )
    coordinate_space = document.get("coordinate_space", "pixels")
    if coordinate_space not in {"pixels", "normalized"}:
        raise DetectionBenchmarkError(
            "coordinate_space must be 'pixels' or 'normalized'."
        )
    video = document.get("video", {})
    if not isinstance(video, dict):
        raise DetectionBenchmarkError("video must be an object when supplied.")
    video_filename = _optional_text(video.get("filename"), "video.filename")
    source_width = _optional_positive_int(video.get("width"), "video.width")
    source_height = _optional_positive_int(video.get("height"), "video.height")

    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise DetectionBenchmarkError("frames must be a list.")
    frames: dict[int, tuple[GroundTruthPerson, ...]] = {}
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            raise DetectionBenchmarkError(f"frames[{index}] must be an object.")
        frame_number = _non_negative_int(
            raw_frame.get("frame"), f"frames[{index}].frame"
        )
        if frame_number in frames:
            raise DetectionBenchmarkError(
                f"Detection ground truth contains duplicate frame {frame_number}."
            )
        raw_people = raw_frame.get("persons", raw_frame.get("boxes"))
        if not isinstance(raw_people, list):
            raise DetectionBenchmarkError(
                f"frames[{index}].persons must be a list, including for empty frames."
            )
        people = tuple(
            _ground_truth_person(value, coordinate_space, index, person_index)
            for person_index, value in enumerate(raw_people)
        )
        frames[frame_number] = people
    return DetectionGroundTruth(
        coordinate_space=coordinate_space,
        frames=frames,
        video_filename=video_filename,
        source_width=source_width,
        source_height=source_height,
    )


def benchmark_detector(
    *,
    video_path: Path,
    detector: DetectorBackend,
    model_name: str,
    image_size: int,
    confidence_threshold: float,
    device_label: str,
    requested_precision: str = "fp32",
    effective_precision: str | None = None,
    ground_truth: DetectionGroundTruth | None = None,
    low_confidence_threshold: float = 0.35,
    matching_iou_threshold: float = 0.50,
    stability_iou_threshold: float = 0.10,
    edge_margin_pixels: int = 2,
    maximum_frames: int | None = None,
) -> DetectionBenchmarkResult:
    """Measure one detector without invoking any TrackBus downstream stage."""

    _validate_benchmark_options(
        confidence_threshold=confidence_threshold,
        low_confidence_threshold=low_confidence_threshold,
        matching_iou_threshold=matching_iou_threshold,
        stability_iou_threshold=stability_iou_threshold,
        edge_margin_pixels=edge_margin_pixels,
        maximum_frames=maximum_frames,
    )
    resolved_video = video_path.expanduser().resolve()
    if not resolved_video.is_file():
        raise DetectionBenchmarkError(f"Input video does not exist: {resolved_video}")
    capture = cv2.VideoCapture(str(resolved_video))
    if not capture.isOpened():
        raise DetectionBenchmarkError(f"Could not open input video: {resolved_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        capture.release()
        raise DetectionBenchmarkError(
            "Input video reported invalid dimensions "
            f"{width}x{height}: {resolved_video}"
        )
    _validate_ground_truth_video(ground_truth, resolved_video, width, height)

    frames: list[FrameDetectionMetrics] = []
    latencies: list[float] = []
    preprocessing_latencies: list[float] = []
    confidences: list[float] = []
    previous_detections: tuple[Detection, ...] = ()
    started = time.perf_counter()
    try:
        frame_number = 0
        while maximum_frames is None or frame_number < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            inference_started = time.perf_counter()
            detections = tuple(detector.detect(frame, source_view="full"))
            latency_ms = (time.perf_counter() - inference_started) * 1000.0
            preprocessing_ms = preprocessing_latency_ms(detector)
            _validate_detector_output(detections)
            frame_confidences = [detection.confidence for detection in detections]
            confidences.extend(frame_confidences)
            latencies.append(latency_ms)
            preprocessing_latencies.append(preprocessing_ms)
            previous_pairs = maximum_iou_matching(
                [detection.bounding_box for detection in previous_detections],
                [detection.bounding_box for detection in detections],
                threshold=stability_iou_threshold,
            )
            truth_people = (
                ground_truth.boxes_for(frame_number, width, height)
                if ground_truth is not None
                else None
            )
            tp: int | None = None
            fp: int | None = None
            fn: int | None = None
            if truth_people is not None:
                truth_pairs = maximum_iou_matching(
                    [person.bounding_box for person in truth_people],
                    [detection.bounding_box for detection in detections],
                    threshold=matching_iou_threshold,
                )
                tp = len(truth_pairs)
                fp = len(detections) - tp
                fn = len(truth_people) - tp
            frames.append(
                FrameDetectionMetrics(
                    frame=frame_number,
                    timestamp_seconds=(frame_number / fps if fps else 0.0),
                    detections=len(detections),
                    confidence_minimum=(
                        min(frame_confidences) if frame_confidences else None
                    ),
                    confidence_mean=(
                        statistics.fmean(frame_confidences)
                        if frame_confidences
                        else None
                    ),
                    confidence_maximum=(
                        max(frame_confidences) if frame_confidences else None
                    ),
                    low_confidence_detections=sum(
                        confidence < low_confidence_threshold
                        for confidence in frame_confidences
                    ),
                    edge_clipped_detections=sum(
                        _touches_edge(
                            detection.bounding_box,
                            frame_width=width,
                            frame_height=height,
                            margin=edge_margin_pixels,
                        )
                        for detection in detections
                    ),
                    inference_latency_ms=latency_ms,
                    preprocessing_latency_ms=preprocessing_ms,
                    previous_frame_matches=len(previous_pairs),
                    previous_frame_mean_iou=(
                        statistics.fmean(pair[2] for pair in previous_pairs)
                        if previous_pairs
                        else None
                    ),
                    ground_truth_persons=(
                        len(truth_people) if truth_people is not None else None
                    ),
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            )
            previous_detections = detections
            frame_number += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not frames:
        raise DetectionBenchmarkError(
            f"Input video contains no decoded frames: {resolved_video}"
        )
    summary = _build_summary(
        video_path=resolved_video,
        model_name=model_name,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
        requested_precision=requested_precision,
        effective_precision=effective_precision or requested_precision,
        low_confidence_threshold=low_confidence_threshold,
        matching_iou_threshold=matching_iou_threshold,
        stability_iou_threshold=stability_iou_threshold,
        device_label=device_label,
        fps=fps,
        width=width,
        height=height,
        reported_frames=reported_frames,
        elapsed=elapsed,
        confidences=confidences,
        latencies=latencies,
        preprocessing_latencies=preprocessing_latencies,
        preprocessing=preprocessing_summary(detector),
        frames=frames,
        ground_truth=ground_truth,
    )
    return DetectionBenchmarkResult(summary=summary, frames=tuple(frames))


def maximum_iou_matching(
    first_boxes: Sequence[BoundingBox],
    second_boxes: Sequence[BoundingBox],
    *,
    threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    """Return a deterministic maximum-cardinality IoU-threshold matching."""

    if not 0.0 <= threshold <= 1.0:
        raise DetectionBenchmarkError("IoU matching threshold must be in [0, 1].")
    neighbors: list[list[tuple[int, float]]] = []
    for first in first_boxes:
        candidates = [
            (index, bounding_box_iou(first, second))
            for index, second in enumerate(second_boxes)
        ]
        neighbors.append(
            sorted(
                (candidate for candidate in candidates if candidate[1] >= threshold),
                key=lambda candidate: (-candidate[1], candidate[0]),
            )
        )
    second_to_first: dict[int, int] = {}

    def augment(first_index: int, visited: set[int]) -> bool:
        for second_index, _iou in neighbors[first_index]:
            if second_index in visited:
                continue
            visited.add(second_index)
            previous = second_to_first.get(second_index)
            if previous is None or augment(previous, visited):
                second_to_first[second_index] = first_index
                return True
        return False

    for first_index in range(len(first_boxes)):
        augment(first_index, set())
    pairs = [
        (
            first_index,
            second_index,
            bounding_box_iou(first_boxes[first_index], second_boxes[second_index]),
        )
        for second_index, first_index in second_to_first.items()
    ]
    return tuple(sorted(pairs))


def bounding_box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection-over-union for two source-coordinate boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def write_benchmark_json(path: Path, result: DetectionBenchmarkResult) -> None:
    """Write the complete benchmark result as stable formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_frame_metrics_csv(
    path: Path, frames: Sequence[FrameDetectionMetrics]
) -> None:
    """Write auditable frame-level detector measurements."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [frame.to_dict() for frame in frames]
    if not rows:
        raise DetectionBenchmarkError("Cannot write an empty frame metrics CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    *,
    video_path: Path,
    model_name: str,
    image_size: int,
    confidence_threshold: float,
    requested_precision: str,
    effective_precision: str,
    low_confidence_threshold: float,
    matching_iou_threshold: float,
    stability_iou_threshold: float,
    device_label: str,
    fps: float,
    width: int,
    height: int,
    reported_frames: int,
    elapsed: float,
    confidences: Sequence[float],
    latencies: Sequence[float],
    preprocessing_latencies: Sequence[float],
    preprocessing: Mapping[str, Any],
    frames: Sequence[FrameDetectionMetrics],
    ground_truth: DetectionGroundTruth | None,
) -> dict[str, Any]:
    frame_count = len(frames)
    detection_count = sum(frame.detections for frame in frames)
    zero_streaks = _zero_detection_streaks(frames)
    matched_iou_values = [
        frame.previous_frame_mean_iou
        for frame in frames[1:]
        if frame.previous_frame_mean_iou is not None
    ]
    eligible_previous_boxes = sum(
        frames[index - 1].detections for index in range(1, frame_count)
    )
    eligible_current_boxes = sum(
        frames[index].detections for index in range(1, frame_count)
    )
    previous_matches = sum(frame.previous_frame_matches for frame in frames[1:])
    stability_denominator = max(eligible_previous_boxes, eligible_current_boxes)
    annotated = [frame for frame in frames if frame.ground_truth_persons is not None]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_type": "detector_only",
        "uses_tracking": False,
        "uses_counting": False,
        "metric_status": (
            "ground_truth_evaluated" if annotated else "proxy_only_no_box_ground_truth"
        ),
        "video": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "source_fps": round(fps, 6) if fps else None,
            "reported_frame_count": reported_frames,
            "processed_frame_count": frame_count,
        },
        "configuration": {
            "model": model_name,
            "image_size": image_size,
            "confidence_threshold": confidence_threshold,
            "detector_floor": confidence_threshold,
            "requested_precision": requested_precision,
            "effective_precision": effective_precision,
            "low_confidence_threshold": low_confidence_threshold,
            "device": device_label,
            "ground_truth_iou_threshold": matching_iou_threshold,
            "stability_iou_threshold": stability_iou_threshold,
            "preprocessing": dict(preprocessing),
        },
        "detections": {
            "total": detection_count,
            "average_per_frame": round(detection_count / frame_count, 6),
            "maximum_in_frame": max(frame.detections for frame in frames),
            "frames_with_detections": sum(frame.detections > 0 for frame in frames),
            "frames_without_detections": sum(frame.detections == 0 for frame in frames),
            "zero_detection_streak_count": len(zero_streaks),
            "maximum_zero_detection_streak_frames": max(zero_streaks, default=0),
            "zero_detection_streak_lengths": zero_streaks,
            "low_confidence_total": sum(
                frame.low_confidence_detections for frame in frames
            ),
            "edge_clipped_total": sum(
                frame.edge_clipped_detections for frame in frames
            ),
            "confidence": _distribution(confidences),
        },
        "frame_to_frame_box_stability_proxy": {
            "description": (
                "Best IoU matches between consecutive frames; this is not a "
                "tracking metric and can change because people move."
            ),
            "matched_boxes": previous_matches,
            "match_rate": (
                round(previous_matches / stability_denominator, 6)
                if stability_denominator
                else None
            ),
            "matched_iou": _distribution(matched_iou_values),
        },
        "performance": {
            "wall_time_seconds": round(elapsed, 6),
            "pipeline_fps_including_decode": round(frame_count / elapsed, 6)
            if elapsed
            else None,
            "inference_latency_ms": _distribution(latencies),
            "preprocessing_latency_ms": _distribution(preprocessing_latencies),
            "inference_latency_includes_preprocessing": True,
            "detector_fps_from_mean_latency": (
                round(1000.0 / statistics.fmean(latencies), 6)
                if latencies and statistics.fmean(latencies) > 0
                else None
            ),
        },
        "ground_truth": None,
        "limitations": [
            "Proxy metrics do not measure person recall without box-level labels.",
            "Event/counting accuracy is intentionally outside this benchmark.",
            "Frame-to-frame IoU is a stability proxy, not identity tracking.",
        ],
    }
    if annotated:
        tp = sum(frame.true_positives or 0 for frame in annotated)
        fp = sum(frame.false_positives or 0 for frame in annotated)
        fn = sum(frame.false_negatives or 0 for frame in annotated)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        summary["ground_truth"] = {
            "annotated_frames_processed": len(annotated),
            "annotated_persons": sum(
                frame.ground_truth_persons or 0 for frame in annotated
            ),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "missed_persons": fn,
            "precision": _rounded(precision),
            "recall": _rounded(recall),
            "f1": _rounded(f1),
        }
    return summary


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p90": round(_percentile(ordered, 0.90), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "maximum": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _zero_detection_streaks(
    frames: Sequence[FrameDetectionMetrics],
) -> list[int]:
    streaks: list[int] = []
    current = 0
    for frame in frames:
        if frame.detections == 0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _touches_edge(
    box: BoundingBox, *, frame_width: int, frame_height: int, margin: int
) -> bool:
    left, top, right, bottom = box
    return (
        left <= margin
        or top <= margin
        or right >= frame_width - 1 - margin
        or bottom >= frame_height - 1 - margin
    )


def _ground_truth_person(
    value: object,
    coordinate_space: str,
    frame_index: int,
    person_index: int,
) -> GroundTruthPerson:
    name = f"frames[{frame_index}].persons[{person_index}]"
    annotation_id: str | None = None
    raw_box = value
    if isinstance(value, dict):
        raw_box = value.get("bbox")
        annotation_id = _optional_text(value.get("id"), f"{name}.id")
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise DetectionBenchmarkError(f"{name}.bbox must contain four numbers.")
    try:
        box = tuple(float(component) for component in raw_box)
    except (TypeError, ValueError) as exc:
        raise DetectionBenchmarkError(
            f"{name}.bbox must contain four numbers."
        ) from exc
    if not all(math.isfinite(component) for component in box):
        raise DetectionBenchmarkError(f"{name}.bbox values must be finite.")
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise DetectionBenchmarkError(
            f"{name}.bbox must have positive width and height."
        )
    if coordinate_space == "normalized" and not all(
        0.0 <= component <= 1.0 for component in box
    ):
        raise DetectionBenchmarkError(
            f"{name}.bbox normalized values must be between 0 and 1."
        )
    return GroundTruthPerson(box, annotation_id)


def _validate_benchmark_options(
    *,
    confidence_threshold: float,
    low_confidence_threshold: float,
    matching_iou_threshold: float,
    stability_iou_threshold: float,
    edge_margin_pixels: int,
    maximum_frames: int | None,
) -> None:
    for name, value in {
        "confidence_threshold": confidence_threshold,
        "low_confidence_threshold": low_confidence_threshold,
    }.items():
        if not 0.0 < value <= 1.0:
            raise DetectionBenchmarkError(f"{name} must be in (0, 1].")
    for name, value in {
        "matching_iou_threshold": matching_iou_threshold,
        "stability_iou_threshold": stability_iou_threshold,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise DetectionBenchmarkError(f"{name} must be in [0, 1].")
    if edge_margin_pixels < 0:
        raise DetectionBenchmarkError("edge_margin_pixels cannot be negative.")
    if maximum_frames is not None and maximum_frames < 1:
        raise DetectionBenchmarkError("maximum_frames must be at least 1.")


def _validate_ground_truth_video(
    ground_truth: DetectionGroundTruth | None,
    video_path: Path,
    width: int,
    height: int,
) -> None:
    if ground_truth is None:
        return
    if ground_truth.video_filename not in {None, video_path.name}:
        raise DetectionBenchmarkError(
            "Ground-truth video filename does not match the benchmark input: "
            f"{ground_truth.video_filename!r} != {video_path.name!r}."
        )
    if ground_truth.source_width not in {None, width}:
        raise DetectionBenchmarkError(
            "Ground-truth video width "
            f"{ground_truth.source_width} does not match {width}."
        )
    if ground_truth.source_height not in {None, height}:
        raise DetectionBenchmarkError(
            "Ground-truth video height "
            f"{ground_truth.source_height} does not match {height}."
        )


def _validate_detector_output(detections: Sequence[Detection]) -> None:
    if any(detection.class_id != 0 for detection in detections):
        raise DetectionBenchmarkError(
            "Detector-only benchmark accepts person-class (class 0) predictions only."
        )


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DetectionBenchmarkError(f"{name} must be non-empty text when supplied.")
    return value.strip()


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    parsed = _non_negative_int(value, name)
    if parsed < 1:
        raise DetectionBenchmarkError(f"{name} must be at least 1.")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DetectionBenchmarkError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionBenchmarkError(
            f"{name} must be a non-negative integer."
        ) from exc
    if parsed < 0 or (isinstance(value, float) and not value.is_integer()):
        raise DetectionBenchmarkError(f"{name} must be a non-negative integer.")
    return parsed


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
