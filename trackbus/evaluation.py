"""Direction-aware evaluation of predicted TrackBus crossing events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from trackbus.annotate_events import (
    AnnotatedEvent,
    AnnotationError,
    EventDirection,
    GroundTruthAnnotations,
    load_annotations,
)


class EvaluationError(ValueError):
    """Raised when event inputs or evaluation options are invalid."""


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Model-independent event representation accepted by the evaluator."""

    frame: int
    timestamp: float
    direction: EventDirection
    note: str | None = None

    def __post_init__(self) -> None:
        try:
            validated = AnnotatedEvent(
                frame=self.frame,
                timestamp=self.timestamp,
                direction=self.direction,
                note=self.note,
            )
        except AnnotationError as exc:
            raise EvaluationError(str(exc)) from exc
        object.__setattr__(self, "frame", validated.frame)
        object.__setattr__(self, "timestamp", validated.timestamp)
        object.__setattr__(self, "direction", validated.direction)
        object.__setattr__(self, "note", validated.note)

    @classmethod
    def from_annotated(cls, event: AnnotatedEvent) -> EventRecord:
        return cls(event.frame, event.timestamp, event.direction, event.note)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "direction": self.direction.value,
        }
        if self.note is not None:
            result["note"] = self.note
        return result


@dataclass(frozen=True, slots=True)
class EventMatch:
    """One ground-truth event paired one-to-one with one prediction."""

    direction: EventDirection
    ground_truth_index: int
    prediction_index: int
    ground_truth_frame: int
    prediction_frame: int
    ground_truth_timestamp: float
    prediction_timestamp: float
    timing_error: float
    timing_unit: str

    @property
    def absolute_timing_error(self) -> float:
        return abs(self.timing_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "ground_truth_index": self.ground_truth_index,
            "prediction_index": self.prediction_index,
            "ground_truth_frame": self.ground_truth_frame,
            "prediction_frame": self.prediction_frame,
            "ground_truth_timestamp": self.ground_truth_timestamp,
            "prediction_timestamp": self.prediction_timestamp,
            "timing_error": self.timing_error,
            "absolute_timing_error": self.absolute_timing_error,
            "timing_unit": self.timing_unit,
        }


@dataclass(frozen=True, slots=True)
class EventMetrics:
    """Precision/recall summary for a complete set or one direction."""

    true_positive_events: int
    false_positive_events: int
    false_negative_events: int
    precision: float
    recall: float
    f1: float
    mean_timing_error: float | None
    mean_absolute_timing_error: float | None
    timing_error_unit: str

    @property
    def tp(self) -> int:
        return self.true_positive_events

    @property
    def fp(self) -> int:
        return self.false_positive_events

    @property
    def fn(self) -> int:
        return self.false_negative_events

    def to_dict(self) -> dict[str, Any]:
        result = {
            "true_positive_events": self.true_positive_events,
            "false_positive_events": self.false_positive_events,
            "false_negative_events": self.false_negative_events,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_timing_error": self.mean_timing_error,
            "mean_absolute_timing_error": self.mean_absolute_timing_error,
            "timing_error_unit": self.timing_error_unit,
        }
        result[f"mean_timing_error_{self.timing_error_unit}"] = self.mean_timing_error
        result[f"mean_absolute_timing_error_{self.timing_error_unit}"] = (
            self.mean_absolute_timing_error
        )
        return result


@dataclass(frozen=True, slots=True)
class CountErrors:
    """Absolute differences between expected and predicted event totals."""

    entered: int
    exited: int
    total_crossings: int

    @property
    def aggregate(self) -> int:
        """Sum of direction-specific errors, useful for experiment ranking."""

        return self.entered + self.exited

    def to_dict(self) -> dict[str, int]:
        return {
            "absolute_in_count_error": self.entered,
            "absolute_out_count_error": self.exited,
            "total_crossing_count_error": self.total_crossings,
            "aggregate_count_error": self.aggregate,
            "aggregate_direction_count_error": self.aggregate,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Full frame-level, direction-aware event evaluation result."""

    overall: EventMetrics
    by_direction: Mapping[EventDirection, EventMetrics]
    matches: tuple[EventMatch, ...]
    expected_entered: int
    expected_exited: int
    predicted_entered: int
    predicted_exited: int
    count_errors: CountErrors
    tolerance_value: float
    tolerance_unit: str

    @property
    def precision(self) -> float:
        return self.overall.precision

    @property
    def recall(self) -> float:
        return self.overall.recall

    @property
    def f1(self) -> float:
        return self.overall.f1

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_mode": "event",
            "aggregate_only": False,
            "tolerance": {
                "value": self.tolerance_value,
                "unit": self.tolerance_unit,
            },
            "overall": self.overall.to_dict(),
            "by_direction": {
                direction.value: self.by_direction[direction].to_dict()
                for direction in (EventDirection.IN, EventDirection.OUT)
            },
            "counts": {
                "expected": {
                    "entered": self.expected_entered,
                    "exited": self.expected_exited,
                    "total": self.expected_entered + self.expected_exited,
                },
                "predicted": {
                    "entered": self.predicted_entered,
                    "exited": self.predicted_exited,
                    "total": self.predicted_entered + self.predicted_exited,
                },
            },
            "count_errors": self.count_errors.to_dict(),
            "matches": [match.to_dict() for match in self.matches],
        }


AGGREGATE_WARNING = (
    "Aggregate totals cannot measure event precision, recall, F1, duplicate events, "
    "missed-and-replaced events, or timing error. Add frame-level ground truth."
)


@dataclass(frozen=True, slots=True)
class AggregateEvaluationResult:
    """Count-error-only result when no frame-level labels are available."""

    expected_entered: int
    expected_exited: int
    predicted_entered: int
    predicted_exited: int
    count_errors: CountErrors
    warning: str = AGGREGATE_WARNING

    @property
    def precision(self) -> None:
        return None

    @property
    def recall(self) -> None:
        return None

    @property
    def f1(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_mode": "aggregate_only",
            "aggregate_only": True,
            "sufficient_for_precision_recall": False,
            "warning": self.warning,
            "overall": {
                "true_positive_events": None,
                "false_positive_events": None,
                "false_negative_events": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "mean_timing_error": None,
                "mean_absolute_timing_error": None,
                "timing_error_unit": None,
            },
            "counts": {
                "expected": {
                    "entered": self.expected_entered,
                    "exited": self.expected_exited,
                    "total": self.expected_entered + self.expected_exited,
                },
                "predicted": {
                    "entered": self.predicted_entered,
                    "exited": self.predicted_exited,
                    "total": self.predicted_entered + self.predicted_exited,
                },
            },
            "count_errors": self.count_errors.to_dict(),
        }


def _validate_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be a non-negative integer")
    return value


def _count_errors(
    expected_entered: int,
    expected_exited: int,
    predicted_entered: int,
    predicted_exited: int,
) -> CountErrors:
    return CountErrors(
        entered=abs(predicted_entered - expected_entered),
        exited=abs(predicted_exited - expected_exited),
        total_crossings=abs(
            (predicted_entered + predicted_exited)
            - (expected_entered + expected_exited)
        ),
    )


def evaluate_aggregate(
    *,
    expected_entered: int,
    expected_exited: int,
    predicted_entered: int,
    predicted_exited: int,
) -> AggregateEvaluationResult:
    """Evaluate total-count errors without claiming event-level accuracy."""

    expected_entered = _validate_count("expected_entered", expected_entered)
    expected_exited = _validate_count("expected_exited", expected_exited)
    predicted_entered = _validate_count("predicted_entered", predicted_entered)
    predicted_exited = _validate_count("predicted_exited", predicted_exited)
    return AggregateEvaluationResult(
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        predicted_entered=predicted_entered,
        predicted_exited=predicted_exited,
        count_errors=_count_errors(
            expected_entered,
            expected_exited,
            predicted_entered,
            predicted_exited,
        ),
    )


def _coerce_event(value: object, fps: float | None = None) -> EventRecord:
    if isinstance(value, EventRecord):
        return value
    if isinstance(value, AnnotatedEvent):
        return EventRecord.from_annotated(value)
    if isinstance(value, Mapping):
        return _event_from_mapping(value, fps=fps)
    # This also accepts CountEvent-like objects without importing the counter.
    frame = getattr(value, "frame", getattr(value, "video_frame", None))
    direction = getattr(value, "direction", getattr(value, "event_type", None))
    timestamp = getattr(value, "timestamp", None)
    if frame is not None and direction is not None:
        if isinstance(frame, bool) or not isinstance(frame, Integral):
            raise EvaluationError("event frame/video_frame must be an integer")
        parsed_frame = int(frame)
        if timestamp is None:
            if fps is None or not math.isfinite(fps) or fps <= 0:
                raise EvaluationError(
                    "event has no timestamp; supply a finite positive FPS to derive it"
                )
            timestamp = parsed_frame / fps
        return EventRecord(
            frame=parsed_frame,
            timestamp=float(timestamp),
            direction=EventDirection.parse(direction),
            note=getattr(value, "note", None),
        )
    raise EvaluationError(f"unsupported event value: {value!r}")


def _parse_timestamp(value: object) -> float:
    if isinstance(value, bool):
        raise EvaluationError("timestamp must be seconds or HH:MM:SS.sss")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if ":" not in stripped:
            try:
                result = float(stripped)
            except ValueError as exc:
                raise EvaluationError(f"invalid timestamp: {value!r}") from exc
        else:
            parts = stripped.split(":")
            if len(parts) != 3:
                raise EvaluationError(f"invalid timestamp: {value!r}")
            try:
                hours, minutes, seconds = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                )
            except ValueError as exc:
                raise EvaluationError(f"invalid timestamp: {value!r}") from exc
            if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
                raise EvaluationError(f"invalid timestamp: {value!r}")
            result = hours * 3600 + minutes * 60 + seconds
    else:
        raise EvaluationError("timestamp must be seconds or HH:MM:SS.sss")
    if not math.isfinite(result) or result < 0:
        raise EvaluationError("timestamp must be finite and non-negative")
    return result


def _event_from_mapping(
    value: Mapping[str, object], fps: float | None = None
) -> EventRecord:
    frame = value.get("frame", value.get("video_frame"))
    direction = value.get("direction", value.get("event_type"))
    if isinstance(frame, bool) or not isinstance(frame, (int, str)):
        raise EvaluationError("event frame/video_frame must be an integer")
    try:
        parsed_frame = int(frame)
    except ValueError as exc:
        raise EvaluationError(f"invalid event frame: {frame!r}") from exc
    if direction is None:
        raise EvaluationError("event is missing direction/event_type")
    timestamp_value = value.get("timestamp")
    if timestamp_value in (None, ""):
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise EvaluationError(
                "event is missing timestamp; supply a positive FPS to derive it"
            )
        timestamp = parsed_frame / fps
    else:
        timestamp = _parse_timestamp(timestamp_value)
    note = value.get("note")
    if note is not None and not isinstance(note, str):
        raise EvaluationError("event note must be text")
    try:
        parsed_direction = EventDirection.parse(direction)
    except AnnotationError as exc:
        raise EvaluationError(str(exc)) from exc
    return EventRecord(parsed_frame, timestamp, parsed_direction, note)


def load_event_records(path: Path, *, fps: float | None = None) -> list[EventRecord]:
    """Load predictions or labels from TrackBus event JSON or event CSV."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvaluationError(f"event CSV has no header: {path}")
            try:
                return [_event_from_mapping(row, fps=fps) for row in reader]
            except EvaluationError as exc:
                raise EvaluationError(f"invalid event CSV {path}: {exc}") from exc
    if suffix != ".json":
        raise EvaluationError("event input must be a .json or .csv file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {path}: {exc}") from exc
    raw_events: object
    document_fps = fps
    if isinstance(raw, list):
        raw_events = raw
    elif isinstance(raw, Mapping):
        raw_events = raw.get("events")
        possible_fps = raw.get("source_fps", raw.get("fps"))
        if document_fps is None and isinstance(possible_fps, (int, float)):
            document_fps = float(possible_fps)
    else:
        raise EvaluationError("event JSON root must be an object or event list")
    if not isinstance(raw_events, list):
        raise EvaluationError("event JSON must contain an events list")
    records: list[EventRecord] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise EvaluationError(f"event {index} must be an object")
        try:
            records.append(_event_from_mapping(raw_event, fps=document_fps))
        except EvaluationError as exc:
            raise EvaluationError(f"invalid event {index} in {path}: {exc}") from exc
    return records


def _metric_value(event: EventRecord, timing_unit: str) -> float:
    return float(event.frame) if timing_unit == "frames" else event.timestamp


def _optimal_pairs(
    ground_truth: list[tuple[int, EventRecord]],
    predictions: list[tuple[int, EventRecord]],
    *,
    tolerance: float,
    timing_unit: str,
) -> tuple[tuple[int, int], ...]:
    """Maximise matches, then minimise total timing error for sorted 1-D events."""

    ground_truth = sorted(
        ground_truth,
        key=lambda item: (_metric_value(item[1], timing_unit), item[0]),
    )
    predictions = sorted(
        predictions,
        key=lambda item: (_metric_value(item[1], timing_unit), item[0]),
    )
    rows, columns = len(ground_truth), len(predictions)
    empty: tuple[tuple[int, int], ...] = ()
    table = [[empty for _ in range(columns + 1)] for _ in range(rows + 1)]

    def rank(pairs: tuple[tuple[int, int], ...]) -> tuple[int, float]:
        total_error = sum(
            abs(
                _metric_value(ground_truth[i][1], timing_unit)
                - _metric_value(predictions[j][1], timing_unit)
            )
            for i, j in pairs
        )
        return len(pairs), -total_error

    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            candidates = [table[i + 1][j], table[i][j + 1]]
            distance = abs(
                _metric_value(ground_truth[i][1], timing_unit)
                - _metric_value(predictions[j][1], timing_unit)
            )
            if distance <= tolerance:
                candidates.append(((i, j),) + table[i + 1][j + 1])
            table[i][j] = max(candidates, key=rank)
    return tuple((ground_truth[i][0], predictions[j][0]) for i, j in table[0][0])


def match_events(
    ground_truth: Iterable[object],
    predictions: Iterable[object],
    *,
    tolerance_frames: int | None = None,
    tolerance_seconds: float | None = None,
    fps: float | None = None,
) -> tuple[EventMatch, ...]:
    """Match same-direction events one-to-one within a temporal tolerance.

    Matching maximises the number of true positives and then minimises total
    absolute timing error.  This prevents input ordering from changing results.
    """

    truth = [_coerce_event(event, fps=fps) for event in ground_truth]
    predicted = [_coerce_event(event, fps=fps) for event in predictions]
    if tolerance_frames is not None and tolerance_seconds is not None:
        raise EvaluationError("choose tolerance_frames or tolerance_seconds, not both")
    if tolerance_frames is None and tolerance_seconds is None:
        tolerance_frames = 25
    if tolerance_seconds is not None:
        if (
            isinstance(tolerance_seconds, bool)
            or not isinstance(tolerance_seconds, (int, float))
            or not math.isfinite(float(tolerance_seconds))
            or tolerance_seconds < 0
        ):
            raise EvaluationError("tolerance_seconds must be finite and non-negative")
        tolerance = float(tolerance_seconds)
        timing_unit = "seconds"
    else:
        if (
            isinstance(tolerance_frames, bool)
            or not isinstance(tolerance_frames, int)
            or tolerance_frames < 0
        ):
            raise EvaluationError("tolerance_frames must be a non-negative integer")
        tolerance = float(tolerance_frames)
        timing_unit = "frames"

    matches: list[EventMatch] = []
    for direction in (EventDirection.IN, EventDirection.OUT):
        direction_truth = [
            (index, event)
            for index, event in enumerate(truth)
            if event.direction is direction
        ]
        direction_predictions = [
            (index, event)
            for index, event in enumerate(predicted)
            if event.direction is direction
        ]
        for truth_index, prediction_index in _optimal_pairs(
            direction_truth,
            direction_predictions,
            tolerance=tolerance,
            timing_unit=timing_unit,
        ):
            truth_event = truth[truth_index]
            prediction_event = predicted[prediction_index]
            timing_error = _metric_value(prediction_event, timing_unit) - _metric_value(
                truth_event, timing_unit
            )
            matches.append(
                EventMatch(
                    direction=direction,
                    ground_truth_index=truth_index,
                    prediction_index=prediction_index,
                    ground_truth_frame=truth_event.frame,
                    prediction_frame=prediction_event.frame,
                    ground_truth_timestamp=truth_event.timestamp,
                    prediction_timestamp=prediction_event.timestamp,
                    timing_error=timing_error,
                    timing_unit=timing_unit,
                )
            )
    return tuple(
        sorted(matches, key=lambda match: (match.ground_truth_frame, match.direction))
    )


def _metrics(
    *,
    ground_truth_count: int,
    prediction_count: int,
    matches: Sequence[EventMatch],
    timing_unit: str,
) -> EventMetrics:
    true_positives = len(matches)
    false_positives = prediction_count - true_positives
    false_negatives = ground_truth_count - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_error = (
        sum(match.timing_error for match in matches) / true_positives
        if true_positives
        else None
    )
    mean_absolute_error = (
        sum(match.absolute_timing_error for match in matches) / true_positives
        if true_positives
        else None
    )
    return EventMetrics(
        true_positive_events=true_positives,
        false_positive_events=false_positives,
        false_negative_events=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_timing_error=mean_error,
        mean_absolute_timing_error=mean_absolute_error,
        timing_error_unit=timing_unit,
    )


def evaluate_events(
    ground_truth: Iterable[object],
    predictions: Iterable[object],
    *,
    tolerance_frames: int | None = None,
    tolerance_seconds: float | None = None,
    fps: float | None = None,
) -> EvaluationResult:
    """Calculate event and count metrics from frame-level ground truth."""

    truth = tuple(_coerce_event(event, fps=fps) for event in ground_truth)
    predicted = tuple(_coerce_event(event, fps=fps) for event in predictions)
    matches = match_events(
        truth,
        predicted,
        tolerance_frames=tolerance_frames,
        tolerance_seconds=tolerance_seconds,
    )
    if tolerance_seconds is not None:
        tolerance_value = float(tolerance_seconds)
        timing_unit = "seconds"
    else:
        tolerance_value = float(25 if tolerance_frames is None else tolerance_frames)
        timing_unit = "frames"

    by_direction: dict[EventDirection, EventMetrics] = {}
    for direction in (EventDirection.IN, EventDirection.OUT):
        direction_matches = tuple(
            match for match in matches if match.direction is direction
        )
        by_direction[direction] = _metrics(
            ground_truth_count=sum(event.direction is direction for event in truth),
            prediction_count=sum(event.direction is direction for event in predicted),
            matches=direction_matches,
            timing_unit=timing_unit,
        )
    overall = _metrics(
        ground_truth_count=len(truth),
        prediction_count=len(predicted),
        matches=matches,
        timing_unit=timing_unit,
    )
    expected_entered = sum(event.direction is EventDirection.IN for event in truth)
    expected_exited = sum(event.direction is EventDirection.OUT for event in truth)
    predicted_entered = sum(event.direction is EventDirection.IN for event in predicted)
    predicted_exited = sum(event.direction is EventDirection.OUT for event in predicted)
    return EvaluationResult(
        overall=overall,
        by_direction=by_direction,
        matches=matches,
        expected_entered=expected_entered,
        expected_exited=expected_exited,
        predicted_entered=predicted_entered,
        predicted_exited=predicted_exited,
        count_errors=_count_errors(
            expected_entered,
            expected_exited,
            predicted_entered,
            predicted_exited,
        ),
        tolerance_value=tolerance_value,
        tolerance_unit=timing_unit,
    )


def evaluate_files(
    ground_truth_path: Path,
    predictions_path: Path,
    *,
    tolerance_frames: int | None = None,
    tolerance_seconds: float | None = None,
) -> EvaluationResult:
    """Load files and perform frame-level evaluation."""

    try:
        document: GroundTruthAnnotations = load_annotations(ground_truth_path)
    except AnnotationError as exc:
        raise EvaluationError(str(exc)) from exc
    ground_truth = [EventRecord.from_annotated(event) for event in document.events]
    predictions = load_event_records(predictions_path, fps=document.source_fps)
    return evaluate_events(
        ground_truth,
        predictions,
        tolerance_frames=tolerance_frames,
        tolerance_seconds=tolerance_seconds,
    )


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.evaluation",
        description=(
            "Evaluate predicted crossing events against frame-level labels, or "
            "report aggregate count errors when labels do not yet exist."
        ),
    )
    parser.add_argument("--ground-truth", type=Path, help="Annotated events JSON")
    parser.add_argument("--predictions", type=Path, help="Predicted events JSON or CSV")
    tolerance = parser.add_mutually_exclusive_group()
    tolerance.add_argument(
        "--tolerance-frames",
        type=_non_negative_int,
        help="Maximum frame difference for a match (default: 25)",
    )
    tolerance.add_argument(
        "--tolerance-seconds",
        type=_non_negative_float,
        help="Maximum timestamp difference for a match",
    )
    parser.add_argument("--expected-entered", type=_non_negative_int)
    parser.add_argument("--expected-exited", type=_non_negative_int)
    parser.add_argument("--predicted-entered", type=_non_negative_int)
    parser.add_argument("--predicted-exited", type=_non_negative_int)
    parser.add_argument("--output", type=Path, help="Optional metrics JSON output")
    return parser


def _run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.ground_truth is not None or args.predictions is not None:
        if args.ground_truth is None or args.predictions is None:
            raise EvaluationError(
                "--ground-truth and --predictions must be supplied together"
            )
        if any(
            value is not None
            for value in (
                args.expected_entered,
                args.expected_exited,
                args.predicted_entered,
                args.predicted_exited,
            )
        ):
            raise EvaluationError(
                "aggregate count options cannot be mixed with event input files"
            )
        result = evaluate_files(
            args.ground_truth.expanduser().resolve(),
            args.predictions.expanduser().resolve(),
            tolerance_frames=args.tolerance_frames,
            tolerance_seconds=args.tolerance_seconds,
        )
        return result.to_dict()

    aggregate_values = (
        args.expected_entered,
        args.expected_exited,
        args.predicted_entered,
        args.predicted_exited,
    )
    if any(value is None for value in aggregate_values):
        raise EvaluationError(
            "supply event files, or all four aggregate options: --expected-entered, "
            "--expected-exited, --predicted-entered, and --predicted-exited"
        )
    if args.tolerance_frames is not None or args.tolerance_seconds is not None:
        raise EvaluationError(
            "temporal tolerance is unavailable in aggregate-only mode"
        )
    result = evaluate_aggregate(
        expected_entered=args.expected_entered,
        expected_exited=args.expected_exited,
        predicted_entered=args.predicted_entered,
        predicted_exited=args.predicted_exited,
    )
    return result.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _run_from_args(args)
        serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    except (EvaluationError, AnnotationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
