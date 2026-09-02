from __future__ import annotations

import csv
import json

import pytest

from trackbus.annotate_events import (
    AnnotatedEvent,
    EventDirection,
    GroundTruthAnnotations,
    save_annotations,
)
from trackbus.evaluation import (
    AGGREGATE_WARNING,
    EvaluationError,
    EventRecord,
    evaluate_aggregate,
    evaluate_events,
    evaluate_files,
    load_event_records,
    main,
    match_events,
)


def event(frame: int, direction: str, fps: float = 25.0) -> EventRecord:
    return EventRecord(frame, frame / fps, EventDirection.parse(direction))


class ObjectEvent:
    def __init__(self, frame: object, direction: str) -> None:
        self.frame = frame
        self.direction = direction


def test_event_matching_is_direction_aware_and_one_to_one() -> None:
    ground_truth = [event(100, "IN"), event(200, "OUT")]
    predictions = [event(102, "IN"), event(101, "IN"), event(202, "IN")]

    result = evaluate_events(ground_truth, predictions, tolerance_frames=5)

    assert result.overall.tp == 1
    assert result.overall.fp == 2
    assert result.overall.fn == 1
    assert result.matches[0].prediction_frame == 101
    assert result.by_direction[EventDirection.IN].precision == pytest.approx(1 / 3)
    assert result.by_direction[EventDirection.OUT].recall == 0.0


def test_matching_maximises_cardinality_then_minimises_timing_error() -> None:
    ground_truth = [event(10, "IN"), event(20, "IN")]
    predictions = [event(8, "IN"), event(13, "IN"), event(22, "IN")]

    matches = match_events(ground_truth, predictions, tolerance_frames=5)

    assert [
        (match.ground_truth_frame, match.prediction_frame) for match in matches
    ] == [
        (10, 8),
        (20, 22),
    ]


def test_precision_recall_f1_timing_and_count_errors() -> None:
    ground_truth = [event(100, "IN"), event(200, "IN"), event(300, "OUT")]
    predictions = [event(102, "IN"), event(250, "IN"), event(298, "OUT")]

    result = evaluate_events(ground_truth, predictions, tolerance_frames=5)

    assert result.overall.tp == 2
    assert result.overall.fp == 1
    assert result.overall.fn == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.overall.mean_timing_error == pytest.approx(0.0)
    assert result.overall.mean_absolute_timing_error == pytest.approx(2.0)
    assert result.count_errors.entered == 0
    assert result.count_errors.exited == 0
    assert result.count_errors.total_crossings == 0


def test_seconds_tolerance_uses_timestamps_not_frames() -> None:
    ground_truth = [EventRecord(100, 4.0, EventDirection.IN)]
    predictions = [EventRecord(500, 4.08, EventDirection.IN)]

    result = evaluate_events(
        ground_truth,
        predictions,
        tolerance_seconds=0.1,
    )

    assert result.overall.tp == 1
    assert result.overall.timing_error_unit == "seconds"
    assert result.overall.mean_absolute_timing_error == pytest.approx(0.08)


def test_match_rejects_two_simultaneous_tolerance_modes() -> None:
    with pytest.raises(EvaluationError, match="choose"):
        match_events([], [], tolerance_frames=2, tolerance_seconds=0.2)


def test_empty_frame_level_evaluation_is_explicit_and_finite() -> None:
    result = evaluate_events([], [], tolerance_frames=0)

    assert result.overall.tp == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.overall.mean_timing_error is None


def test_aggregate_evaluation_does_not_claim_precision_or_recall() -> None:
    result = evaluate_aggregate(
        expected_entered=6,
        expected_exited=2,
        predicted_entered=3,
        predicted_exited=2,
    )

    report = result.to_dict()
    assert report["evaluation_mode"] == "aggregate_only"
    assert report["sufficient_for_precision_recall"] is False
    assert report["overall"]["precision"] is None
    assert result.count_errors.entered == 3
    assert result.count_errors.exited == 0
    assert result.count_errors.total_crossings == 3
    assert result.warning == AGGREGATE_WARNING


def test_load_event_records_accepts_existing_trackbus_csv(tmp_path) -> None:
    path = tmp_path / "predicted.events.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "video_frame", "event_type"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "00:00:04.080",
                "video_frame": "102",
                "event_type": "IN",
            }
        )

    records = load_event_records(path)

    assert records == [EventRecord(102, 4.08, EventDirection.IN)]


def test_evaluate_files_uses_ground_truth_fps_for_missing_csv_timestamp(
    tmp_path,
) -> None:
    ground_truth_path = tmp_path / "truth.json"
    predictions_path = tmp_path / "predictions.csv"
    save_annotations(
        ground_truth_path,
        GroundTruthAnnotations(
            "door.mp4",
            source_fps=25,
            events=(AnnotatedEvent(100, 4.0, EventDirection.OUT),),
        ),
    )
    predictions_path.write_text(
        "video_frame,event_type\n101,OUT\n",
        encoding="utf-8",
    )

    result = evaluate_files(
        ground_truth_path,
        predictions_path,
        tolerance_frames=2,
    )

    assert result.overall.tp == 1
    assert result.matches[0].prediction_timestamp == pytest.approx(4.04)


def test_aggregate_cli_prints_and_writes_json(tmp_path, capsys) -> None:
    output_path = tmp_path / "metrics.json"

    exit_code = main(
        [
            "--expected-entered",
            "6",
            "--expected-exited",
            "2",
            "--predicted-entered",
            "3",
            "--predicted-exited",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["count_errors"]["absolute_in_count_error"] == 3
    assert json.loads(capsys.readouterr().out) == report


def test_invalid_aggregate_count_is_rejected() -> None:
    with pytest.raises(EvaluationError, match="non-negative"):
        evaluate_aggregate(
            expected_entered=-1,
            expected_exited=0,
            predicted_entered=0,
            predicted_exited=0,
        )


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_object_event_requires_finite_positive_fps(fps: float) -> None:
    with pytest.raises(EvaluationError, match="finite positive FPS"):
        evaluate_events(
            [ObjectEvent(10, "IN")],
            [],
            tolerance_frames=1,
            fps=fps,
        )


def test_object_event_does_not_truncate_non_integral_frames() -> None:
    with pytest.raises(EvaluationError, match="must be an integer"):
        evaluate_events(
            [ObjectEvent(10.5, "IN")],
            [],
            tolerance_frames=1,
            fps=25,
        )
