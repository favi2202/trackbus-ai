from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from trackbus.benchmark_report import (
    BenchmarkReportError,
    build_benchmark_report,
    main,
    write_benchmark_report,
)


def detection_artifact(*, with_truth: bool = True) -> dict[str, object]:
    truth = (
        {
            "precision": 0.8,
            "recall": 0.6,
            "f1": 0.685714,
            "true_positives": 6,
            "false_positives": 2,
            "missed_persons": 4,
        }
        if with_truth
        else None
    )
    return {
        "summary": {
            "benchmark_type": "detector_only",
            "metric_status": (
                "ground_truth_evaluated"
                if with_truth
                else "proxy_only_no_box_ground_truth"
            ),
            "configuration": {
                "model": "yolo11n.pt",
                "image_size": 640,
                "detector_floor": 0.1,
                "device": "cpu",
                "preprocessing": {"profile": "none"},
            },
            "detections": {"total": 8, "frames_without_detections": 2},
            "frame_to_frame_box_stability_proxy": {"match_rate": 0.75},
            "performance": {
                "pipeline_fps_including_decode": 18.5,
                "inference_latency_ms": {"mean": 48.0},
                "preprocessing_latency_ms": {"mean": 0.0},
            },
            "ground_truth": truth,
        }
    }


def processing_summary() -> dict[str, object]:
    return {
        "model_used": "yolo11n.pt",
        "confidence_threshold": 0.35,
        "detector_floor": 0.1,
        "inference_image_size": 640,
        "model_precision": "fp32",
        "device_used": "cpu",
        "tracker_backend": "ByteTrackAdapter",
        "tracker_config_path": "configs/tracking_balanced.yaml",
        "tracker_effective_config": {"track_buffer": 45},
        "tracking_continuity": {
            "enabled": True,
            "stitched_track_fragments": 2,
            "maximum_stitched_gap_frames": 3,
        },
        "unique_tracking_ids": 4,
        "possible_id_restart_count": 1,
        "entered_total": 3,
        "exited_total": 2,
        "average_fps": 16.2,
        "processing_time_seconds": 6.1,
        "processed_frames": 100,
        "track_diagnostics": [
            {
                "first_observed_frame": 1,
                "last_observed_frame": 20,
                "observed_frames": 18,
                "left_lane_detection_gap_frames": 2,
                "unassigned_detection_gap_frames": 1,
            },
            {
                "first_observed_frame": 10,
                "last_observed_frame": 39,
                "observed_frames": 25,
                "left_lane_detection_gap_frames": 0,
                "unassigned_detection_gap_frames": 0,
            },
        ],
    }


def counting_evaluation(*, event_ground_truth: bool = True) -> dict[str, object]:
    return {
        "evaluation_mode": "event" if event_ground_truth else "aggregate_only",
        "aggregate_only": not event_ground_truth,
        "overall": {
            "true_positive_events": 4 if event_ground_truth else None,
            "false_positive_events": 1 if event_ground_truth else None,
            "false_negative_events": 2 if event_ground_truth else None,
            "precision": 0.8 if event_ground_truth else None,
            "recall": 2 / 3 if event_ground_truth else None,
            "f1": 0.727273 if event_ground_truth else None,
        },
        "counts": {
            "expected": {"entered": 4, "exited": 2},
            "predicted": {"entered": 3, "exited": 2},
        },
        "count_errors": {"aggregate_count_error": 1},
    }


def test_report_keeps_detection_tracking_counting_and_runtime_separate() -> None:
    report = build_benchmark_report(
        detection_artifact=detection_artifact(),
        processing_summary=processing_summary(),
        counting_evaluation=counting_evaluation(),
    )

    assert report["detection"]["status"] == "ground_truth_evaluated"
    assert report["detection"]["f1"] == 0.685714
    assert report["tracking"]["status"].startswith("diagnostics_only")
    assert report["tracking"]["id_switches"] is None
    assert report["tracking"]["fragmentation_proxy"]["diagnostic_gap_frames"] == 3
    assert report["tracking"]["track_lifetime_frames"]["p50"] == 25
    assert report["counting"]["status"] == "ground_truth_evaluated"
    assert report["counting"]["f1"] == 0.727273
    assert report["runtime"]["detector"]["pipeline_fps_including_decode"] == 18.5
    assert report["runtime"]["end_to_end"]["average_fps"] == 16.2
    assert report["metric_contract"]["event_f1_is_not_detector_f1"] is True
    assert report["metric_contract"]["ground_truth_accuracy_sections"] == [
        "detection",
        "counting",
    ]


def test_missing_ground_truth_is_labeled_diagnostics_not_accuracy() -> None:
    report = build_benchmark_report(
        detection_artifact=detection_artifact(with_truth=False),
        processing_summary=processing_summary(),
        counting_evaluation=counting_evaluation(event_ground_truth=False),
    )

    assert report["detection"]["precision"] is None
    assert report["detection"]["accuracy_claimed"] is False
    assert report["counting"]["status"] == "aggregate_diagnostics_only"
    assert report["counting"]["f1"] is None
    assert report["metric_contract"]["ground_truth_accuracy_sections"] == []


def test_report_writes_machine_readable_json_and_csv(tmp_path: Path) -> None:
    report = build_benchmark_report(
        detection_artifact=detection_artifact(),
        processing_summary=processing_summary(),
        counting_evaluation=counting_evaluation(),
    )
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    write_benchmark_report(json_path, csv_path, report)

    assert json.loads(json_path.read_text("utf-8"))["detection"]["f1"] == 0.685714
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["detection_f1"] == "0.685714"
    assert row["counting_f1"] == "0.727273"
    assert row["id_switches"] == ""


def test_report_rejects_missing_inputs_and_output_alias(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkReportError, match="At least one"):
        build_benchmark_report()
    report = build_benchmark_report(processing_summary=processing_summary())
    same = tmp_path / "same.json"
    with pytest.raises(BenchmarkReportError, match="different"):
        write_benchmark_report(same, same, report)


def test_cli_refuses_to_run_without_any_metric_artifact(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path / "report.json")]) == 2
