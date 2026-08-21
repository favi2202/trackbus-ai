from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackbus.benchmark_detection import main as benchmark_main
from trackbus.detection import Detection
from trackbus.detection_benchmark import (
    DetectionBenchmarkError,
    benchmark_detector,
    load_detection_ground_truth,
    maximum_iou_matching,
    write_benchmark_json,
    write_frame_metrics_csv,
)


class SequenceDetector:
    def __init__(self, frames: list[list[Detection]]) -> None:
        self.frames = frames
        self.index = 0

    def detect(self, image: np.ndarray, *, source_view: str) -> list[Detection]:
        assert image.shape == (24, 32, 3)
        assert source_view == "full"
        detections = self.frames[self.index]
        self.index += 1
        return detections


def detection(
    box: tuple[float, float, float, float], confidence: float = 0.8
) -> Detection:
    return Detection(box, confidence, 0, "full")


def write_video(tmp_path: Path, frame_count: int = 4) -> Path:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    return path


def write_truth(tmp_path: Path) -> Path:
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_space": "pixels",
                "video": {"filename": "tiny.avi", "width": 32, "height": 24},
                "frames": [
                    {"frame": 0, "persons": [{"id": "p1", "bbox": [1, 1, 11, 11]}]},
                    {"frame": 1, "persons": []},
                    {"frame": 2, "persons": [[12, 2, 22, 18]]},
                    {"frame": 3, "persons": [[14, 2, 24, 18]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_detector_only_benchmark_reports_proxies_without_claiming_accuracy(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path)
    detector = SequenceDetector(
        [
            [detection((1, 1, 11, 11), 0.9)],
            [],
            [],
            [detection((14, 3, 24, 18), 0.3)],
        ]
    )

    result = benchmark_detector(
        video_path=video,
        detector=detector,
        model_name="fake.pt",
        image_size=640,
        confidence_threshold=0.2,
        device_label="cpu",
        low_confidence_threshold=0.35,
    )

    assert result.summary["benchmark_type"] == "detector_only"
    assert result.summary["uses_tracking"] is False
    assert result.summary["uses_counting"] is False
    assert result.summary["metric_status"] == "proxy_only_no_box_ground_truth"
    assert result.summary["ground_truth"] is None
    assert result.summary["detections"]["total"] == 2
    assert result.summary["detections"]["frames_without_detections"] == 2
    assert result.summary["detections"]["zero_detection_streak_lengths"] == [2]
    assert result.summary["detections"]["low_confidence_total"] == 1
    assert result.summary["detections"]["edge_clipped_total"] == 1


def test_detector_ground_truth_enables_real_precision_recall_and_f1(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path)
    truth = load_detection_ground_truth(write_truth(tmp_path))
    detector = SequenceDetector(
        [
            [detection((1, 1, 11, 11)), detection((20, 1, 30, 11))],
            [],
            [],
            [detection((14, 2, 24, 18))],
        ]
    )

    result = benchmark_detector(
        video_path=video,
        detector=detector,
        model_name="fake.pt",
        image_size=640,
        confidence_threshold=0.2,
        device_label="cpu",
        ground_truth=truth,
    )

    metrics = result.summary["ground_truth"]
    assert result.summary["metric_status"] == "ground_truth_evaluated"
    assert metrics == {
        "annotated_frames_processed": 4,
        "annotated_persons": 3,
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "missed_persons": 1,
        "precision": pytest.approx(2 / 3, abs=1e-6),
        "recall": pytest.approx(2 / 3, abs=1e-6),
        "f1": pytest.approx(2 / 3, abs=1e-6),
    }
    assert result.frames[2].false_negatives == 1


def test_unlisted_ground_truth_frames_are_unannotated_not_empty(tmp_path: Path) -> None:
    video = write_video(tmp_path, frame_count=2)
    truth_path = tmp_path / "partial.json"
    truth_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_space": "normalized",
                "frames": [{"frame": 0, "persons": [[0.0, 0.0, 0.5, 0.5]]}],
            }
        ),
        encoding="utf-8",
    )
    truth = load_detection_ground_truth(truth_path)
    detector = SequenceDetector(
        [[detection((0, 0, 16, 12))], [detection((1, 1, 5, 5))]]
    )

    result = benchmark_detector(
        video_path=video,
        detector=detector,
        model_name="fake.pt",
        image_size=640,
        confidence_threshold=0.2,
        device_label="cpu",
        ground_truth=truth,
    )

    assert result.summary["ground_truth"]["annotated_frames_processed"] == 1
    assert result.summary["ground_truth"]["false_positives"] == 0
    assert result.frames[1].ground_truth_persons is None


def test_maximum_iou_matching_does_not_double_match_boxes() -> None:
    pairs = maximum_iou_matching(
        [(0, 0, 10, 10), (8, 0, 18, 10)],
        [(1, 0, 11, 10), (9, 0, 19, 10)],
        threshold=0.1,
    )

    assert len(pairs) == 2
    assert {first for first, _second, _iou in pairs} == {0, 1}
    assert {second for _first, second, _iou in pairs} == {0, 1}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"schema_version": 2, "frames": []}, "schema_version"),
        (
            {
                "schema_version": 1,
                "frames": [
                    {"frame": 1, "persons": []},
                    {"frame": 1, "persons": []},
                ],
            },
            "duplicate frame",
        ),
        (
            {
                "schema_version": 1,
                "coordinate_space": "normalized",
                "frames": [{"frame": 0, "persons": [[0, 0, 2, 1]]}],
            },
            "between 0 and 1",
        ),
    ],
)
def test_ground_truth_loader_rejects_invalid_documents(
    tmp_path: Path, document: dict[str, object], message: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DetectionBenchmarkError, match=message):
        load_detection_ground_truth(path)


def test_benchmark_artifacts_are_machine_readable(tmp_path: Path) -> None:
    video = write_video(tmp_path, frame_count=1)
    result = benchmark_detector(
        video_path=video,
        detector=SequenceDetector([[detection((2, 2, 10, 12))]]),
        model_name="fake.pt",
        image_size=640,
        confidence_threshold=0.2,
        device_label="cpu",
    )
    json_path = tmp_path / "output" / "benchmark.json"
    csv_path = tmp_path / "output" / "frames.csv"

    write_benchmark_json(json_path, result)
    write_frame_metrics_csv(csv_path, result.frames)

    assert (
        json.loads(json_path.read_text("utf-8"))["summary"]["detections"]["total"] == 1
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frame"] == "0"
    assert rows[0]["detections"] == "1"


def test_cli_refuses_to_overwrite_the_input_video_before_loading_a_model(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path, frame_count=1)

    assert benchmark_main(["--video", str(video), "--output", str(video)]) == 2
    assert video.is_file()
    assert video.stat().st_size > 0
