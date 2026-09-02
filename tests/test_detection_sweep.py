from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from trackbus.detection import Detection
from trackbus.sweep_detection import (
    DetectionSweepError,
    DetectionSweepRun,
    load_detection_sweep,
    run_detection_sweep,
)


class ConstantDetector:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def detect(self, image: np.ndarray, *, source_view: str) -> list[Detection]:
        assert image.shape == (24, 32, 3)
        return [Detection((3, 3, 15, 20), self.confidence, 0, source_view)]


def write_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.write(np.ones((24, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def write_matrix(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "device": "cpu",
                "maximum_runs": 4,
                "runs": runs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_explicit_sweep_runs_are_loaded_without_hidden_cartesian_product(
    tmp_path: Path,
) -> None:
    matrix = load_detection_sweep(
        write_matrix(
            tmp_path,
            [
                {
                    "name": "nano 640",
                    "model": "yolo11n.pt",
                    "image_size": 640,
                    "confidence": 0.2,
                },
                {
                    "name": "small 960",
                    "model": "yolo11s.pt",
                    "image_size": 960,
                    "detector_floor": 0.15,
                    "precision": "fp16",
                    "preprocessing_profile": "low_light",
                },
            ],
        )
    )

    assert [run.name for run in matrix.runs] == ["nano 640", "small 960"]
    assert matrix.runs[1].precision == "fp16"
    assert matrix.runs[1].detector_floor == 0.15
    assert matrix.runs[1].preprocessing_profile == "low_light"
    assert matrix.device == "cpu"


def test_sweep_refuses_too_many_runs_and_remote_models(tmp_path: Path) -> None:
    too_many = [
        {
            "name": f"run {index}",
            "model": "yolo11n.pt",
            "image_size": 640,
            "confidence": 0.2,
        }
        for index in range(5)
    ]
    with pytest.raises(DetectionSweepError, match="defines 5 runs"):
        load_detection_sweep(write_matrix(tmp_path, too_many))

    with pytest.raises(DetectionSweepError, match="remote URL"):
        load_detection_sweep(
            write_matrix(
                tmp_path,
                [
                    {
                        "name": "remote",
                        "model": "https://example.invalid/model.pt",
                        "image_size": 640,
                        "confidence": 0.2,
                    }
                ],
            )
        )


def test_sweep_writes_per_run_evidence_and_proxy_only_leaderboard(
    tmp_path: Path,
) -> None:
    video = write_video(tmp_path)
    matrix = load_detection_sweep(
        write_matrix(
            tmp_path,
            [
                {
                    "name": "nano 640",
                    "model": "yolo11n.pt",
                    "image_size": 640,
                    "confidence": 0.2,
                },
                {
                    "name": "small 960",
                    "model": "yolo11s.pt",
                    "image_size": 960,
                    "confidence": 0.15,
                    "preprocessing_profile": "contrast",
                },
            ],
        )
    )

    def factory(
        run: DetectionSweepRun, _device: str | int, _device_label: str
    ) -> ConstantDetector:
        return ConstantDetector(run.confidence)

    output = tmp_path / "results"
    results = run_detection_sweep(
        video_path=video,
        matrix=matrix,
        output_dir=output,
        detector_factory=factory,
        device_resolver=lambda _requested: ("cpu", "cpu"),
    )

    assert [result["status"] for result in results] == ["completed", "completed"]
    assert (output / "nano-640" / "benchmark.json").is_file()
    assert (output / "small-960" / "frames.csv").is_file()
    document = json.loads((output / "results.json").read_text("utf-8"))
    assert document["selection_status"] == "no_accuracy_winner_without_box_ground_truth"
    with (output / "leaderboard.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["recall"] == ""
    assert rows[0]["metric_status"] == "proxy_only_no_box_ground_truth"
    assert rows[1]["preprocessing_profile"] == "contrast"
    assert float(rows[1]["mean_preprocessing_latency_ms"]) >= 0
