from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trackbus.camera_quality import (
    analyze_video,
    build_parser,
    calibration_path_coverage,
    classify_camera_quality,
    measure_frame_quality,
)
from trackbus.config import load_config
from trackbus.detection import Detection


def test_frame_metrics_do_not_retain_image_data() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = Detection((0.0, 10.0, 40.0, 80.0), 0.42, 0, "full")

    sample = measure_frame_quality(frame, [detection], edge_margin_pixels=2)

    assert sample.brightness_mean == 0.0
    assert sample.dark_pixel_percentage == 100.0
    assert sample.sharpness_laplacian_variance == 0.0
    assert sample.person_height_ratios == (0.7,)
    assert sample.edge_clipped_detection_count == 1
    assert not hasattr(sample, "frame")


def test_quality_classification_separates_good_marginal_and_unsuitable() -> None:
    common = {
        "width": 1280,
        "height": 720,
        "underexposed_frame_percentage": 0.0,
        "overexposed_frame_percentage": 0.0,
        "complete_path_coverage": True,
        "detection_enabled": True,
        "detection_total": 20,
        "median_person_height_ratio": 0.20,
        "edge_clipped_detection_percentage": 5.0,
        "sudden_count_drop_percentage": 0.0,
    }
    good, _, _ = classify_camera_quality(**common, blurred_frame_percentage=0.0)
    marginal, _, _ = classify_camera_quality(**common, blurred_frame_percentage=30.0)
    unsuitable_input = {
        **common,
        "blurred_frame_percentage": 80.0,
        "complete_path_coverage": False,
    }
    unsuitable, reasons, _ = classify_camera_quality(**unsuitable_input)

    assert good == "GOOD"
    assert marginal == "MARGINAL"
    assert unsuitable == "UNSUITABLE"
    assert any("configured zones" in reason for reason in reasons)


def test_default_calibration_preserves_complete_path() -> None:
    config = load_config(Path("configs/default.yaml"))

    coverage = calibration_path_coverage(config.camera, config.zones)

    assert coverage["single_view_covers_outside_and_inside"] is True
    assert coverage["combined_views_cover_outside_and_inside"] is True


def test_no_detection_video_analysis_is_bounded_and_writes_no_frames(
    tmp_path: Path,
) -> None:
    video = tmp_path / "quality.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360)
    )
    assert writer.isOpened()
    for value in range(8):
        frame = np.full((360, 640, 3), 80 + value * 5, dtype=np.uint8)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 359), (220, 220, 220), 2)
        writer.write(frame)
    writer.release()

    report = analyze_video(
        video,
        config=load_config(Path("configs/default.yaml")),
        detector=None,
        sample_every=2,
        maximum_samples=3,
    )

    assert report.status == "MARGINAL"
    assert report.diagnostic_not_accuracy is True
    assert report.sampling["sampled_frames"] == 3
    assert report.sampling["frame_numbers"] == [0, 2, 4]
    assert report.detection_diagnostics["enabled"] is False
    assert sorted(path.name for path in tmp_path.iterdir()) == ["quality.mp4"]


def test_camera_quality_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "bus.mp4",
            "--detector-floor",
            "0.10",
            "--sample-every",
            "5",
            "--max-samples",
            "20",
            "--no-detection",
        ]
    )

    assert args.source == Path("bus.mp4")
    assert args.detector_floor == 0.10
    assert args.sample_every == 5
    assert args.max_samples == 20
    assert args.no_detection is True
