from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from trackbus.config import (
    CameraConfig,
    DiagnosticsConfig,
    InferenceViewConfig,
    ZonesConfig,
)
from trackbus.counter import PassengerCounter
from trackbus.detection import Detection, TrackedDetection
from trackbus.detection_export import DetectionCsvExporter
from trackbus.event_logger import EventLogger
from trackbus.fusion import NmsDetectionFusion
from trackbus.video_processor import VideoProcessor


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()


class SequencedDetector:
    def __init__(self) -> None:
        self.frame = 0

    def detect(self, _image: np.ndarray, *, source_view: str) -> list[Detection]:
        excluded = Detection((0.0, 60.0, 20.0, 80.0), 0.8, 0, source_view)
        normal = (
            Detection((40.0, 5.0, 60.0, 30.0), 0.9, 0, source_view)
            if self.frame == 0
            else Detection((40.0, 60.0, 60.0, 80.0), 0.9, 0, source_view)
        )
        detections = [excluded, normal] if self.frame != 1 else [excluded]
        self.frame += 1
        return detections


class RecordingFusion(NmsDetectionFusion):
    def __init__(self) -> None:
        super().__init__(0.5)
        self.received_counts: list[int] = []

    def fuse(self, detections: list[Detection]) -> list[Detection]:
        self.received_counts.append(len(detections))
        return super().fuse(detections)


class RecordingTracker:
    device_label = "test"

    def __init__(self) -> None:
        self.received_counts: list[int] = []

    def update(
        self, detections: list[Detection], _frame: np.ndarray
    ) -> list[TrackedDetection]:
        self.received_counts.append(len(detections))
        return [
            TrackedDetection(
                tracking_id=7,
                bounding_box=detection.bounding_box,
                confidence=detection.confidence,
                class_id=detection.class_id,
                source_views=detection.source_views,
            )
            for detection in detections
        ]


class OutsideFrameTracker:
    device_label = "test"

    def __init__(self) -> None:
        self.update_count = 0

    def update(
        self, _detections: list[Detection], _frame: np.ndarray
    ) -> list[TrackedDetection]:
        self.update_count += 1
        return [TrackedDetection(11, (-20.0, 5.0, -2.0, 30.0), 0.9)]


def test_explicit_pipeline_excludes_then_fuses_and_tracks_once_per_frame(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    _video(input_path)
    raw_path = tmp_path / "raw.csv"
    fused_path = tmp_path / "fused.csv"
    tracks_path = tmp_path / "tracks.csv"
    detector = SequencedDetector()
    fusion = RecordingFusion()
    tracker = RecordingTracker()
    zones = ZonesConfig(
        outside=((0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)),
        inside=((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)),
    )
    camera = CameraConfig(
        inference_views=(InferenceViewConfig("full", (0.0, 0.0, 1.0, 1.0)),),
        exclusion_polygons=(((0.0, 0.55), (0.25, 0.55), (0.25, 1.0), (0.0, 1.0)),),
    )

    with (
        EventLogger(tmp_path / "events.csv", tmp_path / "summary.json") as events,
        DetectionCsvExporter(
            enabled=True,
            raw_path=raw_path,
            fused_path=fused_path,
            tracks_path=tracks_path,
        ) as exporter,
    ):
        summary = VideoProcessor(
            detector=detector,
            fusion=fusion,
            tracker=tracker,
            counter=PassengerCounter(capacity=40, minimum_zone_frames=1),
            zones_config=zones,
            event_logger=events,
            model_name="fake",
            model_confidence=0.2,
            inference_image_size=640,
            camera_config=camera,
            diagnostics_config=DiagnosticsConfig(),
            detection_exporter=exporter,
        ).process(input_path, output_path)

    assert fusion.received_counts == [1, 0, 1]
    assert tracker.received_counts == [1, 0, 1]
    assert summary.tracker_updates == summary.processed_frames == 3
    assert summary.entered_total == 1
    assert summary.raw_person_detections_total == 5
    assert summary.fused_detections_total == 2
    assert summary.excluded_person_detections_total == 3

    with raw_path.open(encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    with fused_path.open(encoding="utf-8", newline="") as file:
        fused_rows = list(csv.DictReader(file))
    with tracks_path.open(encoding="utf-8", newline="") as file:
        track_rows = list(csv.DictReader(file))
    assert len(raw_rows) == 5
    assert "timestamp" in raw_rows[0]
    assert sum(row["excluded"] == "true" for row in raw_rows) == 3
    assert len(fused_rows) == 2
    assert fused_rows[0]["contributing_views"] == "full"
    assert '"fusion_method": "nms"' in fused_rows[0]["fusion_metadata"]
    assert len(track_rows) == 2
    assert [row["current_zone"] for row in track_rows] == ["outside", "inside"]
    assert [row["counter_state"] for row in track_rows] == ["outside", "inside"]


def test_detection_csv_exporter_creates_nothing_when_disabled(
    tmp_path: Path,
) -> None:
    paths = (tmp_path / "raw.csv", tmp_path / "fused.csv", tmp_path / "tracks.csv")
    detection = Detection((1.0, 2.0, 3.0, 4.0), 0.9, 0, "full")
    with DetectionCsvExporter(
        enabled=False,
        raw_path=paths[0],
        fused_path=paths[1],
        tracks_path=paths[2],
    ) as exporter:
        exporter.record_raw(0, 0.0, [(detection, False)])
        exporter.record_fused(0, [detection])
        exporter.record_tracks(0, [], {}, {}, {})

    assert not any(path.exists() for path in paths)


def test_tracker_boxes_collapsed_outside_source_frame_are_not_counted(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    _video(input_path)
    tracker = OutsideFrameTracker()
    zones = ZonesConfig(
        outside=((0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)),
        inside=((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)),
    )

    with EventLogger(tmp_path / "events.csv", tmp_path / "summary.json") as events:
        summary = VideoProcessor(
            detector=SequencedDetector(),
            fusion=NmsDetectionFusion(0.5),
            tracker=tracker,
            counter=PassengerCounter(capacity=40, minimum_zone_frames=1),
            zones_config=zones,
            event_logger=events,
            model_name="fake",
            model_confidence=0.2,
            inference_image_size=640,
            camera_config=CameraConfig(
                inference_views=(InferenceViewConfig("full", (0.0, 0.0, 1.0, 1.0)),),
            ),
            diagnostics_config=DiagnosticsConfig(),
        ).process(input_path, output_path)

    assert summary.tracker_updates == tracker.update_count == 3
    assert summary.tracked_person_observations_total == 0
    assert summary.entered_total == summary.exited_total == 0
