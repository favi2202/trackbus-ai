import csv
import json
from pathlib import Path

import cv2
import numpy as np

from trackbus.config import ZonesConfig
from trackbus.counter import PassengerCounter
from trackbus.event_logger import EventLogger
from trackbus.tracker import TrackedPerson
from trackbus.video_processor import VideoProcessor


class FakeTracker:
    device_label = "test"

    def __init__(self) -> None:
        self.frame = 0

    def update(self, _image: np.ndarray) -> list[TrackedPerson]:
        bottom_by_frame = (20.0, 50.0, 80.0)
        bottom = bottom_by_frame[self.frame]
        self.frame += 1
        return [
            TrackedPerson(
                tracking_id=12,
                bounding_box=(40.0, bottom - 15.0, 60.0, bottom),
                confidence=0.9,
            )
        ]


def make_input_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()


def test_video_pipeline_writes_all_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "result.mp4"
    csv_path = tmp_path / "events.csv"
    summary_path = tmp_path / "summary.json"
    make_input_video(input_path)

    zones = ZonesConfig(
        outside=((0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)),
        inside=((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)),
    )
    counter = PassengerCounter(capacity=40, minimum_zone_frames=1)
    tracker = FakeTracker()

    with EventLogger(csv_path, summary_path) as event_logger:
        processor = VideoProcessor(
            tracker=tracker,  # type: ignore[arg-type]
            counter=counter,
            zones_config=zones,
            event_logger=event_logger,
            model_name="test-model",
        )
        summary = processor.process(input_path, output_path)

    assert output_path.is_file() and output_path.stat().st_size > 0
    assert summary.processed_frames == 3
    assert summary.entered_total == 1
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "IN"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["final_occupancy"] == 1
    assert saved_summary["device_used"] == "test"
