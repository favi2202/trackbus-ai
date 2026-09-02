from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trackbus.detection import Detection
from trackbus.tracker import (
    ByteTrackAdapter,
    TrackerAdapterError,
    validate_confidence_contract,
)


class FakeBoxes:
    instances: list[FakeBoxes] = []

    def __init__(self, data: np.ndarray, orig_shape: tuple[int, int]) -> None:
        self.data = data
        self.orig_shape = orig_shape
        self.cpu_calls = 0
        self.numpy_calls = 0
        self.__class__.instances.append(self)

    def cpu(self) -> FakeBoxes:
        self.cpu_calls += 1
        return self

    def numpy(self) -> FakeBoxes:
        self.numpy_calls += 1
        return self


class FakeByteTracker:
    def __init__(self, tracks: object = ()) -> None:
        self.tracks = tracks
        self.calls: list[tuple[FakeBoxes, np.ndarray]] = []
        self.reset_calls = 0
        self.error: Exception | None = None

    def update(self, boxes: FakeBoxes, frame: np.ndarray) -> object:
        self.calls.append((boxes, frame))
        if self.error is not None:
            raise self.error
        return self.tracks

    def reset(self) -> None:
        self.reset_calls += 1
        if self.error is not None:
            raise self.error


def make_adapter(tracker: FakeByteTracker) -> ByteTrackAdapter:
    adapter = object.__new__(ByteTrackAdapter)
    adapter.tracker_config = "fake-bytetrack.yaml"
    adapter.device_label = "cpu"
    adapter.update_count = 0
    adapter.ultralytics_version = "test-version"
    adapter._boxes_type = FakeBoxes
    adapter._tracker = tracker
    return adapter


def make_detection(
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
    class_id: int = 0,
    view: str = "full",
    contributing_views: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Detection:
    return Detection(
        bounding_box=box,
        confidence=confidence,
        class_id=class_id,
        source_view=view,
        contributing_views=contributing_views,
        metadata={} if metadata is None else metadata,
    )


def test_adapter_converts_fused_detections_and_maps_track_metadata() -> None:
    tracker = FakeByteTracker(
        np.asarray(
            [
                [101.0, 102.0, 121.0, 142.0, 17.0, 0.76, 0.0, 1.0],
                [11.0, 12.0, 31.0, 42.0, 8.0, 0.88, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    adapter = make_adapter(tracker)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = [
        make_detection((10.0, 11.0, 30.0, 41.0), metadata={"token": "first"}),
        make_detection(
            (100.0, 101.0, 120.0, 141.0),
            confidence=0.7,
            view="right",
            contributing_views=("right", "full"),
            metadata={"token": "second"},
        ),
    ]

    tracked = adapter.update(detections, frame)

    assert adapter.update_count == 1
    assert len(tracker.calls) == 1
    boxes, submitted_frame = tracker.calls[0]
    assert submitted_frame is frame
    assert boxes.orig_shape == (240, 320)
    assert boxes.data.dtype == np.float32
    np.testing.assert_allclose(
        boxes.data,
        np.asarray(
            [
                [10.0, 11.0, 30.0, 41.0, 0.8, 0.0],
                [100.0, 101.0, 120.0, 141.0, 0.7, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert (boxes.cpu_calls, boxes.numpy_calls) == (1, 1)
    assert [item.tracking_id for item in tracked] == [17, 8]
    assert tracked[0].bounding_box == (101.0, 102.0, 121.0, 142.0)
    assert tracked[0].confidence == pytest.approx(0.76)
    assert tracked[0].source_views == ("right", "full")
    assert tracked[0].metadata == {"token": "second", "tracker_input_index": 1}
    assert tracked[0].anchor == (111.0, 142.0)
    assert tracked[1].metadata == {"token": "first", "tracker_input_index": 0}
    assert detections[1].metadata == {"token": "second"}


def test_empty_update_still_advances_tracker_once_with_n_by_six_input() -> None:
    tracker = FakeByteTracker(np.empty((0, 8), dtype=np.float32))
    adapter = make_adapter(tracker)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    assert adapter.update([], frame) == []

    assert adapter.update_count == 1
    assert len(tracker.calls) == 1
    boxes, submitted_frame = tracker.calls[0]
    assert submitted_frame is frame
    assert boxes.orig_shape == (20, 30)
    assert boxes.data.shape == (0, 6)


def test_adapter_updates_exactly_once_per_source_frame() -> None:
    tracker = FakeByteTracker(np.empty((0, 8), dtype=np.float32))
    adapter = make_adapter(tracker)

    for value in (1, 2, 3):
        adapter.update([], np.full((4, 5, 3), value, dtype=np.uint8))

    assert adapter.update_count == 3
    assert len(tracker.calls) == 3
    assert [int(frame[0, 0, 0]) for _boxes, frame in tracker.calls] == [1, 2, 3]


def test_adapter_rejects_non_person_detections_before_tracker_update() -> None:
    tracker = FakeByteTracker()
    adapter = make_adapter(tracker)

    with pytest.raises(TrackerAdapterError, match="person-class detections only"):
        adapter.update(
            [make_detection((0.0, 0.0, 10.0, 10.0), class_id=1)],
            np.zeros((20, 20, 3), dtype=np.uint8),
        )

    assert tracker.calls == []
    assert adapter.update_count == 0


def test_adapter_wraps_internal_tracker_errors_without_fallback() -> None:
    tracker = FakeByteTracker()
    tracker.error = RuntimeError("incompatible update")
    adapter = make_adapter(tracker)

    with pytest.raises(
        TrackerAdapterError,
        match="installed version test-version.*incompatible update",
    ):
        adapter.update([], np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(tracker.calls) == 1
    assert adapter.update_count == 0


@pytest.mark.parametrize(
    ("tracks", "message"),
    [
        (np.zeros((1, 7), dtype=np.float32), "expected Nx8"),
        (np.zeros((1, 9), dtype=np.float32), "expected Nx8"),
        (np.zeros(8, dtype=np.float32), "expected Nx8"),
        (
            np.asarray([[0, 0, 1, 1, 4, 0.8, 0, 2]], dtype=np.float32),
            "invalid source-detection index 2",
        ),
    ],
)
def test_adapter_rejects_incompatible_tracker_outputs(
    tracks: np.ndarray, message: str
) -> None:
    adapter = make_adapter(FakeByteTracker(tracks))

    with pytest.raises(TrackerAdapterError, match=message):
        adapter.update(
            [make_detection((0.0, 0.0, 10.0, 10.0))],
            np.zeros((20, 20, 3), dtype=np.uint8),
        )


def test_reset_delegates_to_tracker_and_clears_update_count() -> None:
    tracker = FakeByteTracker(np.empty((0, 8), dtype=np.float32))
    adapter = make_adapter(tracker)
    adapter.update([], np.zeros((5, 5, 3), dtype=np.uint8))

    adapter.reset()

    assert tracker.reset_calls == 1
    assert adapter.update_count == 0


def test_reset_wraps_tracker_error() -> None:
    tracker = FakeByteTracker()
    tracker.error = RuntimeError("reset unsupported")
    adapter = make_adapter(tracker)

    with pytest.raises(TrackerAdapterError, match="Could not reset ByteTrack"):
        adapter.reset()


def test_adapter_rejects_remote_or_missing_tracker_configuration() -> None:
    for value in ("https://example.invalid/tracker.yaml", "ul://tracker.yaml"):
        with pytest.raises(TrackerAdapterError, match="Remote"):
            ByteTrackAdapter(value, device_label="cpu")
    with pytest.raises(TrackerAdapterError, match="existing local file"):
        ByteTrackAdapter("definitely-missing-tracker.yaml", device_label="cpu")


def test_confidence_contract_rejects_weak_new_track_threshold() -> None:
    with pytest.raises(TrackerAdapterError, match="weak detections"):
        validate_confidence_contract(
            0.10,
            {
                "track_low_thresh": 0.10,
                "track_high_thresh": 0.25,
                "new_track_thresh": 0.20,
            },
        )


def test_real_bytetrack_uses_weak_detection_only_for_existing_track() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    weak = make_detection((20.0, 20.0, 40.0, 80.0), confidence=0.15)
    strong = make_detection((20.0, 20.0, 40.0, 80.0), confidence=0.80)

    new_track_adapter = ByteTrackAdapter(
        "configs/tracking_balanced.yaml", device_label="cpu"
    )
    assert new_track_adapter.update([weak], frame) == []

    existing_track_adapter = ByteTrackAdapter(
        "configs/tracking_balanced.yaml", device_label="cpu"
    )
    started = existing_track_adapter.update([strong], frame)
    maintained = existing_track_adapter.update([weak], frame)

    assert len(started) == len(maintained) == 1
    assert maintained[0].tracking_id == started[0].tracking_id
    assert maintained[0].confidence == pytest.approx(0.15)


def test_installed_ultralytics_bytetrack_contract_and_id_continuity() -> None:
    """Diagnose changes to the real, model-free Ultralytics internal API."""

    adapter = ByteTrackAdapter(
        str(Path("configs/bytetrack_trackbus.yaml")),
        device_label="cpu",
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = make_detection((20.0, 10.0, 50.0, 80.0), confidence=0.95)

    first = adapter.update([detection], frame)
    missing = adapter.update([], frame)
    resumed = adapter.update([detection], frame)

    assert len(first) == 1
    assert missing == []
    assert len(resumed) == 1
    assert resumed[0].tracking_id == first[0].tracking_id
    assert adapter.update_count == 3
