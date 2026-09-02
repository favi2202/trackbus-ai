from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackbus.config import ConfigError, load_config
from trackbus.counter import EventType, PassengerCounter
from trackbus.detection import TrackedDetection
from trackbus.tracker import ByteTrackAdapter, TrackContinuityAdapter
from trackbus.zones import ZoneMembership


class SequenceTracker:
    device_label = "cpu"
    tracker_config = "fake-bytetrack.yaml"
    ultralytics_version = "test-version"
    effective_config = {"track_buffer": 30}

    def __init__(self, frames: list[list[TrackedDetection]]) -> None:
        self.frames = frames
        self.update_count = 0
        self.reset_count = 0

    def update(self, _detections: object, _frame: np.ndarray) -> list[TrackedDetection]:
        output = self.frames[self.update_count]
        self.update_count += 1
        return output

    def reset(self) -> None:
        self.update_count = 0
        self.reset_count += 1


def person(
    tracking_id: int,
    box: tuple[float, float, float, float],
    *,
    metadata: dict[str, object] | None = None,
) -> TrackedDetection:
    return TrackedDetection(
        tracking_id=tracking_id,
        bounding_box=box,
        confidence=0.8,
        metadata=metadata or {},
    )


def run(
    frames: list[list[TrackedDetection]],
    **settings: float | int,
) -> tuple[TrackContinuityAdapter, list[list[TrackedDetection]], SequenceTracker]:
    wrapped = SequenceTracker(frames)
    adapter = TrackContinuityAdapter(wrapped, **settings)  # type: ignore[arg-type]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    outputs = [adapter.update([], image) for _ in frames]
    return adapter, outputs, wrapped


def test_same_bytetrack_id_survives_short_detection_gap_without_stitch() -> None:
    adapter, outputs, wrapped = run(
        [
            [person(7, (10, 10, 30, 50))],
            [],
            [],
            [person(7, (12, 10, 32, 50))],
        ]
    )

    assert outputs[0][0].tracking_id == 7
    assert outputs[3][0].tracking_id == 7
    assert outputs[3][0].metadata["continuity_stitched"] is False
    assert outputs[3][0].metadata["continuity_gap_frames"] == 2
    assert adapter.continuity_summary["stitched_track_fragments"] == 0
    assert adapter.update_count == wrapped.update_count == 4


def test_new_raw_id_is_stitched_after_short_consistent_gap() -> None:
    adapter, outputs, _wrapped = run(
        [
            [person(4, (10, 10, 30, 50))],
            [],
            [],
            [person(19, (12, 10, 32, 50), metadata={"quality_flags": ("edge",)})],
        ]
    )

    resumed = outputs[3][0]
    assert resumed.tracking_id == 4
    assert resumed.metadata["raw_tracking_id"] == 19
    assert resumed.metadata["continuity_previous_raw_tracking_id"] == 4
    assert resumed.metadata["continuity_stitched"] is True
    assert resumed.metadata["continuity_gap_frames"] == 2
    assert resumed.metadata["quality_flags"] == (
        "edge",
        "track_continuity_stitch",
    )
    assert adapter.continuity_summary["stitched_track_fragments"] == 1
    assert adapter.continuity_summary["maximum_stitched_gap_frames"] == 2


def test_short_dropout_exposes_visual_only_motion_prediction() -> None:
    wrapped = SequenceTracker(
        [
            [person(4, (10, 10, 30, 50))],
            [person(4, (14, 10, 34, 50))],
            [],
        ]
    )
    adapter = TrackContinuityAdapter(wrapped, max_gap_frames=3)  # type: ignore[arg-type]
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    assert adapter.update([], image)
    assert adapter.update([], image)
    assert adapter.update([], image) == []
    prediction = adapter.predicted_tracks(image.shape)[0]

    assert prediction.tracking_id == 4
    assert prediction.bounding_box == pytest.approx((18, 10, 38, 50))
    assert prediction.metadata["prediction_only"] is True
    assert prediction.metadata["continuity_prediction_age_frames"] == 1
    assert "track_prediction_only" in prediction.metadata["quality_flags"]
    assert adapter.continuity_summary["predicted_track_frames"] == 1


def test_visual_prediction_expires_and_never_appears_as_tracker_output() -> None:
    wrapped = SequenceTracker(
        [[person(4, (10, 10, 30, 50))], [], [], [], []]
    )
    adapter = TrackContinuityAdapter(wrapped, max_gap_frames=2)  # type: ignore[arg-type]
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    outputs = []
    predictions = []
    for _frame in wrapped.frames:
        outputs.append(adapter.update([], image))
        predictions.append(adapter.predicted_tracks(image.shape))

    assert outputs[1:] == [[], [], [], []]
    assert predictions[1]
    assert predictions[2]
    assert predictions[3] == []
    assert predictions[4] == []
    assert adapter.continuity_summary["expired_memories"] == 1


def test_expired_raw_id_cannot_resurrect_an_old_stable_identity() -> None:
    _adapter, outputs, _wrapped = run(
        [
            [person(4, (10, 10, 30, 50))],
            [],
            [],
            [],
            [person(4, (12, 10, 32, 50))],
        ],
        max_gap_frames=2,
    )

    assert outputs[0][0].tracking_id == 4
    assert outputs[4][0].tracking_id != 4


def test_fragment_beyond_maximum_gap_becomes_new_stable_track() -> None:
    adapter, outputs, _wrapped = run(
        [
            [person(4, (10, 10, 30, 50))],
            [],
            [],
            [],
            [person(19, (12, 10, 32, 50))],
        ],
        max_gap_frames=2,
    )

    assert outputs[4][0].tracking_id == 19
    assert outputs[4][0].metadata["continuity_stitched"] is False
    assert adapter.continuity_summary["stitched_track_fragments"] == 0
    assert adapter.continuity_summary["expired_memories"] == 1


def test_distant_or_scale_incompatible_fragment_is_not_stitched() -> None:
    _adapter, outputs, _wrapped = run(
        [
            [person(2, (10, 10, 30, 50))],
            [],
            [person(9, (70, 10, 90, 50))],
        ]
    )
    assert outputs[2][0].tracking_id == 9

    _adapter, outputs, _wrapped = run(
        [
            [person(2, (10, 10, 20, 30))],
            [],
            [person(9, (8, 5, 32, 45))],
        ],
        maximum_size_ratio=1.5,
    )
    assert outputs[2][0].tracking_id == 9


def test_opposite_motion_fragment_is_not_stitched() -> None:
    _adapter, outputs, _wrapped = run(
        [
            [person(2, (10, 10, 30, 50))],
            [person(2, (14, 10, 34, 50))],
            [],
            [person(9, (5, 10, 25, 50))],
        ],
        max_centroid_distance=0.30,
    )

    assert outputs[3][0].tracking_id == 9


def test_simultaneous_fragments_are_matched_one_to_one() -> None:
    adapter, outputs, _wrapped = run(
        [
            [
                person(1, (5, 10, 25, 50)),
                person(2, (70, 10, 90, 50)),
            ],
            [],
            [
                person(11, (7, 10, 27, 50)),
                person(22, (68, 10, 88, 50)),
            ],
        ]
    )

    assert [item.tracking_id for item in outputs[2]] == [1, 2]
    assert adapter.continuity_summary["stitched_track_fragments"] == 2
    assert len({item.tracking_id for item in outputs[2]}) == 2


def test_stitched_id_preserves_three_zone_event_history() -> None:
    _adapter, outputs, _wrapped = run(
        [
            [person(4, (10, 20, 30, 40))],
            [person(4, (10, 30, 30, 50))],
            [],
            [person(19, (10, 52, 30, 72))],
        ]
    )
    counter = PassengerCounter(
        capacity=40,
        minimum_zone_frames=1,
        maximum_transition_gap_frames=3,
        event_cooldown_frames=0,
    )
    zones = {
        0: ZoneMembership.OUTSIDE,
        1: ZoneMembership.TRANSITION,
        3: ZoneMembership.INSIDE,
    }
    event = None
    for frame_number, tracked in enumerate(outputs):
        if tracked:
            event = counter.observe(
                tracked[0].tracking_id,
                zones[frame_number],
                frame_number,
            )

    assert event is not None
    assert event.event_type is EventType.IN
    assert event.tracking_id == 4
    assert counter.entered_total == 1


def test_reset_clears_video_local_identity_and_statistics() -> None:
    adapter, _outputs, wrapped = run([[person(5, (10, 10, 30, 50))]])

    adapter.reset()

    assert wrapped.reset_count == 1
    assert adapter.update_count == 0
    assert adapter.continuity_summary["active_track_memories"] == 0
    assert adapter.continuity_summary["new_stable_tracks"] == 0


@pytest.mark.parametrize(
    "profile",
    [
        "configs/tracking_fast.yaml",
        "configs/tracking_balanced.yaml",
        "configs/tracking_occlusion.yaml",
    ],
)
def test_tracking_profiles_satisfy_real_bytetrack_contract(profile: str) -> None:
    adapter = ByteTrackAdapter(profile, device_label="cpu")

    assert (
        adapter.effective_config["track_low_thresh"]
        <= (adapter.effective_config["track_high_thresh"])
    )
    assert 1 <= adapter.effective_config["track_buffer"] <= 75


def test_default_continuity_is_disabled_and_bounded() -> None:
    continuity = load_config(Path("configs/default.yaml")).tracking.continuity

    assert continuity.enabled is False
    assert continuity.max_gap_frames == 3
    assert continuity.max_centroid_distance == pytest.approx(0.08)


def test_invalid_continuity_limit_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """
tracking:
  continuity:
    enabled: true
    max_gap_frames: 31
zones:
  outside: [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]
  inside: [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]
outputs: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="max_gap_frames"):
        load_config(config)
