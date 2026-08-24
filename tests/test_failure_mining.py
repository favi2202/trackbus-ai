from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trackbus.config import FailureMiningConfig
from trackbus.counter import (
    CounterSnapshot,
    EventType,
    MovementState,
    SuppressionReason,
)
from trackbus.detection import Detection, TrackedDetection
from trackbus.failure_mining import FailureMiner, derive_failure_paths
from trackbus.zones import ZoneMembership


def miner(
    tmp_path: Path,
    config: FailureMiningConfig,
    *,
    threshold: float = 0.35,
) -> FailureMiner:
    return FailureMiner(
        config,
        output_path=tmp_path / "failures.jsonl",
        frames_directory=tmp_path / "failure-frames",
        low_confidence_threshold=threshold,
        edge_margin_pixels=2,
        heavy_overlap_iou=0.5,
        initial_occupancy=0,
    )


def track(
    tracking_id: int,
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
) -> TrackedDetection:
    return TrackedDetection(
        tracking_id=tracking_id,
        bounding_box=box,
        confidence=confidence,
    )


def detection(confidence: float) -> Detection:
    return Detection((10.0, 10.0, 30.0, 60.0), confidence, 0, "full")


def snapshot() -> CounterSnapshot:
    return CounterSnapshot(
        raw_zone=ZoneMembership.INSIDE,
        effective_zone=ZoneMembership.INSIDE,
        stable_zone=ZoneMembership.OUTSIDE,
        state=MovementState.TRANSITIONING_IN,
        pending_direction=EventType.IN,
        origin_dwell_frames=3,
        destination_confirmation_frames=1,
        cooldown_remaining=0,
        suppression_reasons=(SuppressionReason.MISSING_NEUTRAL_TRANSITION,),
        emitted_event=None,
        detection_gap_frames=0,
    )


def test_disabled_failure_mining_creates_no_artifacts(tmp_path: Path) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    failure_miner = miner(tmp_path, FailureMiningConfig())

    with failure_miner:
        failure_miner.observe_frame(
            frame_number=0,
            timestamp_seconds=0.0,
            frame=frame,
            fused_detections=[detection(0.1)],
            tracks=[track(1, (0.0, 5.0, 20.0, 80.0))],
            snapshots={},
            events=[],
        )

    assert failure_miner.summary["enabled"] is False
    assert list(tmp_path.iterdir()) == []


def test_metadata_records_are_anonymous_and_hard_limited(tmp_path: Path) -> None:
    config = FailureMiningConfig(
        enabled=True,
        maximum_records=2,
        short_track_maximum_frames=1,
    )
    failure_miner = miner(tmp_path, config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with failure_miner:
        failure_miner.observe_frame(
            frame_number=0,
            timestamp_seconds=0.0,
            frame=frame,
            fused_detections=[detection(0.1)],
            tracks=[track(7, (0.0, 5.0, 20.0, 80.0))],
            snapshots={7: snapshot()},
            events=[],
        )

    records = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    assert all(
        record["anonymous_id_scope"] == "single_video_only" for record in records
    )
    assert all("person_name" not in record for record in records)
    assert records[1]["anonymous_track_ids"] == [7]
    assert failure_miner.summary["records_written"] == 2
    assert failure_miner.summary["records_dropped_at_limit"] >= 1
    assert failure_miner.summary["captured_frame_count"] == 0
    assert not (tmp_path / "failure-frames").exists()


def test_frame_capture_requires_opt_in_and_obeys_unique_frame_limit(
    tmp_path: Path,
) -> None:
    config = FailureMiningConfig(
        enabled=True,
        capture_frames=True,
        maximum_records=10,
        maximum_captured_frames=1,
    )
    failure_miner = miner(tmp_path, config)
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)

    with failure_miner:
        for frame_number in (0, 1):
            failure_miner.observe_frame(
                frame_number=frame_number,
                timestamp_seconds=frame_number / 10,
                frame=frame,
                fused_detections=[detection(0.1)],
                tracks=[],
                snapshots={},
                events=[],
            )

    captured = list((tmp_path / "failure-frames").glob("*.jpg"))
    assert len(captured) == 1
    assert failure_miner.summary["captured_frame_count"] == 1
    assert failure_miner.summary["frames_may_contain_personal_data"] is True
    assert failure_miner.summary["uploads_performed"] is False


def test_ambiguous_and_incomplete_transition_signals_are_recorded(
    tmp_path: Path,
) -> None:
    failure_miner = miner(
        tmp_path,
        FailureMiningConfig(enabled=True, maximum_records=10),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with failure_miner:
        failure_miner.observe_frame(
            frame_number=5,
            timestamp_seconds=0.5,
            frame=frame,
            fused_detections=[],
            tracks=[track(3, (10.0, 10.0, 30.0, 80.0))],
            snapshots={3: snapshot()},
            events=[],
        )

    assert failure_miner.summary["record_types"]["ambiguous_zone_transition"] == 1
    assert failure_miner.summary["record_types"]["incomplete_transition"] == 1


def test_failure_paths_are_derived_beside_video() -> None:
    jsonl, frames = derive_failure_paths(Path("output/run.mp4"))

    assert jsonl == Path("output/run.failures.jsonl")
    assert frames == Path("output/run.failure-frames")
