from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackbus.extract_head_frames import (
    HeadFrameExtractionError,
    extract_head_frames,
    main,
)


def _write_video(tmp_path: Path, frame_count: int = 8) -> Path:
    path = tmp_path / "bus-test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    return path


def test_extracts_requested_frames_and_writes_privacy_manifest(tmp_path: Path) -> None:
    video = _write_video(tmp_path)
    output = tmp_path / "head-frames"

    report = extract_head_frames(
        video,
        output,
        every=2,
        start_frame=1,
        end_frame=6,
        jpeg_quality=90,
    )

    assert [row["frame"] for row in report["frames"]] == [1, 3, 5]
    assert sorted(path.name for path in output.glob("*.jpg")) == [
        "bus-test-f000001.jpg",
        "bus-test-f000003.jpg",
        "bus-test-f000005.jpg",
    ]
    saved = json.loads((output / "extraction.json").read_text(encoding="utf-8"))
    assert saved["detection_target"] == "head"
    assert saved["annotation_contract"]["classes"] == ["head"]
    assert saved["annotation_contract"]["fabricated_labels"] is False
    assert saved["privacy"]["uploads_performed"] is False
    assert saved["accuracy_claimed"] is False


def test_extraction_respects_maximum_frames(tmp_path: Path) -> None:
    report = extract_head_frames(
        _write_video(tmp_path),
        tmp_path / "limited",
        every=1,
        maximum_frames=2,
    )

    assert report["selected_frame_count"] == 2
    assert [row["frame"] for row in report["frames"]] == [0, 1]


def test_extraction_refuses_nonempty_output(tmp_path: Path) -> None:
    video = _write_video(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(HeadFrameExtractionError, match="refusing to overwrite"):
        extract_head_frames(video, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_cli_reports_invalid_range_without_creating_labels(
    tmp_path: Path,
) -> None:
    video = _write_video(tmp_path)

    assert (
        main(
            [
                "--video",
                str(video),
                "--output",
                str(tmp_path / "invalid"),
                "--start-frame",
                "5",
                "--end-frame",
                "4",
            ]
        )
        == 2
    )
    assert not list(tmp_path.rglob("*.txt"))
