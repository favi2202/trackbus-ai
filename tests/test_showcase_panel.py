from collections import deque
from datetime import UTC, datetime

import cv2
import numpy as np

from showcase import (
    RuntimeState,
    _draw_panel,
    _event_row,
    _prepare_display_frame,
    build_parser,
)
from trackbus.event_contract import vision_event


def _bounds(call: tuple[str, tuple[int, int], float, int]) -> tuple[int, int, int, int]:
    text, (x, baseline), scale, thickness = call
    (width, height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    return x, baseline - height, x + width, baseline


def _overlaps(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def test_narrow_showcase_header_lines_fit_without_overlap(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int], float, int]] = []
    original = cv2.putText

    def record(frame, text, origin, font, scale, color, thickness, *args):
        calls.append((text, origin, scale, thickness))
        return original(frame, text, origin, font, scale, color, thickness, *args)

    monkeypatch.setattr(cv2, "putText", record)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _draw_panel(
        frame,
        RuntimeState(occupancy=0),
        fps=56.8,
        active_tracks=1,
        events=deque(),
        queue_count=0,
        frame_number=1,
    )

    header_calls = calls[:4]
    assert [call[0] for call in header_calls] == [
        "TRACKBUS VISION",
        "EDGE AI | LIVE | ANONYMOUS TEMPORARY IDS",
        "IN 0   OUT 0   ONBOARD 0   TRACKS 1   FPS 56.8",
        "OFFLINE QUEUE | QUEUED 0",
    ]
    bounds = [_bounds(call) for call in header_calls]
    assert all(0 <= left < right <= 320 for left, _, right, _ in bounds)
    assert all(
        not _overlaps(a, b)
        for index, a in enumerate(bounds)
        for b in bounds[index + 1 :]
    )


def test_compact_event_rows_are_ascii_and_fit_narrow_panel(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int], float, int]] = []
    original = cv2.putText

    def record(frame, text, origin, font, scale, color, thickness, *args):
        calls.append((text, origin, scale, thickness))
        return original(frame, text, origin, font, scale, color, thickness, *args)

    monkeypatch.setattr(cv2, "putText", record)
    events = deque(
        vision_event(
            direction="IN" if index % 2 else "OUT",
            occupancy=index,
            capacity=72,
            confidence=0.46 + index / 10,
            bus_id="BUS",
            route_id="22",
            stop_id="STOP",
            door_id="DOOR",
            quality_flags=("low_detection_confidence",),
            observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        )
        for index in range(6)
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _draw_panel(
        frame,
        RuntimeState(occupancy=5),
        fps=30.0,
        active_tracks=1,
        events=events,
        queue_count=0,
        frame_number=1,
    )

    event_calls = calls[5:]
    assert len(event_calls) == 4
    assert all(call[0].isascii() for call in calls)
    assert all(_bounds(call)[2] <= 320 - 21 for call in event_calls)
    assert _event_row(events[-1], compact=True).endswith("| LOW")


def test_wide_showcase_header_keeps_two_columns_separate(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int], float, int]] = []
    original = cv2.putText

    def record(frame, text, origin, font, scale, color, thickness, *args):
        calls.append((text, origin, scale, thickness))
        return original(frame, text, origin, font, scale, color, thickness, *args)

    monkeypatch.setattr(cv2, "putText", record)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _draw_panel(
        frame,
        RuntimeState(occupancy=12, api_status="API ONLINE"),
        fps=30.0,
        active_tracks=2,
        events=deque(),
        queue_count=0,
        frame_number=1,
    )

    bounds = [_bounds(call) for call in calls[:4]]
    assert all(0 <= left < right <= 1280 for left, _, right, _ in bounds)
    assert all(
        not _overlaps(a, b)
        for index, a in enumerate(bounds)
        for b in bounds[index + 1 :]
    )


def test_small_source_is_upscaled_before_overlay_drawing() -> None:
    source = np.zeros((240, 320, 3), dtype=np.uint8)
    display, scale = _prepare_display_frame(source, expand=True)
    assert display.shape == (720, 960, 3)
    assert scale == (3.0, 3.0)


def test_headless_source_keeps_original_resolution() -> None:
    source = np.zeros((240, 320, 3), dtype=np.uint8)
    display, scale = _prepare_display_frame(source, expand=False)
    assert display is source
    assert scale == (1.0, 1.0)


def test_showcase_exposes_a_separate_detector_floor() -> None:
    args = build_parser().parse_args(
        ["--video", "bus.mp4", "--confidence", "0.35", "--detector-floor", "0.10"]
    )

    assert args.confidence == 0.35
    assert args.detector_floor == 0.10
