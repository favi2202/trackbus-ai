#!/usr/bin/env python3
"""TrackBus Vision: real anonymous YOLO + ByteTrack doorway demonstration."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from trackbus.api_client import TrackBusApiClient
from trackbus.detection import TrackedDetection
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device
from trackbus.event_contract import PassengerCountEvent, vision_event
from trackbus.showcase_zones import (
    CrossingEvent,
    CrossingStateMachine,
    Zone,
    ZoneLayout,
    load_zone_layout,
)
from trackbus.tracker import ByteTrackAdapter, TrackerAdapterError

LOGGER = logging.getLogger("trackbus.showcase")
WINDOW = "TrackBus Vision / Edge AI - Live"
MINIMUM_DISPLAY_WIDTH = 960
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ZONE_COLORS = {
    Zone.OUTSIDE: (231, 166, 64),
    Zone.DOOR: (82, 211, 255),
    Zone.INSIDE: (113, 225, 139),
}


@dataclass
class RuntimeState:
    occupancy: int
    boardings: int = 0
    alightings: int = 0
    paused: bool = False
    overlay: bool = True
    fullscreen: bool = False
    api_status: str = "OFFLINE QUEUE"
    flash_until: int = -1
    flash_text: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real YOLO + ByteTrack TrackBus doorway demo. Temporary IDs "
            "exist only for the current process; no face recognition is used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source", help="Camera index or local video path")
    sources.add_argument("--camera", type=int, help="OpenCV camera index")
    sources.add_argument("--video", type=Path, help="Local video file")
    sources.add_argument(
        "--demo", action="store_true", help="Discover local demo footage"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Draw and save three zones without loading YOLO",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        help="YAML file with normalized outside/door/inside polygons",
    )
    parser.add_argument(
        "--save-zones", type=Path, default=Path("configs/showcase_zones.yaml")
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Decode but skip this many frames between inference steps",
    )
    parser.add_argument("--tracker", default="configs/bytetrack_trackbus.yaml")
    parser.add_argument("--minimum-zone-frames", type=int, default=2)
    parser.add_argument("--maximum-transition-gap", type=int, default=45)
    parser.add_argument(
        "--notify-cooldown",
        type=int,
        default=18,
        help="Frames to show a confirmed-event banner",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="TrackBus API base URL; hosted showcase uses https://trackbus-showcase.favi-2202.chatgpt.site/api",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Ingestion key; prefer the TRACKBUS_API_KEY environment variable",
    )
    parser.add_argument("--queue-dir", type=Path, default=Path("data/offline-queue"))
    parser.add_argument("--bus-id", default="BUS-DEMO-01")
    parser.add_argument("--route-id", default="DEMO")
    parser.add_argument("--stop-id", default="DEMO-STOP")
    parser.add_argument("--door-id", default="DOOR-1")
    parser.add_argument("--capacity", type=int, default=72)
    parser.add_argument("--initial-occupancy", type=int, default=0)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="Run inference without an OpenCV window"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames (smoke tests)",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def discover_demo() -> Path:
    roots = (Path("data/input"), Path("demo_for_showcasing"), Path("data/sample"))
    candidates = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            "No demo footage was found. Put an approved .mp4/.mov/.mkv/.avi file "
            "under data/input, or use --video PATH / --camera 0."
        )
    return candidates[0]


def resolve_source(args: argparse.Namespace) -> int | str:
    if args.camera is not None:
        return args.camera
    if args.video is not None:
        path = args.video.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        return str(path)
    if args.demo:
        return str(discover_demo().resolve())
    if args.source is not None:
        return (
            int(args.source)
            if args.source.isdigit()
            else str(Path(args.source).expanduser().resolve())
        )
    return 0


def _capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def calibrate(source: int | str, destination: Path) -> int:
    """Model-free three-polygon editor that works with cameras and video."""

    capture = _capture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("The source opened but did not return a calibration frame")

    points: dict[Zone, list[tuple[int, int]]] = {zone: [] for zone in ZONE_COLORS}
    order = [Zone.OUTSIDE, Zone.DOOR, Zone.INSIDE]
    active = 0
    status = "Click OUTSIDE points, then Enter"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        nonlocal status
        zone = order[active]
        if event == cv2.EVENT_LBUTTONDOWN:
            points[zone].append((x, y))
            status = f"{zone.value.upper()}: {len(points[zone])} points"
        elif event == cv2.EVENT_RBUTTONDOWN and points[zone]:
            points[zone].pop()
            status = f"{zone.value.upper()}: removed last point"

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        while True:
            display = frame.copy()
            for zone, polygon in points.items():
                if polygon:
                    values = np.asarray(polygon, dtype=np.int32)
                    cv2.polylines(
                        display, [values], len(polygon) >= 3, ZONE_COLORS[zone], 2
                    )
                    for point in polygon:
                        cv2.circle(display, point, 4, ZONE_COLORS[zone], -1)
            cv2.rectangle(display, (0, 0), (display.shape[1], 64), (5, 15, 23), -1)
            cv2.putText(
                display,
                "TRACKBUS CALIBRATION",
                (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (90, 225, 190),
                2,
            )
            cv2.putText(
                display,
                f"{status} | Right-click undo | R reset | Q cancel",
                (16, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 230, 235),
                1,
            )
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                print("Calibration cancelled; no file was changed.")
                return 1
            if key in (ord("r"), ord("R")):
                points = {zone: [] for zone in ZONE_COLORS}
                active = 0
                status = "Click OUTSIDE points, then Enter"
            if key in (10, 13):
                zone = order[active]
                if len(points[zone]) < 3:
                    status = f"{zone.value.upper()} needs at least 3 points"
                elif active < len(order) - 1:
                    active += 1
                    status = f"Click {order[active].value.upper()} points, then Enter"
                else:
                    height, width = frame.shape[:2]
                    document = {
                        "zones": {
                            zone.value: [
                                [
                                    round(x / max(1, width - 1), 6),
                                    round(y / max(1, height - 1), 6),
                                ]
                                for x, y in points[zone]
                            ]
                            for zone in order
                        }
                    }
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
                    )
                    print(f"Saved normalized TrackBus showcase zones to {destination}")
                    return 0
    finally:
        cv2.destroyWindow(WINDOW)


def _draw_zone_overlay(frame: np.ndarray, layout: ZoneLayout) -> None:
    translucent = frame.copy()
    for zone, polygon in layout.pixel_polygons(frame.shape).items():
        values = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(translucent, [values], ZONE_COLORS[zone])
    cv2.addWeighted(translucent, 0.11, frame, 0.89, 0, frame)
    for zone, polygon in layout.pixel_polygons(frame.shape).items():
        values = np.asarray(polygon, dtype=np.int32)
        cv2.polylines(frame, [values], True, ZONE_COLORS[zone], 2)
        x, y = values[0]
        cv2.putText(
            frame,
            zone.value.upper(),
            (int(x) + 6, int(y) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            ZONE_COLORS[zone],
            2,
        )


def _draw_track(
    frame: np.ndarray,
    track: TrackedDetection,
    zone: Zone,
    trail: deque[tuple[int, int]],
    scale: tuple[float, float] = (1.0, 1.0),
) -> None:
    scale_x, scale_y = scale
    left, top, right, bottom = (
        round(track.bounding_box[0] * scale_x),
        round(track.bounding_box[1] * scale_y),
        round(track.bounding_box[2] * scale_x),
        round(track.bounding_box[3] * scale_y),
    )
    color = ZONE_COLORS.get(zone, (170, 180, 190))
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(
        frame,
        f"T{track.tracking_id} {track.confidence:.2f}",
        (left, max(18, top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        2,
    )
    trail.append((round(track.anchor[0]), round(track.anchor[1])))
    display_trail = np.asarray(
        [
            (round(x * scale_x), round(y * scale_y))
            for x, y in trail
        ],
        dtype=np.int32,
    )
    if len(trail) > 1:
        cv2.polylines(frame, [display_trail], False, color, 2)
    cv2.circle(frame, tuple(display_trail[-1]), 4, color, -1)


def _prepare_display_frame(
    frame: np.ndarray, *, expand: bool
) -> tuple[np.ndarray, tuple[float, float]]:
    """Upscale small sources before UI drawing instead of enlarging drawn text."""

    height, width = frame.shape[:2]
    if not expand or width >= MINIMUM_DISPLAY_WIDTH:
        return frame, (1.0, 1.0)
    target_width = MINIMUM_DISPLAY_WIDTH
    target_height = max(1, round(height * target_width / width))
    display = cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    return display, (target_width / width, target_height / height)


def _text_width(text: str, scale: float, thickness: int = 1) -> int:
    return cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0]


def _fitted_scale(
    text: str,
    *,
    max_width: int,
    preferred: float,
    thickness: int = 1,
    minimum: float = 0.2,
) -> float:
    """Return a Hershey-font scale that keeps one line inside max_width."""

    if max_width <= 0:
        return minimum
    width = _text_width(text, preferred, thickness)
    if width <= max_width:
        return preferred
    scale = preferred * max_width / max(1, width)
    return max(minimum, min(preferred, scale))


def _put_fitted_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    max_width: int,
    preferred_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> float:
    scale = _fitted_scale(
        text,
        max_width=max_width,
        preferred=preferred_scale,
        thickness=thickness,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return scale


def _event_quality_label(event: PassengerCountEvent, *, compact: bool) -> str:
    if not event.quality_flags:
        return "OK"
    labels = {
        "low_detection_confidence": "LOW" if compact else "LOW CONF",
        "occupancy_boundary_clamped": "LIMIT" if compact else "OCC LIMIT",
    }
    rendered = [
        labels.get(flag, flag.replace("_", " ").upper())
        for flag in event.quality_flags
    ]
    return "+".join(rendered)


def _event_row(event: PassengerCountEvent, *, compact: bool) -> str:
    if compact:
        direction = "IN" if event.boardings else "OUT"
        return (
            f"{direction} | OCC {event.occupancy} | {event.confidence:.0%} | "
            f"{_event_quality_label(event, compact=True)}"
        )
    direction = "BOARDING" if event.boardings else "ALIGHTING"
    return (
        f"{direction} | OCC {event.occupancy} | {event.confidence:.0%} | "
        f"{_event_quality_label(event, compact=False)}"
    )


def _draw_panel(
    frame: np.ndarray,
    state: RuntimeState,
    *,
    fps: float,
    active_tracks: int,
    events: deque[PassengerCountEvent],
    queue_count: int,
    frame_number: int,
) -> None:
    height, width = frame.shape[:2]
    padding = 18
    title = "TRACKBUS VISION"
    subtitle = "EDGE AI | LIVE | ANONYMOUS TEMPORARY IDS"
    summary = (
        f"IN {state.boardings}   OUT {state.alightings}   "
        f"ONBOARD {state.occupancy}   TRACKS {active_tracks}   FPS {fps:.1f}"
    )
    api_text = f"{state.api_status} | QUEUED {queue_count}"
    left_width = max(_text_width(title, 0.76, 2), _text_width(subtitle, 0.42))
    right_width = max(_text_width(summary, 0.53, 2), _text_width(api_text, 0.4))
    wide_header = padding * 2 + left_width + 30 + right_width <= width
    header_height = 76 if wide_header else 104
    cv2.rectangle(frame, (0, 0), (width, header_height), (5, 15, 23), -1)

    if wide_header:
        right_x = width - padding - right_width
        placements = (
            (title, (padding, 29), left_width, 0.76, (90, 225, 190), 2),
            (subtitle, (padding, 54), left_width, 0.42, (165, 185, 195), 1),
            (summary, (right_x, 31), right_width, 0.53, (235, 241, 243), 2),
        )
        api_origin = (right_x, 55)
        api_max_width = right_width
    else:
        available = max(1, width - padding * 2)
        placements = (
            (title, (padding, 25), available, 0.68, (90, 225, 190), 2),
            (subtitle, (padding, 47), available, 0.4, (165, 185, 195), 1),
            (summary, (padding, 72), available, 0.48, (235, 241, 243), 2),
        )
        api_origin = (padding, 94)
        api_max_width = available

    for text, origin, max_width, scale, color, thickness in placements:
        _put_fitted_text(
            frame,
            text,
            origin,
            max_width=max_width,
            preferred_scale=scale,
            color=color,
            thickness=thickness,
        )

    api_color = (90, 225, 190) if state.api_status == "API ONLINE" else (82, 211, 255)
    _put_fitted_text(
        frame,
        api_text,
        api_origin,
        max_width=api_max_width,
        preferred_scale=0.4,
        color=api_color,
        thickness=1,
    )

    panel_width = min(360, max(250, width // 3))
    panel_top = header_height + 6
    panel_bottom = min(height - 8, panel_top + 163)
    if panel_bottom <= panel_top + 30:
        return
    cv2.rectangle(
        frame,
        (width - panel_width, panel_top),
        (width - 8, panel_bottom),
        (7, 22, 32),
        -1,
    )
    _put_fitted_text(
        frame,
        "CONFIRMED EVENTS",
        (width - panel_width + 13, panel_top + 24),
        max_width=panel_width - 34,
        preferred_scale=0.42,
        color=(90, 225, 190),
    )
    first_event_baseline = panel_top + 50
    row_height = 23
    max_rows = max(0, (panel_bottom - first_event_baseline) // row_height + 1)
    compact_events = panel_width < 320
    visible_events = list(events)[-max_rows:] if max_rows else []
    for index, event in enumerate(visible_events[::-1]):
        text = _event_row(event, compact=compact_events)
        _put_fitted_text(
            frame,
            text,
            (width - panel_width + 13, first_event_baseline + index * row_height),
            max_width=panel_width - 34,
            preferred_scale=0.35,
            color=(205, 218, 224),
        )
    if frame_number <= state.flash_until:
        box_width = min(560, width - 40)
        x1 = (width - box_width) // 2
        y1 = max(header_height + 14, height - 110)
        cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + 62), (12, 60, 55), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + 62), (90, 225, 190), 2)
        _put_fitted_text(
            frame,
            state.flash_text,
            (x1 + 18, y1 + 39),
            max_width=box_width - 36,
            preferred_scale=0.72,
            color=(110, 240, 205),
            thickness=2,
        )


def _event_for_crossing(
    args: argparse.Namespace, state: RuntimeState, crossing: CrossingEvent
) -> PassengerCountEvent:
    requested = state.occupancy + crossing.occupancy_delta
    bounded = min(args.capacity, max(0, requested))
    flags: list[str] = []
    if bounded != requested:
        flags.append("occupancy_boundary_clamped")
    if crossing.confidence < 0.75:
        flags.append("low_detection_confidence")
    state.occupancy = bounded
    if crossing.direction == "IN":
        state.boardings += 1
    else:
        state.alightings += 1
    return vision_event(
        direction=crossing.direction,
        occupancy=bounded,
        capacity=args.capacity,
        confidence=crossing.confidence,
        bus_id=args.bus_id,
        route_id=args.route_id,
        stop_id=args.stop_id,
        door_id=args.door_id,
        quality_flags=tuple(flags),
    )


def export_events(events: Sequence[PassengerCountEvent]) -> Path:
    destination = Path(f"showcase-events-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(events[0].to_payload()) if events else ["eventId"]
        )
        writer.writeheader()
        for event in events:
            payload = event.to_payload()
            payload["qualityFlags"] = "|".join(event.quality_flags)
            writer.writerow(payload)
    return destination


def run(args: argparse.Namespace) -> int:
    if not 0 < args.confidence <= 1 or args.imgsz < 32 or args.frame_skip < 0:
        raise ValueError("confidence, image size, and frame skip values are invalid")
    if args.capacity <= 0 or not 0 <= args.initial_occupancy <= args.capacity:
        raise ValueError("initial occupancy must fit a positive capacity")
    source = resolve_source(args)
    if args.calibrate:
        if args.headless:
            raise ValueError("calibration requires a graphical desktop")
        return calibrate(source, args.save_zones)

    layout = load_zone_layout(args.zones)
    device, device_label = resolve_device(args.device)
    detector = UltralyticsDetector(
        args.model,
        args.confidence,
        args.imgsz,
        device=device,
        device_label=device_label,
    )
    tracker = ByteTrackAdapter(args.tracker, device_label=device_label)
    counter = CrossingStateMachine(
        minimum_zone_frames=args.minimum_zone_frames,
        maximum_gap_frames=args.maximum_transition_gap,
    )
    api = TrackBusApiClient(
        args.api_url,
        args.queue_dir,
        api_key=args.api_key or os.environ.get("TRACKBUS_API_KEY"),
    )
    capture = _capture(source)
    state = RuntimeState(occupancy=args.initial_occupancy, fullscreen=args.fullscreen)
    events: deque[PassengerCountEvent] = deque(maxlen=250)
    trails: dict[int, deque[tuple[int, int]]] = {}
    frame_number = 0
    processed = 0
    fps_samples: deque[float] = deque(maxlen=30)
    last_frame: np.ndarray | None = None

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if state.fullscreen:
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
    delivered, remaining = api.flush()
    state.api_status = (
        "API ONLINE" if args.api_url and remaining == 0 else "OFFLINE QUEUE"
    )
    LOGGER.info(
        "startup source=%s device=%s flushed=%d remaining=%d",
        source,
        device_label,
        delivered,
        remaining,
    )

    try:
        while True:
            if not state.paused or last_frame is None:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if args.frame_skip and (frame_number - 1) % (args.frame_skip + 1):
                    continue
                started = time.perf_counter()
                detections = detector.detect(frame, source_view="full")
                tracks = tracker.update(detections, frame)
                counter.expire(frame_number)
                track_zones: dict[int, Zone] = {}
                for track in tracks:
                    zone = layout.classify(track.anchor, frame.shape)
                    track_zones[track.tracking_id] = zone
                    crossing = counter.observe(
                        track_id=track.tracking_id,
                        zone=zone,
                        frame_number=frame_number,
                        confidence=track.confidence,
                    )
                    trails.setdefault(track.tracking_id, deque(maxlen=32))
                    if crossing:
                        event = _event_for_crossing(args, state, crossing)
                        result = api.send(event)
                        state.api_status = (
                            "API ONLINE" if result.delivered else "OFFLINE QUEUE"
                        )
                        events.append(event)
                        state.flash_until = frame_number + args.notify_cooldown
                        state.flash_text = (
                            f"{crossing.direction} CONFIRMED | "
                            f"OCCUPANCY {state.occupancy}"
                        )
                        print(
                            f"{event.observed_at} {crossing.direction} "
                            f"track=T{crossing.track_id} occupancy={state.occupancy} "
                            f"confidence={crossing.confidence:.3f} "
                            f"delivered={result.delivered} queued={result.queued}",
                            flush=True,
                        )
                active = {track.tracking_id for track in tracks}
                trails = {
                    track_id: trail
                    for track_id, trail in trails.items()
                    if track_id in active
                }
                elapsed = time.perf_counter() - started
                fps_samples.append(1 / elapsed if elapsed > 0 else 0)
                measured_fps = sum(fps_samples) / len(fps_samples)
                display_frame, display_scale = _prepare_display_frame(
                    frame, expand=not args.headless
                )
                if state.overlay:
                    for track in tracks:
                        _draw_track(
                            display_frame,
                            track,
                            track_zones[track.tracking_id],
                            trails[track.tracking_id],
                            display_scale,
                        )
                    _draw_zone_overlay(display_frame, layout)
                    _draw_panel(
                        display_frame,
                        state,
                        fps=measured_fps,
                        active_tracks=len(tracks),
                        events=events,
                        queue_count=api.queued_count,
                        frame_number=frame_number,
                    )
                last_frame = display_frame
                processed += 1

            if args.headless:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                continue
            assert last_frame is not None
            cv2.imshow(WINDOW, last_frame)
            key = cv2.waitKey(30 if state.paused else 1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                state.paused = not state.paused
            elif key in (ord("f"), ord("F")):
                state.fullscreen = not state.fullscreen
                cv2.setWindowProperty(
                    WINDOW,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key in (ord("o"), ord("O")):
                state.overlay = not state.overlay
            elif key in (ord("r"), ord("R")):
                state = RuntimeState(
                    occupancy=0, fullscreen=state.fullscreen, overlay=state.overlay
                )
                counter.reset()
                tracker.reset()
                trails.clear()
            elif key in (ord("s"), ord("S")):
                path = Path(f"showcase-{time.strftime('%Y%m%d-%H%M%S')}.png")
                cv2.imwrite(str(path), last_frame)
                LOGGER.info("saved screenshot %s", path)
            elif key in (ord("e"), ord("E")):
                LOGGER.info("exported events %s", export_events(list(events)))
            elif key in (ord("c"), ord("C"), ord("z"), ord("Z")):
                LOGGER.warning(
                    "Zone editing restarts capture; run --calibrate for an "
                    "auditable saved layout"
                )
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(
        f"TrackBus Vision stopped · processed={processed} in={state.boardings} "
        f"out={state.alightings} occupancy={state.occupancy} queued={api.queued_count}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(message)s"
    )
    try:
        return run(args)
    except (
        DetectorError,
        TrackerAdapterError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"TrackBus showcase error: {exc}", file=sys.stderr)
        return 2
    except cv2.error as exc:
        print(
            f"TrackBus showcase error: OpenCV display/video failure: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
