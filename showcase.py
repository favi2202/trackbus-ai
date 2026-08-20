#!/usr/bin/env python3
"""TrackBus Vision: real anonymous YOLO + ByteTrack doorway demonstration."""

from __future__ import annotations

import argparse
import csv
import logging
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
        help="Analytics API base URL; omitted means disk queue only",
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
) -> None:
    left, top, right, bottom = (round(value) for value in track.bounding_box)
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
    if len(trail) > 1:
        cv2.polylines(frame, [np.asarray(trail, dtype=np.int32)], False, color, 2)
    cv2.circle(frame, trail[-1], 4, color, -1)


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
    cv2.rectangle(frame, (0, 0), (width, 76), (5, 15, 23), -1)
    cv2.putText(
        frame,
        "TRACKBUS VISION",
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (90, 225, 190),
        2,
    )
    cv2.putText(
        frame,
        "EDGE AI · LIVE · ANONYMOUS TEMPORARY IDS",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (165, 185, 195),
        1,
    )
    summary = (
        f"IN {state.boardings}   OUT {state.alightings}   "
        f"ONBOARD {state.occupancy}   TRACKS {active_tracks}   FPS {fps:.1f}"
    )
    cv2.putText(
        frame,
        summary,
        (max(18, width - 570), 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (235, 241, 243),
        2,
    )
    api_color = (90, 225, 190) if state.api_status == "API ONLINE" else (82, 211, 255)
    cv2.putText(
        frame,
        f"{state.api_status} · QUEUED {queue_count}",
        (max(18, width - 570), 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        api_color,
        1,
    )

    panel_width = min(360, max(250, width // 3))
    cv2.rectangle(
        frame,
        (width - panel_width, 82),
        (width - 8, min(height - 8, 245)),
        (7, 22, 32),
        -1,
    )
    cv2.putText(
        frame,
        "CONFIRMED EVENTS",
        (width - panel_width + 13, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (90, 225, 190),
        1,
    )
    for index, event in enumerate(list(events)[-5:][::-1]):
        direction = "BOARDING" if event.boardings else "ALIGHTING"
        flags = ",".join(event.quality_flags) or "quality ok"
        text = (
            f"{direction:<9} occ {event.occupancy:>2} · "
            f"{event.confidence:.0%} · {flags}"
        )
        cv2.putText(
            frame,
            text,
            (width - panel_width + 13, 132 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (205, 218, 224),
            1,
        )
    if frame_number <= state.flash_until:
        box_width = min(560, width - 40)
        x1 = (width - box_width) // 2
        y1 = max(90, height - 110)
        cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + 62), (12, 60, 55), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + 62), (90, 225, 190), 2)
        cv2.putText(
            frame,
            state.flash_text,
            (x1 + 18, y1 + 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (110, 240, 205),
            2,
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
    api = TrackBusApiClient(args.api_url, args.queue_dir)
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
                for track in tracks:
                    zone = layout.classify(track.anchor, frame.shape)
                    crossing = counter.observe(
                        track_id=track.tracking_id,
                        zone=zone,
                        frame_number=frame_number,
                        confidence=track.confidence,
                    )
                    trail = trails.setdefault(track.tracking_id, deque(maxlen=32))
                    if state.overlay:
                        _draw_track(frame, track, zone, trail)
                    if crossing:
                        event = _event_for_crossing(args, state, crossing)
                        result = api.send(event)
                        state.api_status = (
                            "API ONLINE" if result.delivered else "OFFLINE QUEUE"
                        )
                        events.append(event)
                        state.flash_until = frame_number + args.notify_cooldown
                        state.flash_text = (
                            f"{crossing.direction} CONFIRMED · "
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
                if state.overlay:
                    _draw_zone_overlay(frame, layout)
                    _draw_panel(
                        frame,
                        state,
                        fps=measured_fps,
                        active_tracks=len(tracks),
                        events=events,
                        queue_count=api.queued_count,
                        frame_number=frame_number,
                    )
                last_frame = frame
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
