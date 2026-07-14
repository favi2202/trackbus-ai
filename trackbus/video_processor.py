"""Video decoding, TrackBus inference, annotation, and output encoding."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.config import ZonesConfig
from trackbus.counter import PassengerCounter
from trackbus.event_logger import EventLogger
from trackbus.tracker import ByteTrackPersonTracker, TrackedPerson
from trackbus.zones import PixelZones, ZoneMembership

LOGGER = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be decoded, displayed, or encoded."""


@dataclass(frozen=True)
class ProcessingSummary:
    input_video: str
    processed_frames: int
    entered_total: int
    exited_total: int
    initial_occupancy: int
    final_occupancy: int
    bus_capacity: int
    final_occupancy_percentage: float
    processing_time_seconds: float
    average_fps: float
    model_used: str
    device_used: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class VideoProcessor:
    """Run tracking and counting for one local video."""

    def __init__(
        self,
        *,
        tracker: ByteTrackPersonTracker,
        counter: PassengerCounter,
        zones_config: ZonesConfig,
        event_logger: EventLogger,
        model_name: str,
    ) -> None:
        self.tracker = tracker
        self.counter = counter
        self.zones_config = zones_config
        self.event_logger = event_logger
        self.model_name = model_name

    def process(
        self, input_path: Path, output_path: Path, *, show: bool = False
    ) -> ProcessingSummary:
        """Process a video and return the statistics written to the JSON summary."""

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise VideoProcessingError(f"Could not open input video: {input_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            capture.release()
            raise VideoProcessingError("Input video reports invalid frame dimensions.")
        if source_fps <= 0 or not np.isfinite(source_fps):
            LOGGER.warning("Input video has no valid FPS metadata; using 30 FPS.")
            source_fps = 30.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise VideoProcessingError(f"Could not create output video: {output_path}")

        zones = PixelZones.from_normalized(self.zones_config, width, height)
        processed_frames = 0
        started = time.perf_counter()
        LOGGER.info(
            "Processing %s (%dx%d at %.2f FPS)", input_path, width, height, source_fps
        )

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number = processed_frames
                self.counter.remove_stale_tracks(frame_number)
                people = self.tracker.update(frame)
                memberships: dict[int, ZoneMembership] = {}

                for person in people:
                    membership = zones.membership(person.anchor)
                    memberships[person.tracking_id] = membership
                    event = self.counter.observe(
                        person.tracking_id, membership, frame_number
                    )
                    if event is not None:
                        self.event_logger.log_event(event, source_fps)
                        LOGGER.info(
                            "%s event: track=%d frame=%d occupancy=%d",
                            event.event_type.value,
                            event.tracking_id,
                            event.video_frame,
                            event.current_occupancy,
                        )

                annotated = self._annotate(frame, zones, people, memberships)
                writer.write(annotated)
                processed_frames += 1

                if show:
                    try:
                        cv2.imshow("TrackBus v0.1 - press q to stop", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            LOGGER.info("Preview stopped by user.")
                            break
                    except cv2.error as exc:
                        raise VideoProcessingError(
                            "OpenCV cannot show a window in this environment. "
                            "Run again without --show."
                        ) from exc
        finally:
            capture.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - started
        summary = ProcessingSummary(
            input_video=str(input_path),
            processed_frames=processed_frames,
            entered_total=self.counter.entered_total,
            exited_total=self.counter.exited_total,
            initial_occupancy=self.counter.initial_occupancy,
            final_occupancy=self.counter.current_occupancy,
            bus_capacity=self.counter.capacity,
            final_occupancy_percentage=round(self.counter.occupancy_percentage, 2),
            processing_time_seconds=round(elapsed, 3),
            average_fps=round(processed_frames / elapsed, 2) if elapsed else 0.0,
            model_used=self.model_name,
            device_used=self.tracker.device_label,
        )
        self.event_logger.write_summary(summary.to_dict())
        return summary

    def _annotate(
        self,
        frame: NDArray[np.uint8],
        zones: PixelZones,
        people: list[TrackedPerson],
        memberships: dict[int, ZoneMembership],
    ) -> NDArray[np.uint8]:
        annotated = frame.copy()
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [zones.outside], (0, 140, 255))
        cv2.fillPoly(overlay, [zones.inside], (60, 180, 75))
        cv2.addWeighted(overlay, 0.16, annotated, 0.84, 0, annotated)
        cv2.polylines(annotated, [zones.outside], True, (0, 165, 255), 2)
        cv2.polylines(annotated, [zones.inside], True, (60, 200, 90), 2)
        _zone_label(annotated, zones.outside, "OUTSIDE", (0, 165, 255))
        _zone_label(annotated, zones.inside, "INSIDE", (60, 200, 90))

        for person in people:
            left, top, right, bottom = (int(value) for value in person.bounding_box)
            membership = memberships[person.tracking_id]
            color = (
                (60, 200, 90)
                if membership is ZoneMembership.INSIDE
                else (0, 165, 255)
                if membership is ZoneMembership.OUTSIDE
                else (230, 200, 40)
            )
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            state = self.counter.track_state(person.tracking_id)
            state_text = state.value if state is not None else "new"
            label = f"ID {person.tracking_id} {person.confidence:.2f} {state_text}"
            _text_with_background(annotated, label, (left, max(18, top - 7)), color)
            anchor = tuple(int(value) for value in person.anchor)
            cv2.circle(annotated, anchor, 4, color, -1)

        occupancy = self.counter.current_occupancy
        actual_percentage = self.counter.occupancy_percentage
        # Keep the overlay readable if configuration or counts are extreme.
        displayed_percentage = min(actual_percentage, 999.0)
        capacity_status = " OVER CAPACITY" if occupancy > self.counter.capacity else ""
        lines = [
            f"Tracked: {len(people)}",
            (
                f"Entered: {self.counter.entered_total}  "
                f"Exited: {self.counter.exited_total}"
            ),
            f"Occupancy: {occupancy}/{self.counter.capacity}",
            f"Capacity: {displayed_percentage:.1f}%{capacity_status}",
        ]
        _draw_status_panel(annotated, lines, occupancy > self.counter.capacity)
        return annotated


def _zone_label(
    frame: NDArray[np.uint8],
    polygon: NDArray[np.int32],
    label: str,
    color: tuple[int, int, int],
) -> None:
    left = int(polygon[:, 0].min()) + 6
    top = int(polygon[:, 1].min()) + 22
    _text_with_background(frame, label, (left, top), color)


def _text_with_background(
    frame: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - 2, y - height - 3),
        (x + width + 3, y + baseline + 2),
        (25, 25, 25),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_status_panel(
    frame: NDArray[np.uint8], lines: list[str], over_capacity: bool
) -> None:
    panel_width = 360
    line_height = 25
    panel_height = len(lines) * line_height + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (panel_width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for index, line in enumerate(lines):
        color = (
            (40, 80, 255)
            if over_capacity and index == len(lines) - 1
            else (245, 245, 245)
        )
        cv2.putText(
            frame,
            line,
            (18, 30 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
