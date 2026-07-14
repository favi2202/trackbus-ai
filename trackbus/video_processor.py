"""Video decoding, TrackBus inference, annotation, and output encoding."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.calibration import DoorwayLane, FrameCalibration
from trackbus.config import CameraConfig, DiagnosticsConfig, ZonesConfig
from trackbus.counter import PassengerCounter
from trackbus.diagnostics import DoorwayDiagnostics
from trackbus.event_logger import EventLogger
from trackbus.tracker import ByteTrackPersonTracker, TrackedPerson
from trackbus.zones import PixelZones, ZoneMembership

LOGGER = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be decoded, displayed, or encoded."""


@dataclass(frozen=True)
class ProcessingSummary:
    input_video: str
    source_width: int
    source_height: int
    source_fps: float
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
    confidence_threshold: float
    inference_image_size: int
    device_used: str
    person_detections_total: int
    frames_with_person_detections: int
    frames_without_person_detections: int
    average_person_detections_per_frame: float
    maximum_person_detections_in_frame: int
    unique_tracking_ids: int
    detection_roi_enabled: bool
    detection_roi_normalized: tuple[float, float, float, float] | None
    excluded_person_detections_total: int
    doorway_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        diagnostics = data.pop("doorway_diagnostics")
        data.update(diagnostics)
        return data


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
        model_confidence: float,
        inference_image_size: int,
        camera_config: CameraConfig,
        diagnostics_config: DiagnosticsConfig,
    ) -> None:
        self.tracker = tracker
        self.counter = counter
        self.zones_config = zones_config
        self.event_logger = event_logger
        self.model_name = model_name
        self.model_confidence = model_confidence
        self.inference_image_size = inference_image_size
        self.camera_config = camera_config
        self.diagnostics_config = diagnostics_config

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
        calibration = FrameCalibration(self.camera_config, width, height)
        doorway_diagnostics = DoorwayDiagnostics(calibration, self.diagnostics_config)
        processed_frames = 0
        person_detections_total = 0
        frames_with_person_detections = 0
        maximum_person_detections_in_frame = 0
        unique_tracking_ids: set[int] = set()
        excluded_person_detections_total = 0
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
                local_people = self.tracker.update(calibration.inference_frame(frame))
                source_people = calibration.to_source(local_people)
                excluded_people: list[TrackedPerson] = []
                people: list[TrackedPerson] = []
                for person in source_people:
                    target = (
                        excluded_people if calibration.is_excluded(person) else people
                    )
                    target.append(person)
                excluded_person_detections_total += len(excluded_people)
                detections_in_frame = len(people)
                person_detections_total += detections_in_frame
                if detections_in_frame:
                    frames_with_person_detections += 1
                maximum_person_detections_in_frame = max(
                    maximum_person_detections_in_frame, detections_in_frame
                )
                unique_tracking_ids.update(person.tracking_id for person in people)
                lanes = {
                    person.tracking_id: calibration.lane_for(person)
                    for person in people
                }
                box_lanes = {
                    person.tracking_id: calibration.box_lanes(
                        person,
                        self.diagnostics_config.multi_lane_minimum_box_overlap,
                    )
                    for person in people
                }
                doorway_diagnostics.observe_frame(
                    frame_number, people, lanes, box_lanes
                )
                memberships: dict[int, ZoneMembership] = {}

                for person in people:
                    membership = zones.membership(person.anchor)
                    memberships[person.tracking_id] = membership
                    event = self.counter.observe(
                        person.tracking_id, membership, frame_number
                    )
                    if event is not None:
                        doorway_diagnostics.record_crossing(event)
                        self.event_logger.log_event(event, source_fps)
                        LOGGER.info(
                            "%s event: track=%d frame=%d occupancy=%d",
                            event.event_type.value,
                            event.tracking_id,
                            event.video_frame,
                            event.current_occupancy,
                        )

                annotated = self._annotate(
                    frame,
                    zones,
                    people,
                    memberships,
                    calibration,
                    lanes,
                    excluded_people,
                )
                writer.write(annotated)
                processed_frames += 1

                if show:
                    try:
                        cv2.imshow("TrackBus v0.1.2 - press q to stop", annotated)
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
            source_width=width,
            source_height=height,
            source_fps=round(source_fps, 3),
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
            confidence_threshold=self.model_confidence,
            inference_image_size=self.inference_image_size,
            device_used=self.tracker.device_label,
            person_detections_total=person_detections_total,
            frames_with_person_detections=frames_with_person_detections,
            frames_without_person_detections=(
                processed_frames - frames_with_person_detections
            ),
            average_person_detections_per_frame=(
                round(person_detections_total / processed_frames, 3)
                if processed_frames
                else 0.0
            ),
            maximum_person_detections_in_frame=maximum_person_detections_in_frame,
            unique_tracking_ids=len(unique_tracking_ids),
            detection_roi_enabled=calibration.roi.enabled,
            detection_roi_normalized=self.camera_config.detection_roi,
            excluded_person_detections_total=excluded_person_detections_total,
            doorway_diagnostics=doorway_diagnostics.summary(),
        )
        self.event_logger.write_summary(summary.to_dict())
        return summary

    def _annotate(
        self,
        frame: NDArray[np.uint8],
        zones: PixelZones,
        people: list[TrackedPerson],
        memberships: dict[int, ZoneMembership],
        calibration: FrameCalibration,
        lanes: dict[int, DoorwayLane | None],
        excluded_people: list[TrackedPerson],
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

        if calibration.config.debug_calibration_overlay:
            _draw_calibration_overlay(annotated, calibration)

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
            lane = lanes[person.tracking_id]
            lane_text = f" {lane.value}" if lane is not None else ""
            label = (
                f"ID {person.tracking_id} {person.confidence:.2f} "
                f"{state_text}{lane_text}"
            )
            _text_with_background(annotated, label, (left, max(18, top - 7)), color)
            anchor = tuple(int(value) for value in person.anchor)
            cv2.circle(annotated, anchor, 4, color, -1)

        if calibration.config.debug_calibration_overlay:
            for person in excluded_people:
                left, top, right, bottom = (int(value) for value in person.bounding_box)
                cv2.rectangle(annotated, (left, top), (right, bottom), (40, 40, 230), 2)
                _text_with_background(
                    annotated,
                    f"ID {person.tracking_id} EXCLUDED",
                    (left, max(18, top - 7)),
                    (40, 40, 230),
                )

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


def _draw_calibration_overlay(
    frame: NDArray[np.uint8], calibration: FrameCalibration
) -> None:
    if calibration.roi.enabled:
        roi = calibration.roi
        cv2.rectangle(
            frame,
            (roi.left, roi.top),
            (roi.right - 1, roi.bottom - 1),
            (255, 180, 40),
            2,
        )
        _text_with_background(
            frame, "DETECTION ROI", (roi.left + 5, roi.top + 20), (255, 180, 40)
        )

    lane_colors = {
        DoorwayLane.LEFT: (255, 120, 70),
        DoorwayLane.CENTER: (220, 220, 60),
        DoorwayLane.RIGHT: (180, 80, 255),
    }
    for lane, polygon in calibration.lanes.items():
        color = lane_colors[lane]
        cv2.polylines(frame, [polygon], True, color, 2)
        _zone_label(frame, polygon, lane.value.upper(), color)
    for index, polygon in enumerate(calibration.exclusions):
        cv2.polylines(frame, [polygon], True, (40, 40, 230), 2)
        _zone_label(frame, polygon, f"EXCLUSION {index + 1}", (40, 40, 230))


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
