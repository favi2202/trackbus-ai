"""Video decoding, explicit detection/tracking stages, and output encoding."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.calibration import DoorwayLane, FrameCalibration
from trackbus.config import (
    CameraConfig,
    DiagnosticsConfig,
    InferenceViewConfig,
    ZonesConfig,
    resolve_inference_views,
)
from trackbus.counter import PassengerCounter
from trackbus.detection import Detection, TrackedDetection
from trackbus.detection_export import DetectionCsvExporter
from trackbus.diagnostics import DoorwayDiagnostics
from trackbus.event_logger import EventLogger
from trackbus.interfaces import DetectionFusion, DetectorBackend, TrackerBackend
from trackbus.views import (
    MultiViewInference,
    PixelInferenceView,
    box_has_positive_area,
    clip_box_to_frame,
)
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
    tracker_backend: str
    tracker_config_path: str | None
    tracker_effective_config: dict[str, Any]
    ultralytics_version: str | None
    person_detections_total: int
    raw_person_detections_total: int
    fused_detections_total: int
    tracked_person_observations_total: int
    frames_with_person_detections: int
    frames_without_person_detections: int
    frames_with_raw_detections: int
    frames_without_raw_detections: int
    frames_with_fused_detections: int
    frames_without_fused_detections: int
    average_person_detections_per_frame: float
    maximum_person_detections_in_frame: int
    unique_tracking_ids: int
    tracker_updates: int
    enabled_inference_views: tuple[str, ...]
    detection_fusion_method: str
    detection_fusion_iou_threshold: float | None
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
    """Run one source-frame pipeline and update one tracker once per frame."""

    def __init__(
        self,
        *,
        tracker: TrackerBackend | Any,
        counter: PassengerCounter,
        zones_config: ZonesConfig,
        event_logger: EventLogger,
        model_name: str,
        model_confidence: float,
        inference_image_size: int,
        camera_config: CameraConfig,
        diagnostics_config: DiagnosticsConfig,
        detector: DetectorBackend | None = None,
        fusion: DetectionFusion | None = None,
        inference_views: tuple[InferenceViewConfig, ...] | None = None,
        detection_exporter: DetectionCsvExporter | None = None,
    ) -> None:
        if (detector is None) != (fusion is None):
            raise ValueError("detector and fusion must be supplied together")
        self.tracker = tracker
        self.detector = detector
        self.fusion = fusion
        self.counter = counter
        self.zones_config = zones_config
        self.event_logger = event_logger
        self.model_name = model_name
        self.model_confidence = model_confidence
        self.inference_image_size = inference_image_size
        self.camera_config = camera_config
        self.diagnostics_config = diagnostics_config
        self.inference_view_configs = inference_views or resolve_inference_views(
            camera_config
        )
        self.detection_exporter = detection_exporter

    @property
    def explicit_pipeline(self) -> bool:
        return self.detector is not None

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
        view_inference = self._build_view_inference(width, height)
        pixel_views = view_inference.views if view_inference is not None else ()
        processed_frames = 0
        raw_detections_total = 0
        fused_detections_total = 0
        tracked_detections_total = 0
        frames_with_raw_detections = 0
        frames_with_fused_detections = 0
        frames_with_person_detections = 0
        maximum_person_detections_in_frame = 0
        unique_tracking_ids: set[int] = set()
        excluded_person_detections_total = 0
        trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.diagnostics_config.trajectory_length)
        )
        started = time.perf_counter()
        LOGGER.info(
            "Processing %s (%dx%d at %.2f FPS; views=%s)",
            input_path,
            width,
            height,
            source_fps,
            ",".join(view.name for view in pixel_views) or "legacy-injected",
        )

        exporter = self.detection_exporter
        if exporter is None:
            exporter = DetectionCsvExporter(
                enabled=False,
                raw_path=Path("unused.raw.csv"),
                fused_path=Path("unused.fused.csv"),
                tracks_path=Path("unused.tracks.csv"),
            )

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number = processed_frames
                self.counter.remove_stale_tracks(frame_number)

                if view_inference is None:
                    raw, included, fused, people, excluded = self._legacy_frame(
                        frame, calibration
                    )
                else:
                    raw = view_inference.collect(frame)
                    classified = [
                        (detection, calibration.is_excluded(detection))
                        for detection in raw
                    ]
                    excluded = [
                        detection
                        for detection, is_excluded in classified
                        if is_excluded
                    ]
                    included = [
                        detection
                        for detection, is_excluded in classified
                        if not is_excluded
                    ]
                    fused = self.fusion.fuse(included) if self.fusion else []
                    # This is the only explicit tracker call in a decoded-frame loop.
                    tracked = self.tracker.update(fused, frame)
                    people = []
                    for person in tracked:
                        clipped_box = clip_box_to_frame(
                            person.bounding_box,
                            width,
                            height,
                        )
                        if not box_has_positive_area(clipped_box):
                            LOGGER.warning(
                                "Dropping zero-area track %d after source clipping.",
                                person.tracking_id,
                            )
                            continue
                        people.append(
                            person.with_box(
                                clipped_box,
                                tracker_box_clipped=(
                                    clipped_box != person.bounding_box
                                ),
                            )
                        )

                raw_detections_total += len(raw)
                fused_detections_total += len(fused)
                tracked_detections_total += len(people)
                excluded_person_detections_total += len(excluded)
                frames_with_raw_detections += bool(raw)
                frames_with_fused_detections += bool(fused)
                frames_with_person_detections += bool(people)
                maximum_person_detections_in_frame = max(
                    maximum_person_detections_in_frame, len(people)
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
                    trajectories[person.tracking_id].append(
                        tuple(int(value) for value in person.center)
                    )
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

                exporter.record_raw(
                    frame_number,
                    frame_number / source_fps,
                    [
                        (detection, calibration.is_excluded(detection))
                        for detection in raw
                    ],
                )
                exporter.record_fused(frame_number, fused)
                exporter.record_tracks(
                    frame_number,
                    people,
                    {
                        tracking_id: membership.value
                        for tracking_id, membership in memberships.items()
                    },
                    {
                        tracking_id: lane.value if lane is not None else None
                        for tracking_id, lane in lanes.items()
                    },
                    {
                        person.tracking_id: (
                            state.value
                            if (state := self.counter.track_state(person.tracking_id))
                            is not None
                            else None
                        )
                        for person in people
                    },
                )

                annotated = self._annotate(
                    frame,
                    zones,
                    people,
                    memberships,
                    calibration,
                    lanes,
                    excluded,
                    raw,
                    fused,
                    pixel_views,
                    trajectories,
                    doorway_diagnostics,
                )
                writer.write(annotated)
                processed_frames += 1

                if show:
                    try:
                        cv2.imshow("TrackBus v0.2.0 - press q to stop", annotated)
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
        measured_tracker_updates = int(
            getattr(self.tracker, "update_count", processed_frames)
        )
        if self.explicit_pipeline and measured_tracker_updates != processed_frames:
            raise VideoProcessingError(
                "Tracker update invariant failed: "
                f"{measured_tracker_updates} updates for {processed_frames} frames."
            )
        diagnostic_summary = doorway_diagnostics.summary()
        fusion_method = getattr(self.fusion, "method", "legacy_combined")
        fusion_iou = getattr(self.fusion, "iou_threshold", None)
        enabled_view_names = tuple(view.name for view in pixel_views)
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
            device_used=getattr(self.tracker, "device_label", "unknown"),
            tracker_backend=type(self.tracker).__name__,
            tracker_config_path=getattr(self.tracker, "tracker_config", None),
            tracker_effective_config=dict(
                getattr(self.tracker, "effective_config", {})
            ),
            ultralytics_version=getattr(self.tracker, "ultralytics_version", None),
            # Preserve the v0.1 summary field's tracked-observation semantics.
            person_detections_total=tracked_detections_total,
            raw_person_detections_total=raw_detections_total,
            fused_detections_total=fused_detections_total,
            tracked_person_observations_total=tracked_detections_total,
            frames_with_person_detections=frames_with_person_detections,
            frames_without_person_detections=(
                processed_frames - frames_with_person_detections
            ),
            frames_with_raw_detections=frames_with_raw_detections,
            frames_without_raw_detections=(
                processed_frames - frames_with_raw_detections
            ),
            frames_with_fused_detections=frames_with_fused_detections,
            frames_without_fused_detections=(
                processed_frames - frames_with_fused_detections
            ),
            average_person_detections_per_frame=(
                round(tracked_detections_total / processed_frames, 3)
                if processed_frames
                else 0.0
            ),
            maximum_person_detections_in_frame=maximum_person_detections_in_frame,
            unique_tracking_ids=len(unique_tracking_ids),
            tracker_updates=measured_tracker_updates,
            enabled_inference_views=enabled_view_names,
            detection_fusion_method=fusion_method,
            detection_fusion_iou_threshold=fusion_iou,
            detection_roi_enabled=calibration.roi.enabled,
            detection_roi_normalized=self.camera_config.detection_roi,
            excluded_person_detections_total=excluded_person_detections_total,
            doorway_diagnostics=diagnostic_summary,
        )
        self.event_logger.write_summary(summary.to_dict())
        return summary

    def _build_view_inference(
        self, width: int, height: int
    ) -> MultiViewInference | None:
        if self.detector is None:
            return None
        return MultiViewInference.from_configs(
            self.detector,
            self.inference_view_configs,
            width,
            height,
        )

    def _legacy_frame(
        self, frame: NDArray[np.uint8], calibration: FrameCalibration
    ) -> tuple[
        list[Detection],
        list[Detection],
        list[Detection],
        list[TrackedDetection],
        list[TrackedDetection],
    ]:
        """Support explicitly injected v0.1-style fake/custom tracked backends."""

        local_people = self.tracker.update(calibration.inference_frame(frame))
        source_people = calibration.to_source(local_people)
        excluded = [
            person for person in source_people if calibration.is_excluded(person)
        ]
        people = [
            person for person in source_people if not calibration.is_excluded(person)
        ]
        raw = [
            _detection_from_track(person, "legacy_combined") for person in source_people
        ]
        included = [
            _detection_from_track(person, "legacy_combined") for person in people
        ]
        return raw, included, included.copy(), people, excluded

    def _annotate(
        self,
        frame: NDArray[np.uint8],
        zones: PixelZones,
        people: list[TrackedDetection],
        memberships: dict[int, ZoneMembership],
        calibration: FrameCalibration,
        lanes: dict[int, DoorwayLane | None],
        excluded: list[Detection] | list[TrackedDetection],
        raw: list[Detection],
        fused: list[Detection],
        views: tuple[PixelInferenceView, ...],
        trajectories: dict[int, deque[tuple[int, int]]],
        doorway_diagnostics: DoorwayDiagnostics,
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

        calibration_debug = calibration.config.debug_calibration_overlay
        visual_debug = self.diagnostics_config.debug_visualization
        if calibration_debug or visual_debug:
            _draw_calibration_overlay(annotated, calibration, views)
        if visual_debug:
            _draw_raw_detections(annotated, raw)
            _draw_fused_detections(annotated, fused)

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
            view_text = (
                f" [{'+'.join(person.source_views)}]"
                if visual_debug and person.source_views
                else ""
            )
            restart = doorway_diagnostics.tracks[person.tracking_id]
            restart_text = (
                f" restart?{restart.possible_restart_of_tracking_id}"
                if visual_debug and restart.possible_restart_of_tracking_id is not None
                else ""
            )
            label = (
                f"ID {person.tracking_id} {person.confidence:.2f} "
                f"{state_text}{lane_text}{view_text}{restart_text}"
            )
            _text_with_background(annotated, label, (left, max(18, top - 7)), color)
            anchor = tuple(int(value) for value in person.anchor)
            cv2.circle(annotated, anchor, 4, color, -1)
            if visual_debug:
                points = np.asarray(trajectories[person.tracking_id], dtype=np.int32)
                if len(points) >= 2:
                    cv2.polylines(annotated, [points], False, color, 1)

        if calibration_debug or visual_debug:
            for detection in excluded:
                left, top, right, bottom = (
                    int(value) for value in detection.bounding_box
                )
                cv2.rectangle(annotated, (left, top), (right, bottom), (40, 40, 230), 2)
                _text_with_background(
                    annotated,
                    f"{getattr(detection, 'source_view', 'track')} EXCLUDED",
                    (left, max(18, top - 7)),
                    (40, 40, 230),
                )

        occupancy = self.counter.current_occupancy
        actual_percentage = self.counter.occupancy_percentage
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
        if visual_debug:
            lines.insert(1, f"Raw: {len(raw)}  Fused: {len(fused)}")
        _draw_status_panel(annotated, lines, occupancy > self.counter.capacity)
        return annotated


def _detection_from_track(person: TrackedDetection, source_view: str) -> Detection:
    return Detection(
        bounding_box=person.bounding_box,
        confidence=person.confidence,
        class_id=person.class_id,
        source_view=source_view,
        contributing_views=person.source_views,
        metadata=person.metadata,
    )


def _draw_calibration_overlay(
    frame: NDArray[np.uint8],
    calibration: FrameCalibration,
    views: tuple[PixelInferenceView, ...] = (),
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
            frame, "LEGACY DETECTION ROI", (roi.left + 5, roi.top + 20), (255, 180, 40)
        )

    view_colors = ((255, 180, 40), (255, 100, 180), (180, 220, 60), (80, 180, 255))
    for index, view in enumerate(views):
        color = view_colors[index % len(view_colors)]
        cv2.rectangle(
            frame,
            (view.left, view.top),
            (view.right - 1, view.bottom - 1),
            color,
            1,
        )
        _text_with_background(
            frame,
            f"VIEW {view.name}",
            (view.left + 4, min(frame.shape[0] - 4, view.top + 17)),
            color,
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


def _draw_raw_detections(frame: NDArray[np.uint8], detections: list[Detection]) -> None:
    overlay = frame.copy()
    for detection in detections:
        left, top, right, bottom = (int(value) for value in detection.bounding_box)
        cv2.rectangle(overlay, (left, top), (right, bottom), (160, 160, 160), 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def _draw_fused_detections(
    frame: NDArray[np.uint8], detections: list[Detection]
) -> None:
    for detection in detections:
        left, top, right, bottom = (int(value) for value in detection.bounding_box)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 220, 40), 1)
        _text_with_background(
            frame,
            f"FUSED {detection.confidence:.2f} {'+'.join(detection.source_views)}",
            (left, max(18, bottom + 14)),
            (255, 220, 40),
        )


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
    panel_width = min(420, max(100, frame.shape[1] - 8))
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
