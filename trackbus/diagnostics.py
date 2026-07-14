"""Model-independent doorway detection and tracking diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from trackbus.calibration import (
    DoorwayLane,
    FrameCalibration,
    box_iou,
    normalized_center_distance,
)
from trackbus.config import DiagnosticsConfig
from trackbus.counter import CountEvent
from trackbus.tracker import TrackedPerson

LANES = (DoorwayLane.LEFT, DoorwayLane.CENTER, DoorwayLane.RIGHT)


@dataclass
class TrackDiagnostic:
    tracking_id: int
    first_observed_frame: int
    last_observed_frame: int
    observed_frames: int = 0
    first_lane_observed: DoorwayLane | None = None
    last_lane_observed: DoorwayLane | None = None
    lanes_visited: list[DoorwayLane] = field(default_factory=list)
    lane_observed_frames: dict[DoorwayLane, int] = field(
        default_factory=lambda: dict.fromkeys(LANES, 0)
    )
    lane_detection_gap_frames: dict[DoorwayLane, int] = field(
        default_factory=lambda: dict.fromkeys(LANES, 0)
    )
    unassigned_detection_gap_frames: int = 0
    crossing_events: list[str] = field(default_factory=list)
    crossed_with_nearby_track: bool = False
    nearby_tracking_ids_at_crossing: set[int] = field(default_factory=set)
    maximum_iou_with_another_track: float = 0.0
    overlap_frames: int = 0
    heavy_overlap_frames: int = 0
    disappeared_after_heavy_overlap: bool = False
    possible_restart_of_tracking_id: int | None = None
    restart_gap_frames: int | None = None
    restart_distance_normalized: float | None = None
    unusually_wide_box_frames: int = 0
    multi_lane_box_frames: int = 0
    maximum_lanes_overlapped: int = 0
    source_edge_touch_frames: int = 0
    roi_edge_touch_frames: int = 0
    any_edge_touch_frames: int = 0
    total_center_displacement_pixels: float = 0.0
    stationary_frames: int = 0
    longest_stationary_run_frames: int = 0
    current_stationary_run_frames: int = 0
    first_center: tuple[float, float] | None = None
    last_lane: DoorwayLane | None = None
    last_center: tuple[float, float] | None = None
    last_heavy_overlap: bool = False

    def to_dict(
        self, config: DiagnosticsConfig, frame_diagonal: float
    ) -> dict[str, Any]:
        movement_opportunities = max(1, self.observed_frames - 1)
        stationary_percentage = self.stationary_frames / movement_opportunities * 100
        net_displacement = (
            math.dist(self.first_center, self.last_center)
            if self.first_center is not None and self.last_center is not None
            else 0.0
        )
        likely_static = (
            self.observed_frames >= config.likely_static_minimum_frames
            and stationary_percentage >= config.likely_static_minimum_percentage * 100
        )
        return {
            "tracking_id": self.tracking_id,
            "first_observed_frame": self.first_observed_frame,
            "last_observed_frame": self.last_observed_frame,
            "observed_frames": self.observed_frames,
            "first_lane_observed": (
                self.first_lane_observed.value if self.first_lane_observed else None
            ),
            "last_lane_observed": (
                self.last_lane_observed.value if self.last_lane_observed else None
            ),
            "lanes_visited": [lane.value for lane in self.lanes_visited],
            **{
                f"{lane.value}_observed_frames": self.lane_observed_frames[lane]
                for lane in LANES
            },
            **{
                f"{lane.value}_detection_gap_frames": (
                    self.lane_detection_gap_frames[lane]
                )
                for lane in LANES
            },
            "unassigned_detection_gap_frames": self.unassigned_detection_gap_frames,
            "crossing_events": self.crossing_events,
            "crossed_with_nearby_track": self.crossed_with_nearby_track,
            "nearby_tracking_ids_at_crossing": sorted(
                self.nearby_tracking_ids_at_crossing
            ),
            "maximum_iou_with_another_track": round(
                self.maximum_iou_with_another_track, 4
            ),
            "overlap_frames": self.overlap_frames,
            "heavy_overlap_frames": self.heavy_overlap_frames,
            "disappeared_after_heavy_overlap": self.disappeared_after_heavy_overlap,
            "possible_restart_of_tracking_id": self.possible_restart_of_tracking_id,
            "restart_gap_frames": self.restart_gap_frames,
            "restart_distance_normalized": (
                round(self.restart_distance_normalized, 4)
                if self.restart_distance_normalized is not None
                else None
            ),
            "unusually_wide_box_frames": self.unusually_wide_box_frames,
            "multi_lane_box_frames": self.multi_lane_box_frames,
            "maximum_lanes_overlapped": self.maximum_lanes_overlapped,
            "possible_merged_detection": (
                self.unusually_wide_box_frames > 0 or self.multi_lane_box_frames > 0
            ),
            "source_edge_touch_frames": self.source_edge_touch_frames,
            "source_edge_touch_percentage": round(
                self.source_edge_touch_frames / self.observed_frames * 100, 2
            ),
            "roi_edge_touch_frames": self.roi_edge_touch_frames,
            "roi_edge_touch_percentage": round(
                self.roi_edge_touch_frames / self.observed_frames * 100, 2
            ),
            "any_edge_touch_frames": self.any_edge_touch_frames,
            "any_edge_touch_percentage": round(
                self.any_edge_touch_frames / self.observed_frames * 100, 2
            ),
            "total_center_displacement_pixels": round(
                self.total_center_displacement_pixels, 3
            ),
            "total_center_displacement_normalized": round(
                self.total_center_displacement_pixels / frame_diagonal, 4
            ),
            "net_center_displacement_pixels": round(net_displacement, 3),
            "stationary_frames": self.stationary_frames,
            "stationary_percentage": round(stationary_percentage, 2),
            "longest_stationary_run_frames": self.longest_stationary_run_frames,
            "likely_static": likely_static,
        }


@dataclass(frozen=True)
class DisappearedTrack:
    frame: int
    center: tuple[float, float]
    lane: DoorwayLane | None


class DoorwayDiagnostics:
    """Accumulate diagnostic signals without changing tracking or counting."""

    def __init__(
        self, calibration: FrameCalibration, config: DiagnosticsConfig
    ) -> None:
        self.calibration = calibration
        self.config = config
        self.frame_diagonal = math.hypot(
            calibration.frame_width, calibration.frame_height
        )
        self.tracks: dict[int, TrackDiagnostic] = {}
        self.maximum_people_in_doorway = 0
        self.frames_with_multiple_people_in_doorway = 0
        self.maximum_doorway_pairwise_iou = 0.0
        self.heavy_overlap_frame_count = 0
        self._previous_people: dict[int, TrackedPerson] = {}
        self._current_people: dict[int, TrackedPerson] = {}
        self._current_lanes: dict[int, DoorwayLane | None] = {}
        self._recently_disappeared: dict[int, DisappearedTrack] = {}

    def observe_frame(
        self,
        frame_number: int,
        people: list[TrackedPerson],
        lanes: dict[int, DoorwayLane | None],
        box_lanes: dict[int, tuple[DoorwayLane, ...]],
    ) -> None:
        current = {person.tracking_id: person for person in people}
        missing_ids = self._previous_people.keys() - current.keys()
        for tracking_id in missing_ids:
            memory = self.tracks[tracking_id]
            if memory.last_heavy_overlap:
                memory.disappeared_after_heavy_overlap = True
            previous = self._previous_people[tracking_id]
            self._recently_disappeared[tracking_id] = DisappearedTrack(
                frame=frame_number - 1,
                center=previous.center,
                lane=memory.last_lane,
            )

        for person in people:
            lane = lanes[person.tracking_id]
            memory = self.tracks.get(person.tracking_id)
            gap = 0
            if memory is None:
                memory = TrackDiagnostic(
                    tracking_id=person.tracking_id,
                    first_observed_frame=frame_number,
                    last_observed_frame=frame_number,
                )
                self.tracks[person.tracking_id] = memory
                self._match_possible_restart(memory, person, lane, frame_number)
            else:
                gap = frame_number - memory.last_observed_frame - 1
                if gap > 0:
                    memory.current_stationary_run_frames = 0
                    if lane is not None and lane is memory.last_lane:
                        memory.lane_detection_gap_frames[lane] += gap
                    else:
                        memory.unassigned_detection_gap_frames += gap
            self._observe_person(
                memory,
                person,
                lane,
                box_lanes[person.tracking_id],
                consecutive=memory.observed_frames > 0 and gap == 0,
            )
            memory.last_observed_frame = frame_number
            memory.last_heavy_overlap = False
            self._recently_disappeared.pop(person.tracking_id, None)

        doorway_people = [
            person for person in people if lanes[person.tracking_id] is not None
        ]
        doorway_count = len(doorway_people)
        self.maximum_people_in_doorway = max(
            self.maximum_people_in_doorway, doorway_count
        )
        if doorway_count >= 2:
            self.frames_with_multiple_people_in_doorway += 1
        if self._observe_overlaps(doorway_people):
            self.heavy_overlap_frame_count += 1

        self._current_people = current
        self._current_lanes = lanes.copy()
        self._previous_people = current
        self._prune_disappeared(frame_number)

    def record_crossing(self, event: CountEvent) -> None:
        memory = self.tracks.get(event.tracking_id)
        person = self._current_people.get(event.tracking_id)
        if memory is None or person is None:
            return
        memory.crossing_events.append(event.event_type.value)
        if self._current_lanes.get(event.tracking_id) is None:
            return
        for other_id, other in self._current_people.items():
            if (
                other_id == event.tracking_id
                or self._current_lanes.get(other_id) is None
            ):
                continue
            if (
                box_iou(person, other) > 0
                or normalized_center_distance(person, other, self.frame_diagonal)
                <= self.config.nearby_distance_normalized
            ):
                memory.crossed_with_nearby_track = True
                memory.nearby_tracking_ids_at_crossing.add(other_id)

    def summary(self) -> dict[str, Any]:
        track_dicts = [
            self.tracks[tracking_id].to_dict(self.config, self.frame_diagonal)
            for tracking_id in sorted(self.tracks)
        ]
        return {
            "maximum_people_in_doorway": self.maximum_people_in_doorway,
            "frames_with_multiple_people_in_doorway": (
                self.frames_with_multiple_people_in_doorway
            ),
            "maximum_doorway_pairwise_iou": round(self.maximum_doorway_pairwise_iou, 4),
            "heavy_overlap_frame_count": self.heavy_overlap_frame_count,
            "tracks_disappearing_after_heavy_overlap": sum(
                track.disappeared_after_heavy_overlap for track in self.tracks.values()
            ),
            "possible_id_restart_count": sum(
                track.possible_restart_of_tracking_id is not None
                for track in self.tracks.values()
            ),
            "tracks_touching_source_edge": sum(
                track.source_edge_touch_frames > 0 for track in self.tracks.values()
            ),
            "tracks_touching_roi_edge": sum(
                track.roi_edge_touch_frames > 0 for track in self.tracks.values()
            ),
            "likely_static_track_count": sum(
                track["likely_static"] for track in track_dicts
            ),
            "track_diagnostics": track_dicts,
        }

    def _observe_person(
        self,
        memory: TrackDiagnostic,
        person: TrackedPerson,
        lane: DoorwayLane | None,
        overlapping_lanes: tuple[DoorwayLane, ...],
        *,
        consecutive: bool,
    ) -> None:
        memory.observed_frames += 1
        if lane is not None:
            if memory.first_lane_observed is None:
                memory.first_lane_observed = lane
            memory.last_lane_observed = lane
            if lane not in memory.lanes_visited:
                memory.lanes_visited.append(lane)
            memory.lane_observed_frames[lane] += 1
        memory.last_lane = lane

        if memory.first_center is None:
            memory.first_center = person.center
        if memory.last_center is not None:
            displacement = math.dist(memory.last_center, person.center)
            memory.total_center_displacement_pixels += displacement
            if (
                consecutive
                and displacement / self.frame_diagonal
                <= self.config.stationary_step_threshold
            ):
                memory.stationary_frames += 1
                memory.current_stationary_run_frames += 1
                memory.longest_stationary_run_frames = max(
                    memory.longest_stationary_run_frames,
                    memory.current_stationary_run_frames,
                )
            else:
                memory.current_stationary_run_frames = 0
        memory.last_center = person.center

        touches_source = self.calibration.touches_source_edge(
            person, self.config.edge_margin_pixels
        )
        touches_roi = self.calibration.touches_roi_edge(
            person, self.config.edge_margin_pixels
        )
        memory.source_edge_touch_frames += touches_source
        memory.roi_edge_touch_frames += touches_roi
        memory.any_edge_touch_frames += touches_source or touches_roi

        left, top, right, bottom = person.bounding_box
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        aspect_ratio = width / height if height else float("inf")
        if (
            aspect_ratio >= self.config.wide_box_aspect_ratio
            or width / self.calibration.doorway_width
            >= self.config.wide_box_doorway_ratio
        ):
            memory.unusually_wide_box_frames += 1
        lane_count = len(overlapping_lanes)
        if lane_count >= 2:
            memory.multi_lane_box_frames += 1
        memory.maximum_lanes_overlapped = max(
            memory.maximum_lanes_overlapped, lane_count
        )

    def _observe_overlaps(self, people: list[TrackedPerson]) -> bool:
        heavy_overlap_in_frame = False
        overlapping_ids: set[int] = set()
        heavy_overlap_ids: set[int] = set()
        for index, first in enumerate(people):
            for second in people[index + 1 :]:
                iou = box_iou(first, second)
                self.maximum_doorway_pairwise_iou = max(
                    self.maximum_doorway_pairwise_iou, iou
                )
                first_memory = self.tracks[first.tracking_id]
                second_memory = self.tracks[second.tracking_id]
                first_memory.maximum_iou_with_another_track = max(
                    first_memory.maximum_iou_with_another_track, iou
                )
                second_memory.maximum_iou_with_another_track = max(
                    second_memory.maximum_iou_with_another_track, iou
                )
                if iou > 0:
                    overlapping_ids.update((first.tracking_id, second.tracking_id))
                if iou >= self.config.heavy_overlap_iou:
                    heavy_overlap_in_frame = True
                    heavy_overlap_ids.update((first.tracking_id, second.tracking_id))
        for tracking_id in overlapping_ids:
            self.tracks[tracking_id].overlap_frames += 1
        for tracking_id in heavy_overlap_ids:
            memory = self.tracks[tracking_id]
            memory.heavy_overlap_frames += 1
            memory.last_heavy_overlap = True
        return heavy_overlap_in_frame

    def _match_possible_restart(
        self,
        memory: TrackDiagnostic,
        person: TrackedPerson,
        lane: DoorwayLane | None,
        frame_number: int,
    ) -> None:
        if lane is None:
            return
        candidates: list[tuple[float, int, DisappearedTrack]] = []
        for tracking_id, disappeared in self._recently_disappeared.items():
            if frame_number - disappeared.frame > self.config.id_restart_window_frames:
                continue
            if disappeared.lane is not lane:
                continue
            distance = (
                math.dist(person.center, disappeared.center) / self.frame_diagonal
            )
            if distance <= self.config.id_restart_distance_normalized:
                candidates.append((distance, tracking_id, disappeared))
        if not candidates:
            return
        distance, tracking_id, disappeared = min(candidates)
        memory.possible_restart_of_tracking_id = tracking_id
        memory.restart_gap_frames = frame_number - disappeared.frame - 1
        memory.restart_distance_normalized = distance

    def _prune_disappeared(self, frame_number: int) -> None:
        expired = [
            tracking_id
            for tracking_id, disappeared in self._recently_disappeared.items()
            if frame_number - disappeared.frame > self.config.id_restart_window_frames
        ]
        for tracking_id in expired:
            del self._recently_disappeared[tracking_id]
