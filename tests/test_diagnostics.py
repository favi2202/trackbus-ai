from trackbus.calibration import FrameCalibration
from trackbus.config import CameraConfig, DiagnosticsConfig
from trackbus.counter import CountEvent, EventType
from trackbus.diagnostics import DoorwayDiagnostics
from trackbus.tracker import TrackedPerson


def person(tracking_id: int, box: tuple[float, float, float, float]) -> TrackedPerson:
    return TrackedPerson(tracking_id, box, 0.9)


def calibration() -> FrameCalibration:
    return FrameCalibration(
        CameraConfig(
            left_lane=((0.0, 0.0), (0.33, 0.0), (0.33, 1.0), (0.0, 1.0)),
            center_lane=((0.34, 0.0), (0.66, 0.0), (0.66, 1.0), (0.34, 1.0)),
            right_lane=((0.67, 0.0), (1.0, 0.0), (1.0, 1.0), (0.67, 1.0)),
        ),
        100,
        100,
    )


def observe(
    diagnostics: DoorwayDiagnostics,
    frame: int,
    people: list[TrackedPerson],
) -> None:
    camera = diagnostics.calibration
    lanes = {item.tracking_id: camera.lane_for(item) for item in people}
    box_lanes = {item.tracking_id: camera.box_lanes(item, 0.05) for item in people}
    diagnostics.observe_frame(frame, people, lanes, box_lanes)


def test_lane_frames_and_same_lane_detection_gap() -> None:
    diagnostics = DoorwayDiagnostics(calibration(), DiagnosticsConfig())
    left = person(1, (5.0, 20.0, 25.0, 60.0))
    center = person(1, (40.0, 20.0, 60.0, 60.0))

    observe(diagnostics, 0, [left])
    observe(diagnostics, 1, [])
    observe(diagnostics, 2, [left])
    observe(diagnostics, 3, [center])

    track = diagnostics.summary()["track_diagnostics"][0]
    assert track["first_lane_observed"] == "left_lane"
    assert track["last_lane_observed"] == "center_lane"
    assert track["lanes_visited"] == ["left_lane", "center_lane"]
    assert track["left_lane_observed_frames"] == 2
    assert track["center_lane_observed_frames"] == 1
    assert track["left_lane_detection_gap_frames"] == 1


def test_overlap_disappearance_restart_and_nearby_crossing() -> None:
    config = DiagnosticsConfig(
        heavy_overlap_iou=0.5,
        id_restart_distance_normalized=0.1,
    )
    diagnostics = DoorwayDiagnostics(calibration(), config)
    first = person(1, (35.0, 20.0, 65.0, 60.0))
    second = person(2, (40.0, 20.0, 70.0, 60.0))

    observe(diagnostics, 0, [first, second])
    diagnostics.record_crossing(CountEvent(0, 1, EventType.IN, 1, 0, 1, 2.5))
    observe(diagnostics, 1, [second])
    observe(diagnostics, 2, [second, person(3, (36.0, 20.0, 66.0, 60.0))])

    summary = diagnostics.summary()
    tracks = {track["tracking_id"]: track for track in summary["track_diagnostics"]}
    assert summary["maximum_people_in_doorway"] == 2
    assert summary["frames_with_multiple_people_in_doorway"] == 2
    assert summary["maximum_doorway_pairwise_iou"] > 0.5
    assert tracks[1]["disappeared_after_heavy_overlap"] is True
    assert tracks[1]["crossed_with_nearby_track"] is True
    assert tracks[3]["possible_restart_of_tracking_id"] == 1


def test_doorway_metrics_are_unavailable_without_lane_or_corridor() -> None:
    diagnostics = DoorwayDiagnostics(
        FrameCalibration(CameraConfig(), 100, 100), DiagnosticsConfig()
    )
    tracked = person(1, (40.0, 20.0, 60.0, 60.0))

    diagnostics.observe_frame(0, [tracked], {1: None}, {1: ()}, {1: False})

    summary = diagnostics.summary()
    assert summary["doorway_diagnostics_available"] is False
    assert summary["doorway_diagnostics_reason"] == (
        "no_doorway_lane_or_crossing_corridor_calibration"
    )
    assert summary["maximum_people_in_doorway"] is None
    assert summary["frames_with_multiple_people_in_doorway"] is None
    assert summary["doorway_lane_diagnostics_available"] is False
    assert summary["possible_id_restart_count"] is None


def test_crossing_corridor_enables_doorway_metrics_without_lanes() -> None:
    camera = FrameCalibration(
        CameraConfig(
            crossing_corridor=(
                (0.2, 0.0),
                (0.8, 0.0),
                (0.8, 1.0),
                (0.2, 1.0),
            )
        ),
        100,
        100,
    )
    diagnostics = DoorwayDiagnostics(camera, DiagnosticsConfig())
    tracked = person(1, (40.0, 20.0, 60.0, 60.0))

    diagnostics.observe_frame(
        0,
        [tracked],
        {1: None},
        {1: ()},
        {1: camera.corridor_contains(tracked.anchor)},
    )
    diagnostics.observe_frame(1, [], {}, {}, {})
    replacement = person(2, (41.0, 20.0, 61.0, 60.0))
    diagnostics.observe_frame(
        2,
        [replacement],
        {2: None},
        {2: ()},
        {2: camera.corridor_contains(replacement.anchor)},
    )

    summary = diagnostics.summary()
    assert summary["doorway_diagnostics_available"] is True
    assert summary["doorway_diagnostics_reason"] is None
    assert summary["maximum_people_in_doorway"] == 1
    assert summary["doorway_lane_diagnostics_available"] is False
    track = summary["track_diagnostics"][0]
    assert track["left_lane_observed_frames"] is None
    assert track["multi_lane_box_frames"] is None
    tracks = {item["tracking_id"]: item for item in summary["track_diagnostics"]}
    assert tracks[2]["possible_restart_of_tracking_id"] == 1


def test_wide_multi_lane_box_is_only_reported() -> None:
    diagnostics = DoorwayDiagnostics(calibration(), DiagnosticsConfig())

    observe(diagnostics, 0, [person(9, (10.0, 20.0, 90.0, 60.0))])

    track = diagnostics.summary()["track_diagnostics"][0]
    assert track["unusually_wide_box_frames"] == 1
    assert track["multi_lane_box_frames"] == 1
    assert track["maximum_lanes_overlapped"] == 3
    assert track["possible_merged_detection"] is True


def test_secondary_source_and_roi_edge_statistics() -> None:
    camera = CameraConfig(
        detection_roi=(0.1, 0.1, 0.9, 0.9),
        center_lane=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )
    diagnostics = DoorwayDiagnostics(
        FrameCalibration(camera, 100, 100), DiagnosticsConfig(edge_margin_pixels=1)
    )

    observe(diagnostics, 0, [person(1, (10.0, 20.0, 30.0, 50.0))])
    observe(diagnostics, 1, [person(1, (0.0, 20.0, 20.0, 50.0))])

    summary = diagnostics.summary()
    track = summary["track_diagnostics"][0]
    assert track["source_edge_touch_frames"] == 1
    assert track["source_edge_touch_percentage"] == 50.0
    assert track["roi_edge_touch_frames"] == 2
    assert track["roi_edge_touch_percentage"] == 100.0
    assert summary["tracks_touching_source_edge"] == 1
    assert summary["tracks_touching_roi_edge"] == 1


def test_likely_static_track_is_reported_but_not_removed() -> None:
    config = DiagnosticsConfig(
        stationary_step_threshold=0.01,
        likely_static_minimum_frames=3,
        likely_static_minimum_percentage=0.9,
    )
    diagnostics = DoorwayDiagnostics(calibration(), config)
    stationary = person(4, (40.0, 20.0, 60.0, 60.0))

    observe(diagnostics, 0, [stationary])
    observe(diagnostics, 1, [stationary])
    observe(diagnostics, 2, [stationary])

    summary = diagnostics.summary()
    track = summary["track_diagnostics"][0]
    assert track["observed_frames"] == 3
    assert track["stationary_frames"] == 2
    assert track["stationary_percentage"] == 100.0
    assert track["likely_static"] is True
    assert summary["likely_static_track_count"] == 1
