import numpy as np
import pytest

from trackbus.calibration import CalibrationError, DoorwayLane, FrameCalibration
from trackbus.config import CameraConfig
from trackbus.tracker import TrackedPerson


def person(tracking_id: int, box: tuple[float, float, float, float]) -> TrackedPerson:
    return TrackedPerson(tracking_id, box, 0.9)


def lane_camera(**overrides: object) -> CameraConfig:
    values = {
        "left_lane": ((0.0, 0.2), (0.33, 0.2), (0.33, 0.9), (0.0, 0.9)),
        "center_lane": ((0.34, 0.2), (0.66, 0.2), (0.66, 0.9), (0.34, 0.9)),
        "right_lane": ((0.67, 0.2), (1.0, 0.2), (1.0, 0.9), (0.67, 0.9)),
    }
    values.update(overrides)
    return CameraConfig(**values)  # type: ignore[arg-type]


def test_roi_crop_and_coordinate_translation() -> None:
    calibration = FrameCalibration(
        CameraConfig(detection_roi=(0.25, 0.25, 0.75, 0.75)), 100, 80
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    crop = calibration.inference_frame(frame)
    translated = calibration.to_source([person(1, (0.0, 0.0, 20.0, 15.0))])[0]

    assert crop.shape == (40, 50, 3)
    assert translated.bounding_box == (25.0, 20.0, 45.0, 35.0)
    assert translated.anchor == (35.0, 35.0)


def test_full_frame_mode_preserves_frame_and_coordinates() -> None:
    calibration = FrameCalibration(CameraConfig(), 100, 80)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    original = person(1, (10.0, 20.0, 30.0, 40.0))

    assert calibration.inference_frame(frame) is frame
    assert calibration.to_source([original])[0] == original


def test_lane_anchor_defaults_to_box_center() -> None:
    calibration = FrameCalibration(lane_camera(), 100, 100)

    assert calibration.lane_for(person(1, (5.0, 20.0, 25.0, 60.0))) is DoorwayLane.LEFT
    assert (
        calibration.lane_for(person(2, (40.0, 20.0, 60.0, 60.0))) is DoorwayLane.CENTER
    )
    assert (
        calibration.lane_for(person(3, (75.0, 20.0, 95.0, 60.0))) is DoorwayLane.RIGHT
    )


def test_lane_anchor_can_use_bottom_center_without_changing_person_anchor() -> None:
    lane = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))
    tracked = person(1, (40.0, 20.0, 60.0, 80.0))
    center_calibration = FrameCalibration(CameraConfig(center_lane=lane), 100, 100)
    bottom_calibration = FrameCalibration(
        CameraConfig(center_lane=lane, lane_anchor="bottom_center"), 100, 100
    )

    assert center_calibration.lane_for(tracked) is None
    assert bottom_calibration.lane_for(tracked) is DoorwayLane.CENTER
    assert tracked.anchor == (50.0, 80.0)


def test_exclusion_uses_existing_bottom_center_anchor() -> None:
    camera = CameraConfig(
        exclusion_polygons=(((0.0, 0.7), (0.2, 0.7), (0.2, 1.0), (0.0, 1.0)),)
    )
    calibration = FrameCalibration(camera, 100, 100)

    assert calibration.is_excluded(person(1, (0.0, 40.0, 20.0, 80.0)))
    assert not calibration.is_excluded(person(2, (30.0, 40.0, 50.0, 80.0)))


def test_exclusion_cannot_overlap_passenger_lane() -> None:
    camera = lane_camera(
        exclusion_polygons=(((0.1, 0.3), (0.2, 0.3), (0.2, 0.4), (0.1, 0.4)),)
    )

    with pytest.raises(CalibrationError, match="static structures only"):
        FrameCalibration(camera, 100, 100)


def test_box_can_overlap_multiple_lanes() -> None:
    calibration = FrameCalibration(lane_camera(), 100, 100)

    lanes = calibration.box_lanes(person(1, (20.0, 30.0, 80.0, 70.0)), 0.05)

    assert lanes == (DoorwayLane.LEFT, DoorwayLane.CENTER, DoorwayLane.RIGHT)
