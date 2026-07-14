from pathlib import Path

import pytest

from trackbus.config import ConfigError, load_config
from trackbus.main import _load_runtime_config, build_parser

VALID_CONFIG = """
model:
  path: yolo11n.pt
  confidence: 0.4
  imgsz: 512
tracking:
  minimum_zone_frames: 2
  stale_track_timeout: 30
zones:
  outside: [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]
  inside: [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]
outputs:
  video: result.mp4
capacity: 40
initial_occupancy: 0
"""


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_configuration_loads(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.capacity == 40
    assert config.model.confidence == 0.4
    assert config.model.imgsz == 512
    assert config.zones.outside[2] == (1.0, 0.4)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("capacity: 40", "capacity: 0", "capacity"),
        ("confidence: 0.4", "confidence: 1.5", "confidence"),
        ("imgsz: 512", "imgsz: 16", "imgsz"),
        ("initial_occupancy: 0", "initial_occupancy: -1", "initial_occupancy"),
        ("stale_track_timeout: 30", "stale_track_timeout: 0", "stale"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = write_config(tmp_path, VALID_CONFIG.replace(old, new))

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_out_of_range_normalized_coordinate_is_rejected(tmp_path: Path) -> None:
    invalid = VALID_CONFIG.replace("[1, 0.4]", "[1.2, 0.4]")

    with pytest.raises(ConfigError, match="normalized"):
        load_config(write_config(tmp_path, invalid))


def test_missing_configuration_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_cli_model_settings_override_yaml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, VALID_CONFIG)
    args = build_parser().parse_args(
        [
            "--input",
            "input.mp4",
            "--config",
            str(config_path),
            "--model",
            "yolo11s.pt",
            "--confidence",
            "0.2",
            "--imgsz",
            "960",
        ]
    )

    config = _load_runtime_config(args)

    assert config.model.path == "yolo11s.pt"
    assert config.model.confidence == 0.2
    assert config.model.imgsz == 960


def test_omitted_cli_model_settings_preserve_yaml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, VALID_CONFIG)
    args = build_parser().parse_args(
        ["--input", "input.mp4", "--config", str(config_path)]
    )

    config = _load_runtime_config(args)

    assert config.model.path == "yolo11n.pt"
    assert config.model.confidence == 0.4
    assert config.model.imgsz == 512


def test_camera_calibration_configuration_loads(tmp_path: Path) -> None:
    content = (
        VALID_CONFIG
        + """
camera:
  detection_roi: [0.1, 0.2, 0.9, 0.8]
  doorway_lanes:
    anchor: bottom_center
    left_lane: [[0.1, 0.2], [0.3, 0.2], [0.3, 0.8], [0.1, 0.8]]
    center_lane: null
    right_lane: null
  exclusion_polygons:
    - [[0.0, 0.0], [0.05, 0.0], [0.05, 0.1], [0.0, 0.1]]
  debug_calibration_overlay: true
"""
    )

    config = load_config(write_config(tmp_path, content))

    assert config.camera.detection_roi == (0.1, 0.2, 0.9, 0.8)
    assert config.camera.lane_anchor == "bottom_center"
    assert config.camera.left_lane is not None
    assert len(config.camera.exclusion_polygons) == 1
    assert config.camera.debug_calibration_overlay is True


@pytest.mark.parametrize(
    ("camera_yaml", "message"),
    [
        ("detection_roi: [0.8, 0.2, 0.1, 0.9]", "left < right"),
        ("doorway_lanes:\n    anchor: feet", "anchor"),
        ("debug_calibration_overlay: 1", "true or false"),
    ],
)
def test_invalid_camera_calibration_is_rejected(
    tmp_path: Path, camera_yaml: str, message: str
) -> None:
    content = VALID_CONFIG + f"\ncamera:\n  {camera_yaml}\n"

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, content))


def test_invalid_diagnostic_threshold_is_rejected(tmp_path: Path) -> None:
    content = VALID_CONFIG + "\ndiagnostics:\n  heavy_overlap_iou: -0.1\n"

    with pytest.raises(ConfigError, match="heavy_overlap_iou"):
        load_config(write_config(tmp_path, content))
