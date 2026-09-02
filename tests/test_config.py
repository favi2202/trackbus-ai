from pathlib import Path

import pytest

from trackbus.config import ConfigError, load_config, resolve_inference_views
from trackbus.main import _load_runtime_config, _validated_artifact_paths, build_parser
from trackbus.video_processor import VideoProcessingError

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
    assert config.model.detector_floor is None
    assert config.model.effective_detector_floor == 0.4
    assert config.model.imgsz == 512
    assert config.zones.outside[2] == (1.0, 0.4)
    assert config.tracking.effective_minimum_origin_zone_frames == 2
    assert config.tracking.effective_minimum_destination_zone_frames == 2
    assert config.tracking.maximum_transition_gap_frames == 15
    assert config.tracking.event_cooldown_frames == 30
    assert config.tracking.zone_anchor == "bottom_center"
    assert config.tracking.zone_boundary_hysteresis == 0.0
    assert config.diagnostics.failure_mining.enabled is False
    assert config.diagnostics.failure_mining.capture_frames is False


def test_event_stability_configuration_loads(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace(
        "  stale_track_timeout: 30",
        """  minimum_origin_zone_frames: 4
  minimum_destination_zone_frames: 5
  maximum_transition_gap_frames: 12
  event_cooldown_frames: 40
  zone_anchor: top_center
  zone_boundary_hysteresis: 0.015
  stale_track_timeout: 30""",
    )

    config = load_config(write_config(tmp_path, content))

    assert config.tracking.effective_minimum_origin_zone_frames == 4
    assert config.tracking.effective_minimum_destination_zone_frames == 5
    assert config.tracking.maximum_transition_gap_frames == 12
    assert config.tracking.event_cooldown_frames == 40
    assert config.tracking.zone_anchor == "top_center"
    assert config.tracking.zone_boundary_hysteresis == 0.015


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("minimum_origin_zone_frames: 0", "minimum_origin_zone_frames"),
        ("minimum_destination_zone_frames: 0", "minimum_destination_zone_frames"),
        ("maximum_transition_gap_frames: -1", "maximum_transition_gap_frames"),
        ("event_cooldown_frames: -1", "event_cooldown_frames"),
        ("zone_anchor: feet", "zone_anchor"),
        ("zone_boundary_hysteresis: 0.11", "zone_boundary_hysteresis"),
    ],
)
def test_invalid_event_stability_configuration_is_rejected(
    tmp_path: Path, setting: str, message: str
) -> None:
    content = VALID_CONFIG.replace(
        "  stale_track_timeout: 30",
        f"  {setting}\n  stale_track_timeout: 30",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, content))


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
            "--detector-floor",
            "0.1",
            "--imgsz",
            "960",
        ]
    )

    config = _load_runtime_config(args)

    assert config.model.path == "yolo11s.pt"
    assert config.model.confidence == 0.2
    assert config.model.detector_floor == 0.1
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


def test_detector_floor_is_separate_and_bounded_by_confidence(tmp_path: Path) -> None:
    configured = VALID_CONFIG.replace(
        "  confidence: 0.4", "  confidence: 0.4\n  detector_floor: 0.1"
    )

    config = load_config(write_config(tmp_path, configured))

    assert config.model.confidence == 0.4
    assert config.model.effective_detector_floor == 0.1

    with pytest.raises(ConfigError, match="cannot exceed"):
        load_config(
            write_config(
                tmp_path,
                configured.replace("detector_floor: 0.1", "detector_floor: 0.5"),
            )
        )


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


def test_failure_mining_configuration_is_explicit_and_bounded(tmp_path: Path) -> None:
    content = (
        VALID_CONFIG
        + """
diagnostics:
  failure_mining:
    enabled: true
    output_jsonl: output/failures.jsonl
    capture_frames: true
    frames_directory: output/failure-frames
    maximum_records: 25
    maximum_captured_frames: 3
    jpeg_quality: 80
    low_confidence_threshold: 0.3
    short_track_maximum_frames: 4
    sudden_detection_drop_ratio: 0.4
"""
    )

    config = load_config(write_config(tmp_path, content))
    mining = config.diagnostics.failure_mining

    assert mining.enabled is True
    assert mining.capture_frames is True
    assert mining.maximum_records == 25
    assert mining.maximum_captured_frames == 3
    assert mining.output_jsonl == Path("output/failures.jsonl")


def test_failure_frame_capture_cannot_be_enabled_implicitly(tmp_path: Path) -> None:
    content = (
        VALID_CONFIG
        + """
diagnostics:
  failure_mining:
    capture_frames: true
"""
    )

    with pytest.raises(ConfigError, match="requires failure mining"):
        load_config(write_config(tmp_path, content))


def test_artifact_paths_cannot_alias_input_or_each_other(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"

    with pytest.raises(VideoProcessingError, match="input video"):
        _validated_artifact_paths(source, video=source)
    with pytest.raises(VideoProcessingError, match="pairwise distinct"):
        _validated_artifact_paths(
            source,
            video=tmp_path / "result.mp4",
            events_csv=tmp_path / "same.csv",
            summary_json=tmp_path / "same.csv",
        )


def test_zero_fusion_iou_is_rejected(tmp_path: Path) -> None:
    content = VALID_CONFIG + "\ndetection_fusion:\n  iou_threshold: 0\n"

    with pytest.raises(ConfigError, match="greater than 0"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("content", "section", "unknown_key"),
    [
        (VALID_CONFIG + "\ncapcity: 40\n", "configuration", "capcity"),
        (
            VALID_CONFIG.replace("  imgsz: 512", "  imgsz: 512\n  image_size: 512"),
            "model",
            "image_size",
        ),
        (
            VALID_CONFIG.replace(
                "  stale_track_timeout: 30",
                "  stale_track_timeout: 30\n  stale_timeout: 30",
            ),
            "tracking",
            "stale_timeout",
        ),
        (
            VALID_CONFIG.replace(
                "  inside: [[0, 0.6]",
                "  doorway: [[0, 0.5], [1, 0.5], [1, 0.6], [0, 0.6]]\n"
                "  inside: [[0, 0.6]",
            ),
            "zones",
            "doorway",
        ),
        (
            VALID_CONFIG.replace(
                "  video: result.mp4", "  video: result.mp4\n  output_video: other.mp4"
            ),
            "outputs",
            "output_video",
        ),
        (
            VALID_CONFIG + "\ncamera:\n  roi: [0, 0, 1, 1]\n",
            "camera",
            "roi",
        ),
        (
            VALID_CONFIG + "\ndiagnostics:\n  overlap_iou: 0.5\n",
            "diagnostics",
            "overlap_iou",
        ),
        (
            VALID_CONFIG + "\ndetection_fusion:\n  threshold: 0.5\n",
            "detection_fusion",
            "threshold",
        ),
        (
            VALID_CONFIG + "\ncamera:\n  doorway_lanes:\n    left: null\n",
            "camera.doorway_lanes",
            "left",
        ),
        (
            VALID_CONFIG + "\ncamera:\n  inference_views:\n"
            "    - {name: full, bounds: [0, 0, 1, 1], active: true}\n",
            "camera.inference_views[0]",
            "active",
        ),
    ],
)
def test_unknown_configuration_keys_are_rejected_at_every_schema_level(
    tmp_path: Path, content: str, section: str, unknown_key: str
) -> None:
    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, content))

    assert str(error.value) == f"'{section}' contains unknown key(s): '{unknown_key}'."


def test_omitted_inference_views_resolves_to_default_full_frame(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    views = resolve_inference_views(config.camera)

    assert [(view.name, view.bounds, view.enabled) for view in views] == [
        ("full", (0.0, 0.0, 1.0, 1.0), True)
    ]


@pytest.mark.parametrize(
    ("yaml_value", "message"),
    [("null", "cannot be null"), ("[]", "cannot be empty")],
)
def test_explicit_null_or_empty_inference_views_is_not_treated_as_omitted(
    tmp_path: Path, yaml_value: str, message: str
) -> None:
    content = VALID_CONFIG + f"\ncamera:\n  inference_views: {yaml_value}\n"

    with pytest.raises(ConfigError, match=message) as error:
        load_config(write_config(tmp_path, content))

    assert "omit it for the default full-frame view" in str(error.value)


@pytest.mark.parametrize(
    ("old", "new", "field", "message"),
    [
        ("confidence: 0.4", "confidence: true", "model.confidence", "boolean"),
        ("confidence: 0.4", "confidence: .nan", "model.confidence", "finite"),
        ("confidence: 0.4", "confidence: .inf", "model.confidence", "finite"),
        ("imgsz: 512", "imgsz: 512.5", "model.imgsz", "integer"),
        (
            "minimum_zone_frames: 2",
            "minimum_zone_frames: false",
            "tracking.minimum_zone_frames",
            "boolean",
        ),
        (
            "stale_track_timeout: 30",
            "stale_track_timeout: .inf",
            "tracking.stale_track_timeout",
            "integer",
        ),
        ("capacity: 40", "capacity: 40.5", "capacity", "integer"),
    ],
)
def test_numeric_fields_reject_booleans_non_finite_values_and_fractional_integers(
    tmp_path: Path, old: str, new: str, field: str, message: str
) -> None:
    content = VALID_CONFIG.replace(old, new)

    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, content))

    assert field in str(error.value)
    assert message in str(error.value)


def test_normalized_coordinates_reject_boolean_values(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("outside: [[0, 0]", "outside: [[true, 0]")

    with pytest.raises(ConfigError) as error:
        load_config(write_config(tmp_path, content))

    assert "zones.outside[0][0]" in str(error.value)
    assert "boolean" in str(error.value)
