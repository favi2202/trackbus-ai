from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from trackbus.calibrate_camera import (
    AnchorPreviewMode,
    CalibrationLayer,
    CalibrationSession,
    CalibrationState,
    CameraCalibrationError,
    apply_calibration_to_config,
    build_parser,
    calibration_state_from_config,
    draw_calibration_overlay,
    load_calibration_config,
    main,
    save_calibration_config,
    validate_calibration,
    validate_polygon,
)

INSIDE = ((0.1, 0.05), (0.9, 0.05), (0.9, 0.35), (0.1, 0.35))
OUTSIDE = ((0.1, 0.65), (0.9, 0.65), (0.9, 0.95), (0.1, 0.95))
CORRIDOR = ((0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8))
EXCLUSION = ((0.01, 0.4), (0.08, 0.4), (0.08, 0.6), (0.01, 0.6))


def valid_state(**overrides: object) -> CalibrationState:
    values = {
        "inside": INSIDE,
        "outside": OUTSIDE,
        "crossing_corridor": CORRIDOR,
    }
    values.update(overrides)
    return CalibrationState(**values)  # type: ignore[arg-type]


def test_config_import_extracts_only_calibration_fields() -> None:
    document = {
        "model": {"path": "private-model.pt", "confidence": 0.2},
        "zones": {"inside": [list(point) for point in INSIDE], "outside": OUTSIDE},
        "camera": {
            "inference_views": [{"name": "full", "bounds": [0, 0, 1, 1]}],
            "crossing_corridor": CORRIDOR,
            "exclusion_polygons": [EXCLUSION],
        },
        "tracking": {"zone_anchor": "top_center"},
    }

    state = calibration_state_from_config(document)

    assert state.inside == INSIDE
    assert state.outside == OUTSIDE
    assert state.crossing_corridor == CORRIDOR
    assert state.exclusion_polygons == (EXCLUSION,)
    assert state.anchor_mode is AnchorPreviewMode.TOP_CENTER


def test_config_import_defaults_to_backward_compatible_bottom_center_anchor() -> None:
    state = calibration_state_from_config(
        {"zones": {"inside": INSIDE, "outside": OUTSIDE}}
    )

    assert state.anchor_mode is AnchorPreviewMode.BOTTOM_CENTER


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        (((0.0, 0.0), (1.0, 1.0)), "at least three"),
        (((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)), "normalized"),
        (((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)), "non-zero"),
    ],
)
def test_polygon_validation_rejects_invalid_geometry(polygon, message) -> None:
    assert any(message in issue for issue in validate_polygon(polygon, "test"))


def test_complete_validation_accepts_separated_zones_and_path_corridor() -> None:
    assert validate_calibration(valid_state()) == ()


def test_complete_validation_rejects_substantial_zone_overlap() -> None:
    overlapping = ((0.1, 0.2), (0.9, 0.2), (0.9, 0.7), (0.1, 0.7))

    issues = validate_calibration(valid_state(outside=overlapping))

    assert any("substantially overlap" in issue for issue in issues)


def test_complete_validation_requires_usable_neutral_transition() -> None:
    inside = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5))
    outside = ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0))

    issues = validate_calibration(CalibrationState(inside=inside, outside=outside))

    assert any("usable neutral" in issue for issue in issues)


@pytest.mark.parametrize(
    ("corridor", "message"),
    [
        (
            ((0.3, 0.05), (0.7, 0.05), (0.7, 0.3), (0.3, 0.3)),
            "intersect both",
        ),
        (
            (
                (0.3, 0.05),
                (0.7, 0.05),
                (0.7, 0.35),
                (0.3, 0.35),
                (0.3, 0.65),
                (0.7, 0.65),
                (0.7, 0.95),
                (0.3, 0.95),
            ),
            "neutral area",
        ),
    ],
)
def test_corridor_must_intersect_path_zones_and_include_neutral(
    corridor, message
) -> None:
    issues = validate_calibration(valid_state(crossing_corridor=corridor))

    assert any(message in issue for issue in issues)


@pytest.mark.parametrize(
    ("exclusion", "pathway"),
    [
        (((0.15, 0.1), (0.25, 0.1), (0.25, 0.2), (0.15, 0.2)), "inside"),
        (((0.4, 0.45), (0.5, 0.45), (0.5, 0.55), (0.4, 0.55)), "corridor"),
    ],
)
def test_exclusions_cannot_cover_calibrated_passenger_paths(exclusion, pathway) -> None:
    issues = validate_calibration(valid_state(exclusion_polygons=(exclusion,)))

    assert any("exclusion_polygons[0]" in issue for issue in issues)
    assert any(pathway in issue for issue in issues)


def test_session_draw_finish_clear_reset_and_undo() -> None:
    original = valid_state(crossing_corridor=None)
    session = CalibrationSession(state=original, original_state=original)
    assert session.select_layer(CalibrationLayer.CORRIDOR)
    for point in CORRIDOR:
        session.add_point(point)
    assert not session.select_layer(CalibrationLayer.OUTSIDE)

    session.finish_polygon()
    assert session.state.crossing_corridor == CORRIDOR
    session.clear_active()
    assert session.state.crossing_corridor is None
    assert session.undo()
    assert session.state.crossing_corridor == CORRIDOR
    session.reset()
    assert session.state == original


def test_session_can_add_exclusions_and_move_an_existing_vertex() -> None:
    original = valid_state(crossing_corridor=None)
    session = CalibrationSession(state=original, original_state=original)
    session.select_layer(CalibrationLayer.EXCLUSION)
    for point in EXCLUSION:
        session.add_point(point)
    session.finish_polygon()

    assert session.state.exclusion_polygons == (EXCLUSION,)
    assert session.select_nearest_vertex(EXCLUSION[0], maximum_distance=0.01)
    assert session.move_selected_vertex((0.02, 0.4))
    assert session.state.exclusion_polygons[0][0] == (0.02, 0.4)
    assert session.undo()
    assert session.state.exclusion_polygons == (EXCLUSION,)


def test_session_cycles_all_anchor_preview_modes() -> None:
    session = CalibrationSession.from_config(
        {"zones": {"inside": INSIDE, "outside": OUTSIDE}}
    )

    assert session.cycle_anchor() is AnchorPreviewMode.TOP_CENTER
    assert session.cycle_anchor() is AnchorPreviewMode.CENTER
    assert session.cycle_anchor() is AnchorPreviewMode.BOTTOM_CENTER


def test_apply_preserves_every_unrelated_semantic_setting() -> None:
    document = {
        "model": {"path": "yolo11s.pt", "confidence": 0.15},
        "tracking": {"tracker": "custom.yaml", "minimum_zone_frames": 4},
        "zones": {"inside": [[0, 0], [1, 0], [1, 0.1]], "label": "keep"},
        "camera": {
            "inference_views": [{"name": "full", "bounds": [0, 0, 1, 1]}],
            "debug_calibration_overlay": True,
            "doorway_lanes": {"anchor": "center"},
        },
        "outputs": {"video": "data/output/result.mp4"},
    }
    before = copy.deepcopy(document)

    updated = apply_calibration_to_config(document, valid_state())

    assert document == before
    assert updated["model"] == document["model"]
    assert updated["tracking"] == {
        **document["tracking"],
        "zone_anchor": "bottom_center",
    }
    assert updated["outputs"] == document["outputs"]
    assert updated["zones"]["label"] == "keep"
    assert updated["camera"]["inference_views"] == document["camera"]["inference_views"]
    assert updated["camera"]["doorway_lanes"] == {"anchor": "center"}
    assert updated["tracking"]["zone_anchor"] == "bottom_center"


def test_atomic_save_round_trip_leaves_no_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "camera.yaml"
    document = {"logging_level": "DEBUG", "custom": {"preserve": [1, 2, 3]}}

    save_calibration_config(output, document, valid_state())

    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded["logging_level"] == "DEBUG"
    assert loaded["custom"] == {"preserve": [1, 2, 3]}
    assert loaded["zones"]["inside"] == [list(point) for point in INSIDE]
    assert list(tmp_path.glob(".camera.yaml.*.tmp")) == []


def test_load_config_supports_missing_file_and_rejects_non_mapping(tmp_path) -> None:
    assert load_calibration_config(tmp_path / "new.yaml") == {}
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(CameraCalibrationError, match="root"):
        load_calibration_config(invalid)


def test_overlay_draws_without_mutating_source_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    session = CalibrationSession(state=valid_state(), original_state=valid_state())

    rendered = draw_calibration_overlay(
        frame, session, frame_number=12, frame_count=100, cursor=(200, 150)
    )

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0


def test_parser_documents_all_destructive_and_exit_controls(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for phrase in ("undo", "clear", "reset", "validate and save", "cancel"):
        assert phrase in output
    assert "--frame" in output


def test_cli_refuses_to_overwrite_input_video(tmp_path, capsys) -> None:
    video = tmp_path / "door.mp4"
    original = b"not a video"
    video.write_bytes(original)

    result = main(
        ["--input", str(video), "--output-config", str(video), "--frame", "0"]
    )

    assert result == 2
    assert video.read_bytes() == original
    assert "must differ" in capsys.readouterr().err
