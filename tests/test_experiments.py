from __future__ import annotations

import csv
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from trackbus.config import load_config
from trackbus.experiments import (
    ExperimentArtifacts,
    ExperimentConfigError,
    ExperimentRun,
    build_artifacts,
    load_experiment_matrix,
    rank_results,
    resolve_run_configuration,
    run_experiments,
    subprocess_executor,
    write_leaderboard,
)


def write_matrix(
    tmp_path: Path,
    *,
    runs: list[dict[str, Any]] | None = None,
    expected_entered: int = 6,
    expected_exited: int = 2,
) -> Path:
    tracker_dir = tmp_path / "configs"
    tracker_dir.mkdir(exist_ok=True)
    (tracker_dir / "bytetrack.yaml").write_text(
        "tracker_type: bytetrack\n", encoding="utf-8"
    )
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "old.pt", "confidence": 0.5, "imgsz": 320},
                "tracking": {"tracker": "old.yaml", "stale_track_timeout": 90},
                "camera": {"detection_roi": [0.1, 0.1, 0.9, 0.9]},
                "zones": {
                    "outside": [[0, 0], [1, 0], [1, 0.4], [0, 0.4]],
                    "inside": [[0, 0.6], [1, 0.6], [1, 1], [0, 1]],
                },
                "outputs": {"video": "unused.mp4"},
            }
        ),
        encoding="utf-8",
    )
    if runs is None:
        runs = [
            {
                "name": "A baseline",
                "model": "yolo11n.pt",
                "confidence": 0.35,
                "image_size": 640,
                "tracker": {
                    "config": "configs/bytetrack.yaml",
                    "track_buffer": 30,
                },
                "enabled_views": ["full"],
                "detection_fusion": {"iou_threshold": 0.5},
            },
            {
                "name": "D multiview",
                "model": "yolo11n.pt",
                "confidence": 0.25,
                "image_size": 640,
                "tracker": {"config": "configs/bytetrack.yaml"},
                "enabled_views": ["full", "left"],
                "detection_fusion": {"iou_threshold": 0.45},
            },
        ]
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "base_config": "base.yaml",
                "aggregate_ground_truth": {
                    "entered": expected_entered,
                    "exited": expected_exited,
                },
                "evaluation": {"tolerance_frames": 20},
                "views": {
                    "full": [0, 0, 1, 1],
                    "left": {"bounds": [0, 0.15, 0.62, 1]},
                },
                "runs": runs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return matrix


def complete_summary(**overrides: Any) -> dict[str, Any]:
    summary = {
        "entered_total": 6,
        "exited_total": 2,
        "raw_person_detections_total": 250,
        "fused_detections_total": 240,
        "unique_tracking_ids": 9,
        "possible_id_restart_count": 1,
        "likely_static_track_count": 0,
        "tracks_disappearing_after_heavy_overlap": 0,
        "frames_without_raw_detections": 300,
        "average_fps": 20.0,
        "processing_time_seconds": 29.45,
    }
    summary.update(overrides)
    return summary


def test_load_matrix_preserves_explicit_run_combinations(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))

    assert matrix.expected_entered == 6
    assert matrix.expected_exited == 2
    assert matrix.tolerance_frames == 20
    assert matrix.views["left"].bounds == (0.0, 0.15, 0.62, 1.0)
    assert [run.name for run in matrix.runs] == ["A baseline", "D multiview"]
    assert matrix.runs[0].enabled_views == ("full",)
    assert matrix.runs[0].tracker["track_buffer"] == 30
    assert matrix.runs[1].inference_mode == "multiview"
    assert matrix.runs[1].detection_fusion["iou_threshold"] == 0.45


def test_matrix_rejects_remote_model_and_unknown_view(tmp_path: Path) -> None:
    remote = [
        {
            "name": "unsafe",
            "model": "https://example.invalid/untrusted.pt",
            "confidence": 0.3,
            "image_size": 640,
            "tracker": {"config": "tracker.yaml"},
            "enabled_views": ["full"],
        }
    ]
    with pytest.raises(ExperimentConfigError, match="remote URL"):
        load_experiment_matrix(write_matrix(tmp_path, runs=remote))

    unknown = [
        {
            "name": "unknown crop",
            "model": "yolo11n.pt",
            "confidence": 0.3,
            "image_size": 640,
            "tracker": {"config": "tracker.yaml"},
            "enabled_views": ["right"],
        }
    ]
    with pytest.raises(ExperimentConfigError, match="unknown views"):
        load_experiment_matrix(write_matrix(tmp_path, runs=unknown))


def test_matrix_requires_explicit_aggregate_totals_and_positive_fusion_iou(
    tmp_path: Path,
) -> None:
    matrix_path = write_matrix(tmp_path)
    raw = yaml.safe_load(matrix_path.read_text("utf-8"))
    raw.pop("aggregate_ground_truth")
    matrix_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="aggregate_ground_truth"):
        load_experiment_matrix(matrix_path)

    zero_iou = [
        {
            "name": "invalid fusion",
            "model": "yolo11n.pt",
            "confidence": 0.3,
            "image_size": 640,
            "tracker": {"config": "configs/bytetrack.yaml"},
            "enabled_views": ["full"],
            "detection_fusion": {"iou_threshold": 0.0},
        }
    ]
    with pytest.raises(ExperimentConfigError, match=r"\(0, 1\]"):
        load_experiment_matrix(write_matrix(tmp_path, runs=zero_iou))


def test_resolve_configuration_replaces_roi_and_sets_artifacts(
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    run = matrix.runs[1]
    artifacts = build_artifacts(tmp_path / "results", run)

    config = resolve_run_configuration(matrix, run, artifacts)

    assert config["model"] == {
        "path": "yolo11n.pt",
        "confidence": 0.25,
        "imgsz": 640,
    }
    assert config["camera"]["detection_roi"] is None
    assert [view["name"] for view in config["camera"]["inference_views"]] == [
        "full",
        "left",
    ]
    assert config["tracking"]["stale_track_timeout"] == 90
    assert config["tracking"]["tracker"] == str(
        (tmp_path / "configs" / "bytetrack.yaml").resolve()
    )
    assert config["outputs"]["summary_json"].endswith("summary.json")


def test_resolved_configuration_loads_through_application_config(
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    run = matrix.runs[0]
    artifacts = build_artifacts(tmp_path / "results", run)
    artifacts.run_dir.mkdir(parents=True)
    artifacts.resolved_config.write_text(
        yaml.safe_dump(resolve_run_configuration(matrix, run, artifacts)),
        encoding="utf-8",
    )

    config = load_config(artifacts.resolved_config)

    assert config.model.path == "yolo11n.pt"
    assert config.camera.detection_roi is None
    assert [view.name for view in config.camera.inference_views] == ["full"]
    assert config.detection_fusion.iou_threshold == 0.5
    assert config.outputs.raw_detections_csv == artifacts.raw_detections_csv.resolve()


def result(
    name: str,
    *,
    count_error: int,
    risk: int,
    fps: float,
    f1: float | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "completed",
        "inference_mode": "full_frame",
        "aggregate_count_error": count_error,
        "false_event_risk_score": risk,
        "processing_fps": fps,
        "event_f1": f1,
    }


def test_ranking_uses_event_f1_only_with_frame_ground_truth() -> None:
    exact_but_poor_f1 = result("exact totals", count_error=0, risk=0, fps=30.0, f1=0.25)
    good_events = result("good events", count_error=2, risk=1, fps=20.0, f1=0.9)

    event_ranked = rank_results(
        [exact_but_poor_f1, good_events], has_frame_ground_truth=True
    )
    aggregate_ranked = rank_results(
        [exact_but_poor_f1, good_events], has_frame_ground_truth=False
    )

    assert [item["name"] for item in event_ranked] == ["good events", "exact totals"]
    assert [item["name"] for item in aggregate_ranked] == [
        "exact totals",
        "good events",
    ]


def test_leaderboard_files_label_aggregate_metrics_as_insufficient(
    tmp_path: Path,
) -> None:
    first = result("first", count_error=0, risk=1, fps=12.5, f1=None)
    first.update(
        {
            "entered": 6,
            "exited": 2,
            "configuration": {"model": "yolo11n.pt"},
            "output_paths": {"summary_json": "first/summary.json"},
        }
    )

    write_leaderboard(
        [first],
        tmp_path,
        expected_entered=6,
        expected_exited=2,
        has_frame_ground_truth=False,
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text("utf-8"))
    rows = list(csv.DictReader((tmp_path / "leaderboard.csv").open(encoding="utf-8")))
    report = (tmp_path / "report.md").read_text("utf-8")
    assert payload["ground_truth"]["precision_recall_available"] is False
    assert payload["runs"][0]["name"] == "first"
    assert rows[0]["rank"] == "1"
    assert "cannot establish event precision, recall, or F1" in report


def test_runner_accepts_fake_executor_without_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = write_matrix(tmp_path)
    matrix = load_experiment_matrix(matrix_path)
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"not decoded by the fake executor")
    observed_configs: list[dict[str, Any]] = []

    class AggregateResult:
        def __init__(self, values: dict[str, Any]) -> None:
            self.values = values

        def to_dict(self) -> dict[str, Any]:
            return self.values

    def evaluate_aggregate(**values: int) -> AggregateResult:
        return AggregateResult({"evaluation_mode": "aggregate_only", **values})

    monkeypatch.setitem(
        sys.modules,
        "trackbus.evaluation",
        SimpleNamespace(evaluate_aggregate=evaluate_aggregate),
    )

    def fake_executor(
        _input: Path, config_path: Path, _artifacts: object
    ) -> dict[str, Any]:
        observed_configs.append(yaml.safe_load(config_path.read_text("utf-8")))
        return complete_summary()

    ranked = run_experiments(
        input_path=input_path,
        matrix=matrix,
        output_dir=tmp_path / "output",
        executor=fake_executor,
    )

    assert len(observed_configs) == 2
    assert len(ranked) == 2
    assert all(item["status"] == "completed" for item in ranked)
    assert ranked[0]["aggregate_count_error"] == 0
    first = ranked[0]
    assert first["configuration"]["model"]["path"] == "yolo11n.pt"
    assert (
        first["configuration_sha256"]
        == hashlib.sha256(
            Path(first["output_paths"]["resolved_config"]).read_bytes()
        ).hexdigest()
    )
    assert (
        first["reproducibility"]["input_video"]["sha256"]
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    assert (tmp_path / "output" / "leaderboard.csv").is_file()
    assert (tmp_path / "output" / "a_baseline" / "resolved_config.yaml").is_file()


def test_runner_uses_frame_annotations_for_event_ranking(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(
        write_matrix(tmp_path, expected_entered=1, expected_exited=1)
    )
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake video")
    ground_truth = tmp_path / "events.json"
    ground_truth.write_text(
        json.dumps(
            {
                "schema_version": "trackbus.events/v1",
                "video": {"filename": "video.mp4"},
                "source_fps": 25.0,
                "events": [
                    {"frame": 25, "timestamp": 1.0, "direction": "IN"},
                    {"frame": 50, "timestamp": 2.0, "direction": "OUT"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_executor(
        _input: Path, _config: Path, artifacts: ExperimentArtifacts
    ) -> dict[str, Any]:
        artifacts.events_csv.write_text(
            "timestamp,video_frame,event_type\n"
            "00:00:01.000,25,IN\n"
            "00:00:02.000,50,OUT\n",
            encoding="utf-8",
        )
        return complete_summary(
            entered_total=1,
            exited_total=1,
            source_fps=25.0,
            average_fps=10.0,
        )

    ranked = run_experiments(
        input_path=input_path,
        matrix=matrix,
        output_dir=tmp_path / "event_output",
        ground_truth_path=ground_truth,
        executor=fake_executor,
    )

    assert ranked[0]["event_true_positives"] == 2
    assert ranked[0]["event_f1"] == 1.0
    assert ranked[0]["event_in_f1"] == 1.0
    assert ranked[0]["event_out_f1"] == 1.0
    leaderboard = json.loads(
        (tmp_path / "event_output" / "leaderboard.json").read_text("utf-8")
    )
    assert leaderboard["ground_truth"]["mode"] == "frame_events"
    assert leaderboard["ground_truth"]["expected_entered"] == 1
    assert leaderboard["ground_truth"]["expected_exited"] == 1


def test_default_matrix_is_the_bounded_a_through_e_set() -> None:
    repository = Path(__file__).resolve().parents[1]
    matrix = load_experiment_matrix(
        repository / "configs" / "experiments" / "test_video.yaml"
    )

    assert len(matrix.runs) == 5
    assert [run.name[0] for run in matrix.runs] == ["A", "B", "C", "D", "E"]
    assert [run.model for run in matrix.runs] == [
        "yolo11n.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]
    assert [run.enabled_views for run in matrix.runs[-2:]] == [
        ("full", "left_doorway", "right_doorway"),
        ("full", "left_doorway", "right_doorway"),
    ]


def test_experiment_run_slug_is_stable() -> None:
    run = ExperimentRun(
        name="D: Multi-view Nano",
        description="",
        model="yolo11n.pt",
        confidence=0.25,
        image_size=640,
        tracker={"config": "tracker.yaml"},
        enabled_views=("full", "left"),
        detection_fusion={"method": "nms", "iou_threshold": 0.5},
        overrides={},
    )

    assert run.slug == "d_multi_view_nano"


def test_matrix_rejects_unknown_keys_and_protected_overrides(tmp_path: Path) -> None:
    matrix_path = write_matrix(tmp_path)
    raw = yaml.safe_load(matrix_path.read_text("utf-8"))
    raw["typo"] = True
    matrix_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="matrix.*unknown key.*typo"):
        load_experiment_matrix(matrix_path)

    protected = [
        {
            "name": "override model",
            "model": "yolo11n.pt",
            "confidence": 0.3,
            "image_size": 640,
            "tracker": {"config": "configs/bytetrack.yaml"},
            "enabled_views": ["full"],
            "overrides": {"model": {"path": "different.pt"}},
        }
    ]
    with pytest.raises(ExperimentConfigError, match="matrix-controlled.*model.path"):
        load_experiment_matrix(write_matrix(tmp_path, runs=protected))

    unknown_override = deepcopy(protected)
    unknown_override[0]["overrides"] = {"diagnostics": {"typo": 1}}
    with pytest.raises(ExperimentConfigError, match="unknown key.*typo"):
        load_experiment_matrix(write_matrix(tmp_path, runs=unknown_override))


def test_resolve_rejects_unknown_base_config_key(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    base = yaml.safe_load(matrix.base_config_path.read_text("utf-8"))
    base["model"]["confidence_typo"] = 0.2
    matrix.base_config_path.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="confidence_typo"):
        resolve_run_configuration(
            matrix,
            matrix.runs[0],
            build_artifacts(tmp_path / "results", matrix.runs[0]),
        )


def test_relative_local_model_and_tracker_paths_are_resolved_from_matrix(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "custom.pt").write_bytes(b"local test weight")
    runs = [
        {
            "name": "local paths",
            "model": "weights/custom.pt",
            "confidence": 0.3,
            "image_size": 640,
            "tracker": {"config": "configs/bytetrack.yaml"},
            "enabled_views": ["full"],
        }
    ]
    matrix = load_experiment_matrix(write_matrix(tmp_path, runs=runs))

    resolved = resolve_run_configuration(
        matrix,
        matrix.runs[0],
        build_artifacts(tmp_path / "results", matrix.runs[0]),
    )

    assert resolved["model"]["path"] == str((weights / "custom.pt").resolve())
    assert resolved["tracking"]["tracker"] == str(
        (tmp_path / "configs" / "bytetrack.yaml").resolve()
    )


def test_missing_required_summary_field_marks_run_failed(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake video")

    def incomplete_executor(
        _input: Path, _config: Path, _artifacts: ExperimentArtifacts
    ) -> dict[str, Any]:
        summary = complete_summary()
        summary.pop("fused_detections_total")
        return summary

    ranked = run_experiments(
        input_path=input_path,
        matrix=matrix,
        output_dir=tmp_path / "output",
        executor=incomplete_executor,
    )

    assert all(item["status"] == "failed" for item in ranked)
    assert all("fused_detections_total" in item["error"] for item in ranked)
    assert all(item["fused_detections"] is None for item in ranked)


def test_runner_refuses_stale_per_run_artifacts(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake video")
    output_dir = tmp_path / "output"
    stale = build_artifacts(output_dir, matrix.runs[0])
    stale.run_dir.mkdir(parents=True)
    stale.summary_json.write_text('{"stale": true}\n', encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="stale run artifacts"):
        run_experiments(
            input_path=input_path,
            matrix=matrix,
            output_dir=output_dir,
            executor=lambda *_args: complete_summary(),
        )

    assert json.loads(stale.summary_json.read_text("utf-8")) == {"stale": True}


def test_leaderboard_report_records_reproducibility_metadata(tmp_path: Path) -> None:
    first = result("first", count_error=0, risk=1, fps=12.5, f1=None)
    metadata = {
        "input_video": {
            "filename": "video.mp4",
            "size_bytes": 123,
            "sha256": "abc123",
        },
        "runtime": {
            "python_version": "3.test",
            "platform": "test-platform",
            "packages": {"trackbus-ai": "0.2.0", "ultralytics": "8.4.95"},
        },
    }

    write_leaderboard(
        [first],
        tmp_path,
        expected_entered=6,
        expected_exited=2,
        has_frame_ground_truth=False,
        reproducibility=metadata,
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text("utf-8"))
    report = (tmp_path / "report.md").read_text("utf-8")
    assert payload["reproducibility"] == metadata
    assert "Input SHA-256: `abc123`" in report
    assert "ultralytics=8.4.95" in report


def test_ground_truth_rejects_wrong_video_and_partial_annotations(
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(write_matrix(tmp_path))
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake video")
    annotations = tmp_path / "events.json"
    annotations.write_text(
        json.dumps(
            {
                "schema_version": "trackbus.events/v1",
                "video": {"filename": "other.mp4"},
                "source_fps": 25.0,
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigError, match="different video"):
        run_experiments(
            input_path=input_path,
            matrix=matrix,
            output_dir=tmp_path / "wrong_video",
            ground_truth_path=annotations,
            executor=lambda *_args: complete_summary(),
        )

    raw = json.loads(annotations.read_text("utf-8"))
    raw["video"]["filename"] = "video.mp4"
    annotations.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="partial"):
        run_experiments(
            input_path=input_path,
            matrix=matrix,
            output_dir=tmp_path / "partial",
            ground_truth_path=annotations,
            executor=lambda *_args: complete_summary(),
        )


def test_ground_truth_fps_mismatch_marks_run_failed(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(
        write_matrix(tmp_path, expected_entered=0, expected_exited=0)
    )
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake video")
    annotations = tmp_path / "events.json"
    annotations.write_text(
        json.dumps(
            {
                "schema_version": "trackbus.events/v1",
                "video": {"filename": "video.mp4"},
                "source_fps": 25.0,
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    ranked = run_experiments(
        input_path=input_path,
        matrix=matrix,
        output_dir=tmp_path / "output",
        ground_truth_path=annotations,
        executor=lambda *_args: complete_summary(source_fps=30.0),
    )

    assert all(item["status"] == "failed" for item in ranked)
    assert all("source FPS" in item["error"] for item in ranked)


def test_subprocess_executor_uses_repository_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = ExperimentRun(
        name="cwd",
        description="",
        model="yolo11n.pt",
        confidence=0.25,
        image_size=640,
        tracker={"config": "bytetrack.yaml"},
        enabled_views=("full",),
        detection_fusion={"method": "nms", "iou_threshold": 0.5},
        overrides={},
    )
    artifacts = build_artifacts(tmp_path, run)
    artifacts.run_dir.mkdir(parents=True)
    config_path = artifacts.resolved_config
    config_path.write_text("{}\n", encoding="utf-8")
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"fake")
    observed: dict[str, Any] = {}

    def fake_run(_command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        artifacts.summary_json.write_text(
            json.dumps(complete_summary()), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("trackbus.experiments.subprocess.run", fake_run)

    summary = subprocess_executor(input_path, config_path, artifacts)

    assert summary["entered_total"] == 6
    cwd = Path(observed["cwd"])
    assert (cwd / "pyproject.toml").is_file()
    assert (cwd / "trackbus").is_dir()
