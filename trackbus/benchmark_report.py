"""Combine existing TrackBus artifacts without mixing metric meanings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class BenchmarkReportError(ValueError):
    """Raised when benchmark artifacts cannot support an honest report."""


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load one local JSON object with a source-specific error."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkReportError(f"Could not read '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkReportError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise BenchmarkReportError(f"'{path}' must contain a JSON object.")
    return document


def build_benchmark_report(
    *,
    detection_artifact: Mapping[str, Any] | None = None,
    processing_summary: Mapping[str, Any] | None = None,
    counting_evaluation: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create four independent evidence sections from existing run artifacts."""

    if not any((detection_artifact, processing_summary, counting_evaluation)):
        raise BenchmarkReportError("At least one benchmark artifact is required.")
    detection = _detection_section(detection_artifact)
    tracking = _tracking_section(processing_summary)
    counting = _counting_section(counting_evaluation, processing_summary)
    runtime = _runtime_section(detection_artifact, processing_summary)
    accuracy_sections = [
        name
        for name, section in {
            "detection": detection,
            "tracking": tracking,
            "counting": counting,
        }.items()
        if section["status"] == "ground_truth_evaluated"
    ]
    return {
        "schema_version": 1,
        "benchmark_type": "trackbus_separated_pipeline_report",
        "metric_contract": {
            "sections_are_not_interchangeable": True,
            "ground_truth_accuracy_sections": accuracy_sections,
            "diagnostics_are_not_accuracy": True,
            "event_f1_is_not_detector_f1": True,
        },
        "sources": dict(source_paths or {}),
        "detection": detection,
        "tracking": tracking,
        "counting": counting,
        "runtime": runtime,
        "effective_configuration": _effective_configuration(
            detection_artifact,
            processing_summary,
        ),
        "limitations": _limitations(detection, tracking, counting, runtime),
    }


def write_benchmark_report(
    json_path: Path,
    csv_path: Path,
    report: Mapping[str, Any],
) -> None:
    """Write the complete report and a one-row comparison-friendly CSV."""

    if json_path.expanduser().resolve() == csv_path.expanduser().resolve():
        raise BenchmarkReportError("JSON and CSV outputs must be different files.")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    row = _csv_row(report)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _detection_section(
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if artifact is None:
        return _unavailable("detector benchmark not supplied")
    summary = artifact.get("summary", artifact)
    if (
        not isinstance(summary, Mapping)
        or summary.get("benchmark_type") != "detector_only"
    ):
        raise BenchmarkReportError(
            "Detection artifact must be a TrackBus detector-only benchmark."
        )
    truth = summary.get("ground_truth")
    if truth is not None and not isinstance(truth, Mapping):
        raise BenchmarkReportError("Detection ground_truth must be an object or null.")
    status = (
        "ground_truth_evaluated"
        if isinstance(truth, Mapping)
        else "diagnostics_only_no_box_ground_truth"
    )
    return {
        "status": status,
        "precision": truth.get("precision") if truth else None,
        "recall": truth.get("recall") if truth else None,
        "f1": truth.get("f1") if truth else None,
        "true_positives": truth.get("true_positives") if truth else None,
        "false_positives": truth.get("false_positives") if truth else None,
        "missed_people": truth.get("missed_persons") if truth else None,
        "diagnostics": summary.get("detections"),
        "box_stability_proxy": summary.get("frame_to_frame_box_stability_proxy"),
        "accuracy_claimed": status == "ground_truth_evaluated",
    }


def _tracking_section(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return _unavailable("processing summary not supplied")
    track_diagnostics = summary.get("track_diagnostics", [])
    if not isinstance(track_diagnostics, list):
        raise BenchmarkReportError("track_diagnostics must be a list when supplied.")
    lifetimes: list[float] = []
    observed: list[float] = []
    gap_frames = 0
    for index, track in enumerate(track_diagnostics):
        if not isinstance(track, Mapping):
            raise BenchmarkReportError(f"track_diagnostics[{index}] must be an object.")
        first = _optional_number(track.get("first_observed_frame"))
        last = _optional_number(track.get("last_observed_frame"))
        observations = _optional_number(track.get("observed_frames"))
        if first is not None and last is not None and last >= first:
            lifetimes.append(last - first + 1)
        if observations is not None:
            observed.append(observations)
        gap_frames += sum(
            int(value)
            for key, value in track.items()
            if key.endswith("_detection_gap_frames")
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
    continuity = summary.get("tracking_continuity", {})
    if not isinstance(continuity, Mapping):
        continuity = {}
    return {
        "status": "diagnostics_only_no_track_identity_ground_truth",
        "fragmentation_proxy": {
            "unique_tracking_ids": summary.get("unique_tracking_ids"),
            "possible_id_restarts": summary.get("possible_id_restart_count"),
            "diagnostic_gap_frames": gap_frames,
        },
        "id_switches": None,
        "id_switches_reason": "track identity ground truth was not supplied",
        "recovered_gaps": {
            "continuity_enabled": continuity.get("enabled", False),
            "stitched_track_fragments": continuity.get("stitched_track_fragments"),
            "maximum_stitched_gap_frames": continuity.get(
                "maximum_stitched_gap_frames"
            ),
        },
        "track_lifetime_frames": _distribution(lifetimes),
        "track_observed_frames": _distribution(observed),
        "accuracy_claimed": False,
    }


def _counting_section(
    evaluation: Mapping[str, Any] | None,
    processing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if evaluation is None:
        if processing is None:
            return _unavailable(
                "counting evaluation and processing summary not supplied"
            )
        return {
            "status": "diagnostics_only_no_event_ground_truth",
            "precision": None,
            "recall": None,
            "f1": None,
            "counts": {
                "predicted": {
                    "entered": processing.get("entered_total"),
                    "exited": processing.get("exited_total"),
                }
            },
            "accuracy_claimed": False,
        }
    mode = evaluation.get("evaluation_mode")
    overall = evaluation.get("overall")
    if not isinstance(overall, Mapping):
        raise BenchmarkReportError("Counting evaluation overall must be an object.")
    evaluated = mode == "event" and evaluation.get("aggregate_only") is not True
    return {
        "status": (
            "ground_truth_evaluated" if evaluated else "aggregate_diagnostics_only"
        ),
        "precision": overall.get("precision") if evaluated else None,
        "recall": overall.get("recall") if evaluated else None,
        "f1": overall.get("f1") if evaluated else None,
        "true_positive_events": (
            overall.get("true_positive_events") if evaluated else None
        ),
        "false_positive_events": (
            overall.get("false_positive_events") if evaluated else None
        ),
        "false_negative_events": (
            overall.get("false_negative_events") if evaluated else None
        ),
        "counts": evaluation.get("counts"),
        "count_errors": evaluation.get("count_errors"),
        "accuracy_claimed": evaluated,
    }


def _runtime_section(
    detection_artifact: Mapping[str, Any] | None,
    processing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detection_summary = (
        detection_artifact.get("summary", detection_artifact)
        if detection_artifact is not None
        else {}
    )
    detector_performance = (
        detection_summary.get("performance", {})
        if isinstance(detection_summary, Mapping)
        else {}
    )
    detector_config = (
        detection_summary.get("configuration", {})
        if isinstance(detection_summary, Mapping)
        else {}
    )
    return {
        "status": "measured" if detection_artifact or processing else "unavailable",
        "detector": {
            "pipeline_fps_including_decode": detector_performance.get(
                "pipeline_fps_including_decode"
            ),
            "inference_latency_ms": detector_performance.get("inference_latency_ms"),
            "preprocessing_latency_ms": detector_performance.get(
                "preprocessing_latency_ms"
            ),
            "device": detector_config.get("device"),
        },
        "end_to_end": {
            "average_fps": processing.get("average_fps") if processing else None,
            "processing_time_seconds": (
                processing.get("processing_time_seconds") if processing else None
            ),
            "processed_frames": (
                processing.get("processed_frames") if processing else None
            ),
            "device": processing.get("device_used") if processing else None,
        },
        "memory": {
            "peak_bytes": None,
            "status": "not_recorded_by_supplied_artifacts",
        },
    }


def _effective_configuration(
    detection_artifact: Mapping[str, Any] | None,
    processing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detection_summary = (
        detection_artifact.get("summary", detection_artifact)
        if detection_artifact is not None
        else {}
    )
    detector_config = (
        detection_summary.get("configuration", {})
        if isinstance(detection_summary, Mapping)
        else {}
    )
    pipeline_keys = (
        "model_used",
        "confidence_threshold",
        "detector_floor",
        "inference_image_size",
        "model_precision",
        "device_used",
        "tracker_backend",
        "tracker_config_path",
        "tracker_effective_config",
        "tracking_continuity",
        "enabled_inference_views",
        "detection_fusion_method",
        "detection_fusion_iou_threshold",
        "zone_anchor",
        "zone_boundary_hysteresis",
        "event_stability_settings",
    )
    return {
        "detector_benchmark": dict(detector_config)
        if isinstance(detector_config, Mapping)
        else {},
        "pipeline": (
            {key: processing.get(key) for key in pipeline_keys}
            if processing is not None
            else {}
        ),
    }


def _limitations(
    detection: Mapping[str, Any],
    tracking: Mapping[str, Any],
    counting: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> list[str]:
    limitations: list[str] = []
    if detection["status"] != "ground_truth_evaluated":
        limitations.append("Detection precision/recall requires box ground truth.")
    if tracking["status"] != "ground_truth_evaluated":
        limitations.append("ID switches require track identity ground truth.")
    if counting["status"] != "ground_truth_evaluated":
        limitations.append("Counting precision/recall/F1 requires event ground truth.")
    if runtime.get("memory", {}).get("peak_bytes") is None:
        limitations.append("Peak CPU/GPU memory was not recorded.")
    return limitations


def _csv_row(report: Mapping[str, Any]) -> dict[str, object]:
    detection = report["detection"]
    tracking = report["tracking"]
    counting = report["counting"]
    runtime = report["runtime"]
    return {
        "detection_status": detection["status"],
        "detection_precision": detection.get("precision"),
        "detection_recall": detection.get("recall"),
        "detection_f1": detection.get("f1"),
        "missed_people": detection.get("missed_people"),
        "false_positive_detections": detection.get("false_positives"),
        "tracking_status": tracking["status"],
        "possible_id_restarts": tracking.get("fragmentation_proxy", {}).get(
            "possible_id_restarts"
        ),
        "id_switches": tracking.get("id_switches"),
        "stitched_track_fragments": tracking.get("recovered_gaps", {}).get(
            "stitched_track_fragments"
        ),
        "median_track_lifetime_frames": tracking.get("track_lifetime_frames", {}).get(
            "p50"
        ),
        "counting_status": counting["status"],
        "counting_precision": counting.get("precision"),
        "counting_recall": counting.get("recall"),
        "counting_f1": counting.get("f1"),
        "detector_pipeline_fps": runtime.get("detector", {}).get(
            "pipeline_fps_including_decode"
        ),
        "end_to_end_fps": runtime.get("end_to_end", {}).get("average_fps"),
        "device": runtime.get("end_to_end", {}).get("device")
        or runtime.get("detector", {}).get("device"),
        "peak_memory_bytes": runtime.get("memory", {}).get("peak_bytes"),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "p50": None, "maximum": None, "mean": None}
    ordered = sorted(values)
    middle = statistics.median(ordered)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6),
        "p50": round(middle, 6),
        "maximum": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "accuracy_claimed": False}


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.benchmark_report",
        description=(
            "Combine detector, tracking, counting, and runtime artifacts without "
            "mixing their metric meanings."
        ),
    )
    parser.add_argument("--detection", type=Path, help="Detector benchmark JSON")
    parser.add_argument("--processing", type=Path, help="Pipeline summary JSON")
    parser.add_argument("--counting", type=Path, help="Event evaluation JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Defaults beside output JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected = {
            "detection": args.detection,
            "processing": args.processing,
            "counting": args.counting,
        }
        if not any(selected.values()):
            raise BenchmarkReportError(
                "Supply at least one of --detection, --processing, or --counting."
            )
        output = args.output.expanduser().resolve()
        csv_path = (
            args.csv.expanduser().resolve()
            if args.csv is not None
            else output.with_suffix(".csv")
        )
        inputs = {
            name: path.expanduser().resolve()
            for name, path in selected.items()
            if path is not None
        }
        if output in inputs.values() or csv_path in inputs.values():
            raise BenchmarkReportError("Outputs cannot overwrite input artifacts.")
        loaded = {name: load_json_artifact(path) for name, path in inputs.items()}
        report = build_benchmark_report(
            detection_artifact=loaded.get("detection"),
            processing_summary=loaded.get("processing"),
            counting_evaluation=loaded.get("counting"),
            source_paths={name: str(path) for name, path in inputs.items()},
        )
        write_benchmark_report(output, csv_path, report)
        print(f"Separated benchmark JSON: {output}")
        print(f"Comparison CSV: {csv_path}")
        for section in ("detection", "tracking", "counting", "runtime"):
            print(f"{section}: {report[section]['status']}")
        return 0
    except (BenchmarkReportError, OSError, ValueError) as exc:
        print(f"TrackBus benchmark report error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
