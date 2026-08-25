"""Compare person and head TrackBus replays against one explicit count truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HeadComparisonError(ValueError):
    """Raised when two logs are not a valid same-clip A/B experiment."""


@dataclass(frozen=True)
class ReplayLog:
    path: Path
    run: Mapping[str, Any]
    summary: Mapping[str, Any]

    @property
    def target(self) -> str:
        return str(self.summary.get("detectionTarget", ""))


def load_replay_log(path: Path) -> ReplayLog:
    resolved = path.expanduser().resolve()
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HeadComparisonError(
            f"Could not read replay log '{resolved}': {exc}"
        ) from exc
    runs: list[Mapping[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise HeadComparisonError(
                f"Replay log {resolved} line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise HeadComparisonError(
                f"Replay log {resolved} line {line_number} must be an object."
            )
        if row.get("recordType") == "trajectory-run":
            runs.append(row)
        elif row.get("recordType") == "trajectory-summary":
            summaries.append(row)
    if len(runs) != 1 or len(summaries) != 1:
        raise HeadComparisonError(
            f"Replay log {resolved} must contain exactly one trajectory-run and "
            "one trajectory-summary record."
        )
    return ReplayLog(resolved, runs[0], summaries[0])


def compare_replays(
    person: ReplayLog,
    head: ReplayLog,
    *,
    ground_truth_in: int,
    ground_truth_out: int,
) -> dict[str, Any]:
    if ground_truth_in < 0 or ground_truth_out < 0:
        raise HeadComparisonError("Ground-truth counts cannot be negative.")
    if person.path == head.path:
        raise HeadComparisonError(
            "Person and head replay logs must be different files."
        )
    if person.target != "person":
        raise HeadComparisonError(
            f"Person replay target is '{person.target}', expected 'person'."
        )
    if head.target != "head":
        raise HeadComparisonError(
            f"Head replay target is '{head.target}', expected 'head'."
        )
    person_source = _normalized_source(person.run.get("source"))
    head_source = _normalized_source(head.run.get("source"))
    if not person_source or person_source != head_source:
        raise HeadComparisonError(
            "Person and head logs must come from the exact same video source."
        )
    person_frames = _non_negative_int(
        person.summary.get("processedFrames"), "person processedFrames"
    )
    head_frames = _non_negative_int(
        head.summary.get("processedFrames"), "head processedFrames"
    )
    if person_frames == 0 or head_frames == 0 or person_frames != head_frames:
        raise HeadComparisonError(
            "Person and head logs must contain the same non-zero processed frame count."
        )
    truth = {"in": ground_truth_in, "out": ground_truth_out}
    person_result = _score_replay(person, truth)
    head_result = _score_replay(head, truth)
    person_error = person_result["total_absolute_count_error"]
    head_error = head_result["total_absolute_count_error"]
    winner = (
        "head"
        if head_error < person_error
        else "person"
        if person_error < head_error
        else "tie"
    )
    return {
        "schema_version": 1,
        "workflow": "trackbus_head_vs_person_replay",
        "valid_same_clip_comparison": True,
        "source": person.run.get("source"),
        "processed_frames": person_frames,
        "ground_truth": truth,
        "results": {"person": person_result, "head": head_result},
        "winner_by_total_absolute_count_error": winner,
        "head_improvement_in_absolute_events": person_error - head_error,
        "decision_rule": (
            "Lower |predicted IN - true IN| + |predicted OUT - true OUT| wins."
        ),
        "limitations": [
            "This comparison covers one labelled clip only.",
            (
                "Count agreement alone does not measure identity continuity or "
                "box accuracy."
            ),
            "No production accuracy claim is made from this experiment.",
        ],
        "accuracy_claimed": False,
    }


def _score_replay(log: ReplayLog, truth: Mapping[str, int]) -> dict[str, Any]:
    boardings = _non_negative_int(log.summary.get("boardings"), "boardings")
    alightings = _non_negative_int(log.summary.get("alightings"), "alightings")
    counting = log.summary.get("counting")
    continuity = log.summary.get("continuity")
    return {
        "log": str(log.path),
        "model": log.run.get("model"),
        "in": boardings,
        "out": alightings,
        "in_absolute_error": abs(boardings - truth["in"]),
        "out_absolute_error": abs(alightings - truth["out"]),
        "total_absolute_count_error": (
            abs(boardings - truth["in"]) + abs(alightings - truth["out"])
        ),
        "confirmed_events": (
            counting.get("confirmed_events") if isinstance(counting, dict) else None
        ),
        "stitched_track_fragments": (
            continuity.get("stitched_track_fragments")
            if isinstance(continuity, dict)
            else None
        ),
        "accuracy_claimed": False,
    }


def _normalized_source(value: object) -> str:
    return value.strip().replace("\\", "/").casefold() if isinstance(value, str) else ""


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HeadComparisonError(f"{name} must be a non-negative integer.")
    return value


def write_comparison(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.compare_head_test",
        description=(
            "Compare same-clip person and head replay counts without overclaiming."
        ),
    )
    parser.add_argument("--person-log", required=True, type=Path)
    parser.add_argument("--head-log", required=True, type=Path)
    parser.add_argument("--ground-truth-in", required=True, type=int)
    parser.add_argument("--ground-truth-out", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        person = load_replay_log(args.person_log)
        head = load_replay_log(args.head_log)
        report = compare_replays(
            person,
            head,
            ground_truth_in=args.ground_truth_in,
            ground_truth_out=args.ground_truth_out,
        )
        write_comparison(args.output, report)
    except (HeadComparisonError, OSError, ValueError) as exc:
        print(f"TrackBus head comparison error: {exc}", file=sys.stderr)
        return 2
    print(
        "Head/person A/B report written: "
        f"{args.output.expanduser().resolve()} · winner="
        f"{report['winner_by_total_absolute_count_error']}"
    )
    print("This one-clip experiment does not establish production accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
