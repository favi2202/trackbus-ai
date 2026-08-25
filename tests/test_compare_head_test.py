from __future__ import annotations

import json
from pathlib import Path

import pytest

from trackbus.compare_head_test import (
    HeadComparisonError,
    compare_replays,
    load_replay_log,
    main,
)


def _write_log(
    path: Path,
    *,
    target: str,
    boardings: int,
    alightings: int,
    frames: int = 589,
    source: str = r"C:\TrackBus\data\input\bus-test-1.mp4",
) -> Path:
    records = [
        {
            "recordType": "trajectory-run",
            "source": source,
            "detectionTarget": target,
            "model": f"{target}.pt",
        },
        {
            "recordType": "trajectory-summary",
            "detectionTarget": target,
            "processedFrames": frames,
            "boardings": boardings,
            "alightings": alightings,
            "counting": {"confirmed_events": boardings + alightings},
            "continuity": {"stitched_track_fragments": 2},
            "accuracyClaimed": False,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_head_wins_when_it_is_closer_to_explicit_six_in_two_out(
    tmp_path: Path,
) -> None:
    person = load_replay_log(
        _write_log(
            tmp_path / "person.jsonl",
            target="person",
            boardings=3,
            alightings=2,
        )
    )
    head = load_replay_log(
        _write_log(tmp_path / "head.jsonl", target="head", boardings=5, alightings=2)
    )

    report = compare_replays(person, head, ground_truth_in=6, ground_truth_out=2)

    assert report["valid_same_clip_comparison"] is True
    assert report["results"]["person"]["total_absolute_count_error"] == 3
    assert report["results"]["head"]["total_absolute_count_error"] == 1
    assert report["winner_by_total_absolute_count_error"] == "head"
    assert report["head_improvement_in_absolute_events"] == 2
    assert report["accuracy_claimed"] is False


def test_comparison_rejects_different_clip_or_frame_count(tmp_path: Path) -> None:
    person = load_replay_log(
        _write_log(
            tmp_path / "person.jsonl",
            target="person",
            boardings=3,
            alightings=2,
        )
    )
    other_source = load_replay_log(
        _write_log(
            tmp_path / "head-source.jsonl",
            target="head",
            boardings=5,
            alightings=2,
            source="other.mp4",
        )
    )
    other_frames = load_replay_log(
        _write_log(
            tmp_path / "head-frames.jsonl",
            target="head",
            boardings=5,
            alightings=2,
            frames=500,
        )
    )

    with pytest.raises(HeadComparisonError, match="exact same video"):
        compare_replays(person, other_source, ground_truth_in=6, ground_truth_out=2)
    with pytest.raises(HeadComparisonError, match="same non-zero processed frame"):
        compare_replays(person, other_frames, ground_truth_in=6, ground_truth_out=2)


def test_loader_requires_run_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.jsonl"
    path.write_text('{"recordType":"trajectory-run"}\n', encoding="utf-8")

    with pytest.raises(HeadComparisonError, match="exactly one"):
        load_replay_log(path)


def test_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    person = _write_log(
        tmp_path / "person.jsonl", target="person", boardings=3, alightings=2
    )
    head = _write_log(tmp_path / "head.jsonl", target="head", boardings=6, alightings=2)
    output = tmp_path / "comparison.json"

    assert (
        main(
            [
                "--person-log",
                str(person),
                "--head-log",
                str(head),
                "--ground-truth-in",
                "6",
                "--ground-truth-out",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))[
        "winner_by_total_absolute_count_error"
    ] == "head"
