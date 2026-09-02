from __future__ import annotations

import json

import pytest

from trackbus.annotate_events import (
    AnnotatedEvent,
    AnnotationError,
    AnnotationSession,
    EventDirection,
    GroundTruthAnnotations,
    build_parser,
    load_annotations,
    save_annotations,
)
from trackbus.annotate_events import (
    main as annotation_main,
)


def test_annotation_document_round_trip_preserves_optional_note(tmp_path) -> None:
    path = tmp_path / "events.json"
    document = GroundTruthAnnotations(
        video_filename="door.mp4",
        video_id="bus-clip-01",
        source_fps=25,
        events=(
            AnnotatedEvent(15, 0.6, EventDirection.IN, "first passenger"),
            AnnotatedEvent(42, 1.68, EventDirection.OUT),
        ),
    )

    save_annotations(path, document)

    assert load_annotations(path) == document
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == "trackbus.events/v1"
    assert serialized["video"] == {
        "filename": "door.mp4",
        "identifier": "bus-clip-01",
    }
    assert serialized["source_fps"] == 25.0
    assert "note" not in serialized["events"][1]
    assert not (tmp_path / ".events.json.tmp").exists()


def test_session_marks_frame_accurate_events_avoids_duplicate_and_undoes() -> None:
    session = AnnotationSession("door.mp4", source_fps=25)

    marked = session.mark(50, "entered")
    duplicate = session.mark(50, EventDirection.IN)
    session.mark(75, EventDirection.OUT)

    assert marked.timestamp == 2.0
    assert duplicate is marked
    assert len(session.events) == 2
    assert session.undo() == AnnotatedEvent(75, 3.0, EventDirection.OUT)
    assert session.undo() == marked
    assert session.undo() is None


def test_session_serialization_sorts_events_without_changing_undo_history() -> None:
    session = AnnotationSession("door.mp4", source_fps=10)
    session.mark(30, "OUT")
    session.mark(10, "IN")

    document = session.to_document()

    assert [event.frame for event in document.events] == [10, 30]
    assert session.undo().frame == 10


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"frame": -1, "timestamp": 0.0, "direction": "IN"}, "negative"),
        ({"frame": 1, "timestamp": -0.1, "direction": "IN"}, "non-negative"),
        ({"frame": 1, "timestamp": 0.1, "direction": "sideways"}, "IN or OUT"),
    ],
)
def test_invalid_event_data_is_rejected(event, message) -> None:
    with pytest.raises(AnnotationError, match=message):
        AnnotatedEvent.from_dict(event)


def test_unknown_schema_is_rejected() -> None:
    with pytest.raises(AnnotationError, match="unsupported"):
        GroundTruthAnnotations.from_dict(
            {
                "schema_version": "trackbus.events/v99",
                "video": {"filename": "door.mp4"},
                "source_fps": 25,
                "events": [],
            }
        )


def test_annotation_parser_documents_controls(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "play-pause" in output
    assert "mark IN" in output
    assert "undo" in output
    assert "save and quit" in output


def test_annotation_cli_refuses_to_overwrite_input_video(tmp_path, capsys) -> None:
    video = tmp_path / "door.mp4"
    original = b"not-a-real-video"
    video.write_bytes(original)

    assert annotation_main(["--input", str(video), "--output", str(video)]) == 2

    assert video.read_bytes() == original
    assert "different files" in capsys.readouterr().err


def test_annotation_document_rejects_inconsistent_or_duplicate_events() -> None:
    with pytest.raises(AnnotationError, match="timestamp"):
        GroundTruthAnnotations(
            video_filename="door.mp4",
            source_fps=25,
            events=(AnnotatedEvent(25, 2.0, EventDirection.IN),),
        )
    duplicate = AnnotatedEvent(25, 1.0, EventDirection.IN)
    with pytest.raises(AnnotationError, match="duplicate"):
        GroundTruthAnnotations(
            video_filename="door.mp4",
            source_fps=25,
            events=(duplicate, duplicate),
        )


def test_annotation_document_rejects_direct_unknown_schema() -> None:
    with pytest.raises(AnnotationError, match="unsupported"):
        GroundTruthAnnotations(
            video_filename="door.mp4",
            source_fps=25,
            schema_version="trackbus.events/v99",
        )
