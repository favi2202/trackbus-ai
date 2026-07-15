"""Manual, frame-accurate crossing-event annotation for local videos.

The annotation data model is intentionally independent of OpenCV.  Importing this
module, loading an annotation file, and running ``--help`` therefore do not require
opening a video or creating a GUI window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AnnotationError(ValueError):
    """Raised when an annotation file or annotation operation is invalid."""


class EventDirection(StrEnum):
    """A manually observed completed crossing direction."""

    IN = "IN"
    OUT = "OUT"

    @classmethod
    def parse(cls, value: object) -> EventDirection:
        """Parse common direction spellings without silently accepting nonsense."""

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        aliases = {
            "IN": cls.IN,
            "ENTRY": cls.IN,
            "ENTERED": cls.IN,
            "OUT": cls.OUT,
            "EXIT": cls.OUT,
            "EXITED": cls.OUT,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise AnnotationError(
                f"direction must be IN or OUT, got {value!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class AnnotatedEvent:
    """One human-labelled crossing at a source-video frame."""

    frame: int
    timestamp: float
    direction: EventDirection
    note: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int):
            raise AnnotationError("event frame must be an integer")
        if self.frame < 0:
            raise AnnotationError("event frame cannot be negative")
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise AnnotationError("event timestamp must be a number of seconds")
        if not math.isfinite(float(self.timestamp)) or self.timestamp < 0:
            raise AnnotationError("event timestamp must be finite and non-negative")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "direction", EventDirection.parse(self.direction))
        if self.note is not None:
            if not isinstance(self.note, str):
                raise AnnotationError("event note must be text when supplied")
            stripped = self.note.strip()
            object.__setattr__(self, "note", stripped or None)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation of this event."""

        result: dict[str, Any] = {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "direction": self.direction.value,
        }
        if self.note is not None:
            result["note"] = self.note
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AnnotatedEvent:
        """Build an event from its JSON object representation."""

        try:
            frame = value["frame"]
            timestamp = value["timestamp"]
            direction = value["direction"]
        except KeyError as exc:
            raise AnnotationError(
                f"event is missing required field {exc.args[0]!r}"
            ) from exc
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise AnnotationError("event frame must be an integer")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise AnnotationError("event timestamp must be numeric")
        note = value.get("note")
        if note is not None and not isinstance(note, str):
            raise AnnotationError("event note must be text when supplied")
        return cls(
            frame=frame,
            timestamp=float(timestamp),
            direction=EventDirection.parse(direction),
            note=note,
        )


@dataclass(frozen=True, slots=True)
class GroundTruthAnnotations:
    """Serializable ground-truth document for one source video."""

    video_filename: str
    source_fps: float
    events: tuple[AnnotatedEvent, ...] = ()
    video_id: str | None = None
    schema_version: str = "trackbus.events/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "trackbus.events/v1":
            raise AnnotationError(
                f"unsupported annotation schema version: {self.schema_version!r}"
            )
        if not isinstance(self.video_filename, str) or not self.video_filename.strip():
            raise AnnotationError("video filename cannot be empty")
        if isinstance(self.source_fps, bool) or not isinstance(
            self.source_fps, (int, float)
        ):
            raise AnnotationError("source FPS must be numeric")
        if not math.isfinite(float(self.source_fps)) or self.source_fps <= 0:
            raise AnnotationError("source FPS must be finite and greater than zero")
        if self.video_id is not None and (
            not isinstance(self.video_id, str) or not self.video_id.strip()
        ):
            raise AnnotationError("video identifier cannot be empty when supplied")
        source_fps = float(self.source_fps)
        events = tuple(self.events)
        if any(not isinstance(event, AnnotatedEvent) for event in events):
            raise AnnotationError(
                "annotation events must contain AnnotatedEvent values"
            )
        ordered = tuple(
            sorted(events, key=lambda event: (event.frame, event.direction.value))
        )
        if events != ordered:
            raise AnnotationError("annotation events must be ordered by frame")
        seen: set[tuple[int, EventDirection]] = set()
        for event in events:
            identity = (event.frame, event.direction)
            if identity in seen:
                raise AnnotationError(
                    "duplicate "
                    f"{event.direction.value} annotation at frame {event.frame}"
                )
            seen.add(identity)
            expected_timestamp = event.frame / source_fps
            if not math.isclose(
                event.timestamp,
                expected_timestamp,
                rel_tol=1e-6,
                abs_tol=1e-3,
            ):
                raise AnnotationError(
                    "annotation timestamp does not match frame/source_fps "
                    f"at frame {event.frame}"
                )
        object.__setattr__(self, "video_filename", self.video_filename.strip())
        object.__setattr__(self, "source_fps", source_fps)
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, human-readable JSON data structure."""

        video: dict[str, str] = {"filename": self.video_filename}
        if self.video_id is not None:
            video["identifier"] = self.video_id
        return {
            "schema_version": self.schema_version,
            "video": video,
            "source_fps": self.source_fps,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GroundTruthAnnotations:
        """Validate and load an annotation document.

        ``fps`` is accepted as a legacy alias for ``source_fps`` so early local
        annotation files remain useful, while newly saved files always use the
        documented v1 key.
        """

        schema_version = value.get("schema_version", "trackbus.events/v1")
        if schema_version != "trackbus.events/v1":
            raise AnnotationError(
                f"unsupported annotation schema version: {schema_version!r}"
            )
        video = value.get("video")
        video_id: str | None = None
        if isinstance(video, str):
            filename = video
        elif isinstance(video, Mapping):
            filename = video.get("filename")
            identifier = video.get("identifier")
            if identifier is not None and not isinstance(identifier, str):
                raise AnnotationError("video identifier must be text")
            video_id = identifier
        else:
            raise AnnotationError("annotation field 'video' must be an object")
        if not isinstance(filename, str):
            raise AnnotationError("annotation video filename must be text")
        source_fps = value.get("source_fps", value.get("fps"))
        if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float)):
            raise AnnotationError("annotation source_fps must be numeric")
        raw_events = value.get("events", [])
        if not isinstance(raw_events, list):
            raise AnnotationError("annotation events must be a list")
        events: list[AnnotatedEvent] = []
        for index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping):
                raise AnnotationError(f"event {index} must be an object")
            try:
                events.append(AnnotatedEvent.from_dict(raw_event))
            except AnnotationError as exc:
                raise AnnotationError(f"invalid event {index}: {exc}") from exc
        return cls(
            video_filename=filename,
            video_id=video_id,
            source_fps=float(source_fps),
            events=tuple(events),
            schema_version=str(schema_version),
        )


def load_annotations(path: Path) -> GroundTruthAnnotations:
    """Load and validate a ground-truth JSON file."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AnnotationError("annotation file root must be an object")
    return GroundTruthAnnotations.from_dict(raw)


def save_annotations(path: Path, annotations: GroundTruthAnnotations) -> None:
    """Atomically save a ground-truth document as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(annotations.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(slots=True)
class AnnotationSession:
    """Mutable editor state separated from OpenCV rendering for easy testing."""

    video_filename: str
    source_fps: float
    video_id: str | None = None
    events: list[AnnotatedEvent] = field(default_factory=list)

    @classmethod
    def from_document(cls, document: GroundTruthAnnotations) -> AnnotationSession:
        return cls(
            video_filename=document.video_filename,
            source_fps=document.source_fps,
            video_id=document.video_id,
            events=list(document.events),
        )

    def mark(
        self,
        frame: int,
        direction: EventDirection | str,
        note: str | None = None,
    ) -> AnnotatedEvent:
        """Add an event unless that frame/direction is already marked."""

        event = AnnotatedEvent(
            frame=frame,
            timestamp=frame / self.source_fps,
            direction=EventDirection.parse(direction),
            note=note,
        )
        for existing in self.events:
            if existing.frame == event.frame and existing.direction is event.direction:
                return existing
        self.events.append(event)
        return event

    def undo(self) -> AnnotatedEvent | None:
        """Remove and return the most recently added or loaded annotation."""

        return self.events.pop() if self.events else None

    def to_document(self) -> GroundTruthAnnotations:
        """Return chronologically ordered, serializable session data."""

        ordered = sorted(
            self.events,
            key=lambda event: (event.frame, event.direction.value, event.timestamp),
        )
        return GroundTruthAnnotations(
            video_filename=self.video_filename,
            video_id=self.video_id,
            source_fps=self.source_fps,
            events=tuple(ordered),
        )


def _resume_session(
    output_path: Path,
    video_path: Path,
    source_fps: float,
    frame_count: int,
    video_id: str | None,
) -> AnnotationSession:
    if not output_path.exists():
        return AnnotationSession(video_path.name, source_fps, video_id)
    document = load_annotations(output_path)
    if document.video_filename != video_path.name:
        raise AnnotationError(
            "existing annotation file belongs to "
            f"{document.video_filename!r}, not {video_path.name!r}"
        )
    if not math.isclose(document.source_fps, source_fps, rel_tol=1e-4, abs_tol=1e-4):
        raise AnnotationError(
            "existing annotation FPS does not match the source video "
            f"({document.source_fps:g} != {source_fps:g})"
        )
    if video_id is not None and document.video_id not in (None, video_id):
        raise AnnotationError("existing annotation video identifier does not match")
    if any(event.frame >= frame_count for event in document.events):
        raise AnnotationError(
            "existing annotation contains a frame outside the source video"
        )
    session = AnnotationSession.from_document(document)
    if session.video_id is None:
        session.video_id = video_id
    return session


def _draw_annotation_overlay(
    cv2: Any,
    frame: Any,
    *,
    frame_number: int,
    frame_count: int,
    source_fps: float,
    events: list[AnnotatedEvent],
    playing: bool,
) -> Any:
    """Draw controls, current time, recent marks, and a compact event timeline."""

    display = frame.copy()
    height, width = display.shape[:2]
    scale = max(0.42, min(width, height) / 700.0)
    line = max(14, round(28 * scale))
    panel_height = min(height, line * 5 + 10)
    cv2.rectangle(display, (0, 0), (width, panel_height), (0, 0, 0), -1)
    status = "PLAYING" if playing else "PAUSED"
    timestamp = frame_number / source_fps
    entered_count = sum(event.direction is EventDirection.IN for event in events)
    exited_count = sum(event.direction is EventDirection.OUT for event in events)
    texts = [
        f"Frame {frame_number}/{max(0, frame_count - 1)}  {timestamp:.3f}s  {status}",
        "Space/P: play-pause   A/, left   D/. right",
        "I: mark IN   O: mark OUT   U: undo   S: save   Q/Esc: save+quit",
        f"Annotations: {len(events)} (IN {entered_count}, OUT {exited_count})",
    ]
    for row, text in enumerate(texts, start=1):
        cv2.putText(
            display,
            text,
            (8, row * line),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    # Timeline: cyan IN ticks, orange OUT ticks, and a white current-frame cursor.
    y = max(panel_height + 8, height - 18)
    x0, x1 = 8, max(9, width - 9)
    cv2.line(display, (x0, y), (x1, y), (140, 140, 140), 1)
    denominator = max(1, frame_count - 1)
    for event in events:
        x = x0 + round((x1 - x0) * min(event.frame, denominator) / denominator)
        color = (255, 220, 0) if event.direction is EventDirection.IN else (0, 140, 255)
        cv2.line(display, (x, y - 8), (x, y + 5), color, 2)
    current_x = x0 + round((x1 - x0) * frame_number / denominator)
    cv2.line(display, (current_x, y - 12), (current_x, y + 8), (255, 255, 255), 1)
    return display


def annotate_video(
    input_path: Path,
    output_path: Path,
    *,
    video_id: str | None = None,
) -> GroundTruthAnnotations:
    """Open an interactive OpenCV annotator and save/resume ground truth."""

    if input_path.expanduser().resolve() == output_path.expanduser().resolve():
        raise AnnotationError(
            "input video and annotation output must be different files"
        )

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - project dependency safety net
        raise AnnotationError("OpenCV is required for interactive annotation") from exc

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise AnnotationError(f"could not open input video: {input_path}")
    window_name = f"TrackBus event annotation - {input_path.name}"
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise AnnotationError("source video reports an invalid FPS")
        if frame_count <= 0:
            raise AnnotationError("source video contains no readable frames")
        session = _resume_session(
            output_path,
            input_path,
            source_fps,
            frame_count,
            video_id,
        )
        current_frame = min(
            session.events[-1].frame if session.events else 0,
            frame_count - 1,
        )
        playing = False
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ok, frame = capture.read()
            if not ok:
                raise AnnotationError(
                    f"could not decode requested source frame {current_frame}"
                )
            display = _draw_annotation_overlay(
                cv2,
                frame,
                frame_number=current_frame,
                frame_count=frame_count,
                source_fps=source_fps,
                events=session.events,
                playing=playing,
            )
            cv2.imshow(window_name, display)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                save_annotations(output_path, session.to_document())
                break
            delay = max(1, round(1000 / source_fps)) if playing else 0
            key = cv2.waitKeyEx(delay)

            if key in (ord("q"), ord("Q"), 27):
                save_annotations(output_path, session.to_document())
                break
            if key in (ord(" "), ord("p"), ord("P")):
                playing = not playing
            elif key in (ord("i"), ord("I")):
                session.mark(current_frame, EventDirection.IN)
            elif key in (ord("o"), ord("O")):
                session.mark(current_frame, EventDirection.OUT)
            elif key in (ord("u"), ord("U")):
                session.undo()
            elif key in (ord("s"), ord("S")):
                save_annotations(output_path, session.to_document())
            elif key in (ord("a"), ord("A"), ord(","), 81, 2424832):
                playing = False
                current_frame = max(0, current_frame - 1)
            elif key in (ord("d"), ord("D"), ord("."), 83, 2555904):
                playing = False
                current_frame = min(frame_count - 1, current_frame + 1)
            elif playing:
                if current_frame >= frame_count - 1:
                    playing = False
                else:
                    current_frame += 1
        return session.to_document()
    except cv2.error as exc:
        raise AnnotationError(
            "OpenCV could not create or update the annotation window; "
            "run this command in a graphical desktop session"
        ) from exc
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.annotate_events",
        description="Manually mark IN and OUT events in a local video.",
        epilog=(
            "Controls: Space/P play-pause; A or comma previous frame; D or period "
            "next frame; I mark IN; O mark OUT; U undo; S save; Q/Esc save and quit."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Local source video")
    parser.add_argument(
        "--output", required=True, type=Path, help="Ground-truth events JSON"
    )
    parser.add_argument(
        "--video-id", help="Optional stable video identifier stored with annotations"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input video does not exist: {input_path}", file=sys.stderr)
        return 2
    if input_path == output_path:
        print(
            "error: input video and annotation output must be different files",
            file=sys.stderr,
        )
        return 2
    try:
        annotations = annotate_video(
            input_path,
            output_path,
            video_id=args.video_id,
        )
    except (AnnotationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Saved {len(annotations.events)} events to {output_path} "
        f"(IN={sum(e.direction is EventDirection.IN for e in annotations.events)}, "
        f"OUT={sum(e.direction is EventDirection.OUT for e in annotations.events)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
