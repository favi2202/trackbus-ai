"""Optional frame-level raw, fused, and tracked diagnostic CSV exports."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path
from typing import IO

from trackbus.detection import Detection, TrackedDetection

RAW_FIELDS = (
    "frame",
    "timestamp",
    "source_view",
    "left",
    "top",
    "right",
    "bottom",
    "confidence",
    "class_id",
    "excluded",
)
FUSED_FIELDS = (
    "frame",
    "left",
    "top",
    "right",
    "bottom",
    "confidence",
    "class_id",
    "contributing_views",
    "fusion_metadata",
)
TRACK_FIELDS = (
    "frame",
    "tracking_id",
    "left",
    "top",
    "right",
    "bottom",
    "confidence",
    "class_id",
    "anchor_x",
    "anchor_y",
    "current_zone",
    "current_lane",
    "counter_state",
    "contributing_views",
)


def derive_detection_export_paths(
    output_video: Path,
    raw_path: Path | None = None,
    fused_path: Path | None = None,
    tracks_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve optional diagnostic paths beside the annotated video."""

    return (
        raw_path or output_video.with_name(f"{output_video.stem}.raw_detections.csv"),
        fused_path
        or output_video.with_name(f"{output_video.stem}.fused_detections.csv"),
        tracks_path or output_video.with_name(f"{output_video.stem}.tracks.csv"),
    )


class DetectionCsvExporter:
    """Create all three potentially large files only when explicitly enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        raw_path: Path,
        fused_path: Path,
        tracks_path: Path,
    ) -> None:
        self.enabled = enabled
        self.raw_path = raw_path
        self.fused_path = fused_path
        self.tracks_path = tracks_path
        self._stack: ExitStack | None = None
        self._files: list[IO[str]] = []
        self._raw: csv.DictWriter[str] | None = None
        self._fused: csv.DictWriter[str] | None = None
        self._tracks: csv.DictWriter[str] | None = None

    def __enter__(self) -> DetectionCsvExporter:
        if not self.enabled:
            return self
        self._stack = ExitStack()
        writers: list[csv.DictWriter[str]] = []
        for path, fields in (
            (self.raw_path, RAW_FIELDS),
            (self.fused_path, FUSED_FIELDS),
            (self.tracks_path, TRACK_FIELDS),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            file = self._stack.enter_context(
                path.open("w", encoding="utf-8", newline="")
            )
            self._files.append(file)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writers.append(writer)
        self._raw, self._fused, self._tracks = writers
        return self

    def __exit__(self, *_args: object) -> None:
        if self._stack is not None:
            self._stack.close()

    def record_raw(
        self,
        frame: int,
        timestamp_seconds: float,
        detections: list[tuple[Detection, bool]],
    ) -> None:
        if self._raw is None:
            return
        for detection, excluded in detections:
            left, top, right, bottom = detection.bounding_box
            self._raw.writerow(
                {
                    "frame": frame,
                    "timestamp": f"{timestamp_seconds:.6f}",
                    "source_view": detection.source_view,
                    "left": _number(left),
                    "top": _number(top),
                    "right": _number(right),
                    "bottom": _number(bottom),
                    "confidence": f"{detection.confidence:.6f}",
                    "class_id": detection.class_id,
                    "excluded": str(excluded).lower(),
                }
            )
        self._flush()

    def record_fused(self, frame: int, detections: list[Detection]) -> None:
        if self._fused is None:
            return
        for detection in detections:
            left, top, right, bottom = detection.bounding_box
            self._fused.writerow(
                {
                    "frame": frame,
                    "left": _number(left),
                    "top": _number(top),
                    "right": _number(right),
                    "bottom": _number(bottom),
                    "confidence": f"{detection.confidence:.6f}",
                    "class_id": detection.class_id,
                    "contributing_views": "|".join(detection.source_views),
                    "fusion_metadata": json.dumps(
                        dict(detection.metadata), sort_keys=True, default=str
                    ),
                }
            )
        self._flush()

    def record_tracks(
        self,
        frame: int,
        tracks: list[TrackedDetection],
        zones: dict[int, str],
        lanes: dict[int, str | None],
        counter_states: dict[int, str | None],
    ) -> None:
        if self._tracks is None:
            return
        for tracked in tracks:
            left, top, right, bottom = tracked.bounding_box
            anchor_x, anchor_y = tracked.anchor
            self._tracks.writerow(
                {
                    "frame": frame,
                    "tracking_id": tracked.tracking_id,
                    "left": _number(left),
                    "top": _number(top),
                    "right": _number(right),
                    "bottom": _number(bottom),
                    "confidence": f"{tracked.confidence:.6f}",
                    "class_id": tracked.class_id,
                    "anchor_x": _number(anchor_x),
                    "anchor_y": _number(anchor_y),
                    "current_zone": zones[tracked.tracking_id],
                    "current_lane": lanes.get(tracked.tracking_id) or "",
                    "counter_state": counter_states.get(tracked.tracking_id) or "",
                    "contributing_views": "|".join(tracked.source_views),
                }
            )
        self._flush()

    def _flush(self) -> None:
        for file in self._files:
            file.flush()


def _number(value: float) -> str:
    return f"{value:.3f}"
