"""Opt-in, bounded failure evidence for TrackBus video diagnostics."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.config import FailureMiningConfig
from trackbus.counter import CounterSnapshot, CountEvent, EventType, SuppressionReason
from trackbus.detection import Detection, TrackedDetection
from trackbus.fusion import box_iou

LOGGER = logging.getLogger(__name__)


class FailureMiningError(RuntimeError):
    """Raised when explicitly requested evidence cannot be stored safely."""


def derive_failure_paths(
    output_video: Path,
    output_jsonl: Path | None = None,
    frames_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve local failure evidence beside the annotated video by default."""

    jsonl = output_jsonl or output_video.with_name(
        f"{output_video.stem}.failures.jsonl"
    )
    frames = frames_directory or output_video.with_name(
        f"{output_video.stem}.failure-frames"
    )
    return jsonl, frames


@dataclass
class _TrackEvidence:
    first_frame: int
    last_frame: int
    observed_frames: int = 1


class FailureMiner:
    """Write local JSONL evidence with optional, explicitly bounded frames.

    Track identifiers are anonymous and scoped to one video processing run. The
    miner never performs face analysis, never uploads evidence, and never writes
    a source frame unless ``capture_frames`` is explicitly enabled.
    """

    RELEVANT_SUPPRESSIONS = {
        SuppressionReason.BOUNDARY_JITTER,
        SuppressionReason.EXCESSIVE_DETECTION_GAP,
        SuppressionReason.MISSING_NEUTRAL_TRANSITION,
    }

    def __init__(
        self,
        config: FailureMiningConfig,
        *,
        output_path: Path,
        frames_directory: Path,
        low_confidence_threshold: float,
        edge_margin_pixels: int,
        heavy_overlap_iou: float,
        initial_occupancy: int,
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.frames_directory = frames_directory
        self.low_confidence_threshold = low_confidence_threshold
        self.edge_margin_pixels = edge_margin_pixels
        self.heavy_overlap_iou = heavy_overlap_iou
        self.initial_occupancy = initial_occupancy
        self._file: IO[str] | None = None
        self._records_written = 0
        self._records_dropped = 0
        self._captured_frames: dict[int, str] = {}
        self._record_types: Counter[str] = Counter()
        self._previous_track_ids: set[int] = set()
        self._missing_since: dict[int, int] = {}
        self._tracks: dict[int, _TrackEvidence] = {}
        self._last_snapshots: dict[int, CounterSnapshot] = {}
        self._reported_restart_pairs: set[tuple[int, int]] = set()
        self._last_suppression_signature: dict[int, tuple[str, ...]] = {}
        self._previous_detection_count: int | None = None
        self._finalized = False

    def __enter__(self) -> FailureMiner:
        if not self.config.enabled:
            return self
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8", newline="\n")
        if self.config.capture_frames:
            self.frames_directory.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                "Failure frame capture is enabled. Stored local images may contain "
                "passengers and must be handled as personal data: %s",
                self.frames_directory,
            )
        return self

    def __exit__(self, *_args: object) -> None:
        self.finalize()
        if self._file is not None:
            self._file.close()
            self._file = None

    def observe_frame(
        self,
        *,
        frame_number: int,
        timestamp_seconds: float,
        frame: NDArray[np.uint8],
        fused_detections: list[Detection],
        tracks: list[TrackedDetection],
        snapshots: dict[int, CounterSnapshot | None],
        events: list[CountEvent],
        possible_restarts: dict[int, int] | None = None,
    ) -> None:
        """Record selected failure signals from one already-processed frame."""

        if not self.config.enabled:
            return
        if self._file is None:
            raise FailureMiningError(
                "FailureMiner must be opened as a context manager when enabled."
            )
        current_ids = {track.tracking_id for track in tracks}
        for tracking_id in current_ids:
            evidence = self._tracks.get(tracking_id)
            if evidence is None:
                self._tracks[tracking_id] = _TrackEvidence(
                    first_frame=frame_number,
                    last_frame=frame_number,
                )
            else:
                evidence.last_frame = frame_number
                evidence.observed_frames += 1

        disappeared = self._previous_track_ids - current_ids
        for tracking_id in sorted(disappeared):
            self._missing_since.setdefault(tracking_id, frame_number)
            self._record(
                "track_disappeared",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=(tracking_id,),
                details={"last_observed_frame": frame_number - 1},
            )
        for tracking_id in sorted(current_ids & self._missing_since.keys()):
            missing_frame = self._missing_since.pop(tracking_id)
            self._record(
                "track_recovered",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=(tracking_id,),
                details={"gap_frames": max(0, frame_number - missing_frame)},
            )

        weak = [
            detection
            for detection in fused_detections
            if detection.confidence < self.low_confidence_threshold
        ]
        if weak:
            self._record(
                "low_confidence_detections",
                frame_number,
                timestamp_seconds,
                frame,
                details={
                    "count": len(weak),
                    "threshold": self.low_confidence_threshold,
                    "minimum_confidence": round(
                        min(detection.confidence for detection in weak), 4
                    ),
                    "maximum_confidence": round(
                        max(detection.confidence for detection in weak), 4
                    ),
                },
            )

        detection_count = len(fused_detections)
        if (
            self._previous_detection_count is not None
            and self._previous_detection_count >= 2
            and detection_count
            <= self._previous_detection_count * self.config.sudden_detection_drop_ratio
        ):
            self._record(
                "sudden_detection_drop",
                frame_number,
                timestamp_seconds,
                frame,
                details={
                    "previous_count": self._previous_detection_count,
                    "current_count": detection_count,
                    "trigger_ratio": self.config.sudden_detection_drop_ratio,
                },
            )
        self._previous_detection_count = detection_count

        edge_ids = tuple(
            sorted(
                track.tracking_id
                for track in tracks
                if _touches_edge(track, frame.shape, self.edge_margin_pixels)
            )
        )
        if edge_ids:
            self._record(
                "edge_clipped_tracks",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=edge_ids,
                details={"edge_margin_pixels": self.edge_margin_pixels},
            )

        overlap_pairs = [
            {
                "anonymous_track_ids": [first.tracking_id, second.tracking_id],
                "iou": round(box_iou(first.bounding_box, second.bounding_box), 4),
            }
            for index, first in enumerate(tracks)
            for second in tracks[index + 1 :]
            if box_iou(first.bounding_box, second.bounding_box)
            >= self.heavy_overlap_iou
        ]
        if overlap_pairs:
            overlap_ids = tuple(
                sorted(
                    {
                        tracking_id
                        for pair in overlap_pairs
                        for tracking_id in pair["anonymous_track_ids"]
                    }
                )
            )
            self._record(
                "heavy_track_overlap",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=overlap_ids,
                details={
                    "threshold_iou": self.heavy_overlap_iou,
                    "pairs": overlap_pairs,
                },
            )

        stitched = tuple(
            sorted(
                track.tracking_id
                for track in tracks
                if track.metadata.get("continuity_stitched") is True
            )
        )
        if stitched:
            self._record(
                "continuity_stitch",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=stitched,
                details={"geometry_based": True, "biometric_identity_used": False},
            )

        for tracking_id, previous_id in sorted((possible_restarts or {}).items()):
            restart_pair = (previous_id, tracking_id)
            if restart_pair in self._reported_restart_pairs:
                continue
            self._reported_restart_pairs.add(restart_pair)
            self._record(
                "possible_track_restart",
                frame_number,
                timestamp_seconds,
                frame,
                anonymous_track_ids=(previous_id, tracking_id),
                details={"diagnostic_only": True},
            )

        for tracking_id, snapshot in snapshots.items():
            if snapshot is None:
                continue
            self._last_snapshots[tracking_id] = snapshot
            relevant = tuple(
                reason.value
                for reason in snapshot.suppression_reasons
                if reason in self.RELEVANT_SUPPRESSIONS
            )
            previous_suppression = self._last_suppression_signature.get(tracking_id)
            if relevant and previous_suppression != relevant:
                self._record(
                    "ambiguous_zone_transition",
                    frame_number,
                    timestamp_seconds,
                    frame,
                    anonymous_track_ids=(tracking_id,),
                    details={
                        "suppression_reasons": list(relevant),
                        "raw_zone": snapshot.raw_zone.value,
                        "effective_zone": snapshot.effective_zone.value,
                        "pending_direction": (
                            snapshot.pending_direction.value
                            if snapshot.pending_direction is not None
                            else None
                        ),
                    },
                )
            if relevant:
                self._last_suppression_signature[tracking_id] = relevant
            else:
                self._last_suppression_signature.pop(tracking_id, None)

        for event in events:
            if (
                event.event_type is EventType.OUT
                and event.exited_total > self.initial_occupancy + event.entered_total
            ):
                self._record(
                    "occupancy_boundary_clamped",
                    frame_number,
                    timestamp_seconds,
                    frame,
                    anonymous_track_ids=(event.tracking_id,),
                    details={
                        "entered_total": event.entered_total,
                        "exited_total": event.exited_total,
                        "reported_occupancy": event.current_occupancy,
                    },
                )

        self._previous_track_ids = current_ids

    def finalize(self) -> None:
        """Record end-of-run short tracks and incomplete transitions once."""

        if not self.config.enabled or self._finalized:
            return
        self._finalized = True
        for tracking_id, evidence in sorted(self._tracks.items()):
            if evidence.observed_frames <= self.config.short_track_maximum_frames:
                self._record(
                    "short_track",
                    evidence.last_frame,
                    None,
                    None,
                    anonymous_track_ids=(tracking_id,),
                    details={
                        "first_frame": evidence.first_frame,
                        "last_frame": evidence.last_frame,
                        "observed_frames": evidence.observed_frames,
                        "threshold_frames": (self.config.short_track_maximum_frames),
                    },
                )
        for tracking_id, snapshot in sorted(self._last_snapshots.items()):
            if snapshot.pending_direction is None:
                continue
            evidence = self._tracks.get(tracking_id)
            self._record(
                "incomplete_transition",
                evidence.last_frame if evidence is not None else -1,
                None,
                None,
                anonymous_track_ids=(tracking_id,),
                details={
                    "pending_direction": snapshot.pending_direction.value,
                    "stable_zone": (
                        snapshot.stable_zone.value
                        if snapshot.stable_zone is not None
                        else None
                    ),
                    "destination_confirmation_frames": (
                        snapshot.destination_confirmation_frames
                    ),
                },
            )

    @property
    def summary(self) -> dict[str, Any]:
        """Return auditable behavior counters, never accuracy claims."""

        return {
            "enabled": self.config.enabled,
            "metadata_only": not self.config.capture_frames,
            "output_jsonl": str(self.output_path) if self.config.enabled else None,
            "capture_frames": self.config.capture_frames,
            "frames_directory": (
                str(self.frames_directory) if self.config.capture_frames else None
            ),
            "maximum_records": self.config.maximum_records,
            "maximum_captured_frames": self.config.maximum_captured_frames,
            "records_written": self._records_written,
            "records_dropped_at_limit": self._records_dropped,
            "captured_frame_count": len(self._captured_frames),
            "record_types": dict(sorted(self._record_types.items())),
            "anonymous_id_scope": "single_video_only",
            "biometric_identity_used": False,
            "uploads_performed": False,
            "frames_may_contain_personal_data": self.config.capture_frames,
        }

    def _record(
        self,
        kind: str,
        frame_number: int,
        timestamp_seconds: float | None,
        frame: NDArray[np.uint8] | None,
        *,
        anonymous_track_ids: tuple[int, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._records_written >= self.config.maximum_records:
            self._records_dropped += 1
            return
        if self._file is None:
            raise FailureMiningError("Failure evidence output is not open.")
        captured_path = self._capture(frame_number, frame)
        record = {
            "schema_version": "1.0",
            "sequence": self._records_written + 1,
            "kind": kind,
            "frame": frame_number,
            "timestamp_seconds": (
                round(timestamp_seconds, 6) if timestamp_seconds is not None else None
            ),
            "anonymous_track_ids": list(anonymous_track_ids),
            "anonymous_id_scope": "single_video_only",
            "details": details or {},
            "captured_frame": captured_path,
        }
        self._file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._file.flush()
        self._records_written += 1
        self._record_types[kind] += 1

    def _capture(
        self, frame_number: int, frame: NDArray[np.uint8] | None
    ) -> str | None:
        if not self.config.capture_frames or frame is None:
            return None
        existing = self._captured_frames.get(frame_number)
        if existing is not None:
            return existing
        if len(self._captured_frames) >= self.config.maximum_captured_frames:
            return None
        path = self.frames_directory / f"failure-frame-{frame_number:08d}.jpg"
        ok = cv2.imwrite(
            str(path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not ok:
            raise FailureMiningError(f"Could not write requested failure frame: {path}")
        rendered = str(path)
        self._captured_frames[frame_number] = rendered
        return rendered


def _touches_edge(
    track: TrackedDetection,
    frame_shape: tuple[int, ...],
    margin: int,
) -> bool:
    height, width = frame_shape[:2]
    left, top, right, bottom = track.bounding_box
    return (
        left <= margin
        or top <= margin
        or right >= width - 1 - margin
        or bottom >= height - 1 - margin
    )
