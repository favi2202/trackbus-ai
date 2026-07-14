"""CSV event logging and JSON run-summary output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import IO, Any

from trackbus.counter import CountEvent

CSV_FIELDS = (
    "timestamp",
    "video_frame",
    "tracking_id",
    "event_type",
    "entered_total",
    "exited_total",
    "current_occupancy",
    "occupancy_percentage",
)


def derive_artifact_paths(
    output_video: Path,
    events_csv: Path | None,
    summary_json: Path | None,
) -> tuple[Path, Path]:
    """Resolve optional log paths beside the annotated output video."""

    csv_path = events_csv or output_video.with_name(f"{output_video.stem}.events.csv")
    json_path = summary_json or output_video.with_name(
        f"{output_video.stem}.summary.json"
    )
    return csv_path, json_path


class EventLogger:
    """Write crossing events incrementally and a final machine-readable summary."""

    def __init__(self, csv_path: Path, summary_path: Path) -> None:
        self.csv_path = csv_path
        self.summary_path = summary_path
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> EventLogger:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        return self

    def __exit__(self, *_args: object) -> None:
        if self._file is not None:
            self._file.close()

    def log_event(self, event: CountEvent, video_fps: float) -> None:
        """Append one confirmed crossing event to CSV."""

        if self._writer is None:
            raise RuntimeError("EventLogger must be opened as a context manager.")
        self._writer.writerow(
            {
                "timestamp": _format_video_timestamp(event.video_frame / video_fps),
                "video_frame": event.video_frame,
                "tracking_id": event.tracking_id,
                "event_type": event.event_type.value,
                "entered_total": event.entered_total,
                "exited_total": event.exited_total,
                "current_occupancy": event.current_occupancy,
                "occupancy_percentage": f"{event.occupancy_percentage:.2f}",
            }
        )
        if self._file is not None:
            self._file.flush()

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write the final run summary as indented JSON."""

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _format_video_timestamp(seconds: float) -> str:
    milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
