"""Extract local video frames for manual, anonymous head-box annotation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2


class HeadFrameExtractionError(ValueError):
    """Raised when extraction would overwrite data or produce weak evidence."""


def extract_head_frames(
    video_path: Path,
    output_dir: Path,
    *,
    every: int = 3,
    start_frame: int = 0,
    end_frame: int | None = None,
    maximum_frames: int | None = None,
    jpeg_quality: int = 95,
) -> dict[str, object]:
    """Write sampled JPEGs and a manifest; never infer or fabricate labels."""

    if every < 1:
        raise HeadFrameExtractionError("every must be at least 1")
    if start_frame < 0:
        raise HeadFrameExtractionError("start_frame cannot be negative")
    if end_frame is not None and end_frame < start_frame:
        raise HeadFrameExtractionError("end_frame cannot be before start_frame")
    if maximum_frames is not None and maximum_frames < 1:
        raise HeadFrameExtractionError("maximum_frames must be at least 1")
    if not 1 <= jpeg_quality <= 100:
        raise HeadFrameExtractionError("jpeg_quality must be between 1 and 100")

    video = video_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if not video.is_file():
        raise HeadFrameExtractionError(f"Input video does not exist: {video}")
    if output.exists():
        if not output.is_dir():
            raise HeadFrameExtractionError(f"Output is not a directory: {output}")
        if any(output.iterdir()):
            raise HeadFrameExtractionError(
                f"Output directory is not empty; refusing to overwrite: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise HeadFrameExtractionError(f"Could not open input video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected: list[dict[str, object]] = []
    decoded = 0
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if end_frame is not None and frame_number > end_frame:
                break
            should_write = (
                frame_number >= start_frame
                and (frame_number - start_frame) % every == 0
            )
            if should_write:
                filename = f"{video.stem}-f{frame_number:06d}.jpg"
                destination = output / filename
                written = cv2.imwrite(
                    str(destination),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not written:
                    raise HeadFrameExtractionError(
                        f"OpenCV could not write extracted frame: {destination}"
                    )
                selected.append(
                    {
                        "frame": frame_number,
                        "timestamp_seconds": (
                            round(frame_number / source_fps, 6)
                            if source_fps
                            else None
                        ),
                        "image": filename,
                    }
                )
                if maximum_frames is not None and len(selected) >= maximum_frames:
                    break
            frame_number += 1
    finally:
        capture.release()

    if not selected:
        raise HeadFrameExtractionError(
            "No frames matched the requested extraction range."
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "workflow": "trackbus_head_frame_extraction",
        "detection_target": "head",
        "source": {
            "path": str(video),
            "width": width,
            "height": height,
            "fps": round(source_fps, 6) if source_fps else None,
            "reported_frame_count": reported_frames,
            "decoded_frame_count": decoded,
        },
        "sampling": {
            "every": every,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "maximum_frames": maximum_frames,
            "jpeg_quality": jpeg_quality,
        },
        "selected_frame_count": len(selected),
        "frames": selected,
        "annotation_contract": {
            "classes": ["head"],
            "fabricated_labels": False,
            "manual_review_required": True,
            "identity_or_face_recognition_allowed": False,
        },
        "privacy": {
            "uploads_performed": False,
            "local_processing_only": True,
            "persistent_passenger_identity_allowed": False,
        },
        "accuracy_claimed": False,
    }
    (output / "extraction.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.extract_head_frames",
        description=(
            "Sample local video frames for manual one-class head annotation. "
            "No labels are generated and no files are uploaded."
        ),
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = extract_head_frames(
            args.video,
            args.output,
            every=args.every,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            maximum_frames=args.max_frames,
            jpeg_quality=args.jpeg_quality,
        )
    except (HeadFrameExtractionError, OSError, ValueError) as exc:
        print(f"TrackBus head extraction error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Extracted {result['selected_frame_count']} local frames to "
        f"{args.output.expanduser().resolve()}"
    )
    print(
        "No labels were fabricated, no files were uploaded, and no identity was used."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
