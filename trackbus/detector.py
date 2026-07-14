"""Ultralytics YOLO model loading and person inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class DetectorError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or used."""


def resolve_device(requested: str) -> tuple[str | int, str]:
    """Resolve a friendly CLI device value to an Ultralytics device value."""

    normalized = requested.strip().lower()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise DetectorError(
            "PyTorch is not installed. Install the dependencies from requirements.txt."
        ) from exc

    if normalized == "auto":
        return (0, "cuda:0") if torch.cuda.is_available() else ("cpu", "cpu")
    if normalized == "cpu":
        return "cpu", "cpu"
    if normalized in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise DetectorError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU. "
                "Use '--device cpu' or install a CUDA-enabled PyTorch build."
            )
        return 0, "cuda:0"
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        index = int(normalized[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise DetectorError(f"CUDA device {index} is not available.")
        return index, f"cuda:{index}"
    if normalized.isdigit():
        index = int(normalized)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise DetectorError(f"CUDA device {index} is not available.")
        return index, f"cuda:{index}"
    raise DetectorError(
        "Invalid device. Use 'auto', 'cpu', 'cuda', 'cuda:N', or a GPU index."
    )


class PersonDetector:
    """Own a pretrained YOLO model and run person-only tracked inference."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, confidence: float) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be greater than 0 and at most 1")
        self.model_path = model_path
        self.confidence = confidence
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
        except Exception as exc:  # Ultralytics exposes several backend exceptions
            raise DetectorError(
                f"Could not load YOLO model '{model_path}': {exc}"
            ) from exc

    def infer_with_tracking(
        self,
        frame: NDArray[np.uint8],
        *,
        tracker: str,
        device: str | int,
    ) -> Any:
        """Run YOLO person detection and Ultralytics' ByteTrack integration."""

        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO/ByteTrack inference failed: {exc}") from exc
        if not results:
            raise DetectorError("YOLO returned no result object for a video frame.")
        return results[0]
