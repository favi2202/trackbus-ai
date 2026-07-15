"""Ultralytics prediction backend with no persistent tracker state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection

LOGGER = logging.getLogger(__name__)


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


class UltralyticsDetector:
    """Run person-only ``YOLO.predict`` and return untracked detections."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        confidence: float,
        image_size: int,
        *,
        device: str | int,
        device_label: str,
        half: bool = False,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be greater than 0 and at most 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if "://" in model_path:
            raise DetectorError(
                "Remote model URIs are not accepted. Use standard Ultralytics weights "
                "or a trusted local model file."
            )
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.device_label = device_label
        self.half = bool(half and device_label.startswith("cuda"))
        if half and not self.half:
            LOGGER.warning(
                "Half precision was requested but is disabled on %s.", device_label
            )
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
        except Exception as exc:  # Ultralytics exposes several backend exceptions
            raise DetectorError(
                f"Could not load YOLO model '{model_path}': {exc}"
            ) from exc

    def detect(self, image: NDArray[np.uint8], *, source_view: str) -> list[Detection]:
        """Return class-0 predictions before any tracking."""

        prediction_options: dict[str, Any] = {
            "source": image,
            "classes": [self.PERSON_CLASS_ID],
            "conf": self.confidence,
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
        }
        # Ultralytics 8.4 warns if the deprecated false value is passed on every
        # frame. Only send the option when FP16 was safely enabled on CUDA.
        if self.half:
            prediction_options["half"] = True
        try:
            results = self._model.predict(**prediction_options)
        except Exception as exc:
            raise DetectorError(f"YOLO prediction failed: {exc}") from exc
        if not results:
            raise DetectorError("YOLO returned no result object for an inference view.")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        return [
            Detection(
                bounding_box=tuple(float(value) for value in coordinate),
                confidence=float(confidence),
                class_id=int(class_id),
                source_view=source_view,
                metadata={"coordinate_space": "view"},
            )
            for coordinate, confidence, class_id in zip(
                coordinates, confidences, classes, strict=True
            )
        ]


class PersonDetector(UltralyticsDetector):
    """Backward-compatible name; production uses the prediction-only API.

    ``infer_with_tracking`` remains solely so v0.1.2 adapter tests and external
    callers receive an explicit deprecated path. TrackBus v0.2 never invokes it
    and never falls back to it when the explicit ByteTrack adapter fails.
    """

    def __init__(self, model_path: str, confidence: float, image_size: int) -> None:
        super().__init__(
            model_path,
            confidence,
            image_size,
            device="cpu",
            device_label="cpu",
        )

    def infer_with_tracking(
        self,
        frame: NDArray[np.uint8],
        *,
        tracker: str,
        device: str | int,
    ) -> Any:
        """Invoke the old combined API only when explicitly called by a client."""

        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                imgsz=self.image_size,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO/ByteTrack inference failed: {exc}") from exc
        if not results:
            raise DetectorError("YOLO returned no result object for a video frame.")
        return results[0]


def is_local_model(model_path: str) -> bool:
    """Return whether a configured model resolves to an existing local file."""

    return Path(model_path).expanduser().is_file()
