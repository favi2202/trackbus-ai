"""Ultralytics prediction backend with no persistent tracker state."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection

LOGGER = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or used."""


def validate_class_zero_contract(
    model_names: object,
    expected_name: str,
) -> str:
    """Require the configured target to be class 0 before ByteTrack receives it."""

    expected = _normalized_class_name(expected_name)
    if not expected:
        raise DetectorError("Expected detector class name cannot be empty.")
    names: dict[int, str] = {}
    if isinstance(model_names, Mapping):
        for raw_id, raw_name in model_names.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_name, str):
                names[class_id] = raw_name
    elif isinstance(model_names, Sequence) and not isinstance(
        model_names, (str, bytes)
    ):
        names = {
            class_id: raw_name
            for class_id, raw_name in enumerate(model_names)
            if isinstance(raw_name, str)
        }
    actual = names.get(0)
    if actual is None:
        raise DetectorError(
            "YOLO model does not expose a text label for class 0; the detector "
            "target cannot be verified safely."
        )
    if _normalized_class_name(actual) != expected:
        available = ", ".join(
            f"{class_id}:{name}" for class_id, name in sorted(names.items())
        )
        raise DetectorError(
            f"YOLO class 0 is '{actual}', but this run requires "
            f"'{expected_name}'. Use a trained {expected_name} model whose only "
            f"target is class 0. Model labels: {available}."
        )
    return actual


def _normalized_class_name(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


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
    """Run one verified class-0 ``YOLO.predict`` target without tracker state."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str,
        confidence: float,
        image_size: int,
        *,
        device: str | int,
        device_label: str,
        detector_floor: float | None = None,
        half: bool = False,
        precision: str | None = None,
        target_class_name: str | None = None,
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
        self.detector_floor = confidence if detector_floor is None else detector_floor
        if not 0.0 < self.detector_floor <= confidence:
            raise ValueError(
                "detector_floor must be greater than 0 and at most confidence"
            )
        self.image_size = image_size
        self.device = device
        self.device_label = device_label
        requested_precision = precision or ("fp16" if half else "fp32")
        if requested_precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        self.half = bool(
            requested_precision == "fp16" and device_label.startswith("cuda")
        )
        self.precision = "fp16" if self.half else "fp32"
        if requested_precision == "fp16" and not self.half:
            LOGGER.warning(
                "FP16 was requested but is disabled on %s; using FP32.", device_label
            )
        # Ultralytics 8.4.95 deprecated the per-prediction ``half`` argument.
        # Its supported quantize=16 value selects FP16 inference; no INT8/export
        # quantization is enabled. Resolve this once instead of warning per frame.
        self._prediction_precision_options: dict[str, Any] = (
            {"quantize": 16} if self.half else {}
        )
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
        except Exception as exc:  # Ultralytics exposes several backend exceptions
            raise DetectorError(
                f"Could not load YOLO model '{model_path}': {exc}"
            ) from exc
        self.target_class_name = target_class_name or "person"
        if target_class_name is not None:
            validate_class_zero_contract(
                getattr(self._model, "names", None),
                target_class_name,
            )

    def detect(self, image: NDArray[np.uint8], *, source_view: str) -> list[Detection]:
        """Return verified class-0 predictions before any tracking."""

        prediction_options: dict[str, Any] = {
            "source": image,
            "classes": [self.PERSON_CLASS_ID],
            "conf": getattr(self, "detector_floor", self.confidence),
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
        }
        prediction_options.update(
            getattr(
                self,
                "_prediction_precision_options",
                {"quantize": 16} if getattr(self, "half", False) else {},
            )
        )
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
                metadata={
                    "coordinate_space": "view",
                    "detection_target": getattr(
                        self, "target_class_name", "person"
                    ),
                },
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
            target_class_name="person",
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
