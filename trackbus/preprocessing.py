"""Deterministic, same-size preprocessing experiments for TrackBus vision."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from trackbus.detection import Detection
from trackbus.interfaces import DetectorBackend


class PreprocessingProfile(StrEnum):
    """Bounded image transformations that never change frame geometry."""

    NONE = "none"
    LOW_LIGHT = "low_light"
    CONTRAST = "contrast"


class PreprocessingError(ValueError):
    """Raised when preprocessing input or configuration is invalid."""


@dataclass(frozen=True)
class PreprocessingSettings:
    """Auditable constants for one preprocessing profile."""

    profile: PreprocessingProfile
    gamma: float | None
    clahe_clip_limit: float | None
    clahe_tile_grid_size: tuple[int, int] | None
    sharpening_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.value,
            "gamma": self.gamma,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": (
                list(self.clahe_tile_grid_size)
                if self.clahe_tile_grid_size is not None
                else None
            ),
            "sharpening_enabled": self.sharpening_enabled,
            "preserves_dimensions": True,
        }


SETTINGS = {
    PreprocessingProfile.NONE: PreprocessingSettings(
        PreprocessingProfile.NONE,
        gamma=None,
        clahe_clip_limit=None,
        clahe_tile_grid_size=None,
    ),
    PreprocessingProfile.LOW_LIGHT: PreprocessingSettings(
        PreprocessingProfile.LOW_LIGHT,
        gamma=0.75,
        clahe_clip_limit=1.8,
        clahe_tile_grid_size=(8, 8),
    ),
    PreprocessingProfile.CONTRAST: PreprocessingSettings(
        PreprocessingProfile.CONTRAST,
        gamma=None,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
    ),
}


def parse_preprocessing_profile(
    value: str | PreprocessingProfile,
) -> PreprocessingProfile:
    """Parse a stable profile name with a useful error message."""

    try:
        return PreprocessingProfile(value)
    except ValueError as exc:
        choices = ", ".join(profile.value for profile in PreprocessingProfile)
        raise PreprocessingError(
            f"preprocessing profile must be one of: {choices}"
        ) from exc


def preprocess_frame(
    frame: NDArray[np.uint8],
    profile: str | PreprocessingProfile,
) -> NDArray[np.uint8]:
    """Return a deterministic BGR frame with the original shape and dtype."""

    parsed = parse_preprocessing_profile(profile)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise PreprocessingError("preprocessing requires a non-empty BGR frame")
    if frame.dtype != np.uint8:
        raise PreprocessingError("preprocessing requires uint8 image data")
    if parsed is PreprocessingProfile.NONE:
        return frame

    settings = SETTINGS[parsed]
    enhanced = _apply_luminance_clahe(frame, settings)
    if settings.gamma is not None:
        enhanced = cv2.LUT(enhanced, _gamma_table(settings.gamma))
    if enhanced.shape != frame.shape or enhanced.dtype != frame.dtype:
        raise PreprocessingError("preprocessing changed frame geometry or dtype")
    return enhanced


class PreprocessingDetector:
    """Apply one opt-in profile before an existing detector backend."""

    def __init__(
        self,
        detector: DetectorBackend,
        profile: str | PreprocessingProfile = PreprocessingProfile.NONE,
    ) -> None:
        self.detector = detector
        self.profile = parse_preprocessing_profile(profile)
        self.settings = SETTINGS[self.profile]
        self.calls = 0
        self.total_preprocessing_seconds = 0.0
        self.last_preprocessing_seconds = 0.0

    @property
    def device_label(self) -> str:
        return str(getattr(self.detector, "device_label", "unknown"))

    @property
    def precision(self) -> str:
        return str(getattr(self.detector, "precision", "unknown"))

    def detect(self, image: NDArray[np.uint8], *, source_view: str) -> list[Detection]:
        started = time.perf_counter()
        prepared = preprocess_frame(image, self.profile)
        self.last_preprocessing_seconds = time.perf_counter() - started
        self.total_preprocessing_seconds += self.last_preprocessing_seconds
        self.calls += 1
        return self.detector.detect(prepared, source_view=source_view)

    @property
    def summary(self) -> dict[str, object]:
        mean_ms = (
            self.total_preprocessing_seconds / self.calls * 1000.0
            if self.calls
            else 0.0
        )
        return {
            **self.settings.to_dict(),
            "calls": self.calls,
            "total_preprocessing_seconds": round(self.total_preprocessing_seconds, 6),
            "mean_preprocessing_latency_ms": round(mean_ms, 6),
            "enabled": self.profile is not PreprocessingProfile.NONE,
            "experimental": True,
        }


def preprocessing_latency_ms(detector: DetectorBackend) -> float:
    """Read the latest preprocessing overhead without coupling benchmarks."""

    seconds = getattr(detector, "last_preprocessing_seconds", 0.0)
    return max(0.0, float(seconds)) * 1000.0


def preprocessing_summary(detector: DetectorBackend) -> dict[str, object]:
    """Return an honest default for detectors without a preprocessing wrapper."""

    summary = getattr(detector, "summary", None)
    if isinstance(summary, dict) and "profile" in summary:
        return dict(summary)
    return {
        **SETTINGS[PreprocessingProfile.NONE].to_dict(),
        "calls": 0,
        "total_preprocessing_seconds": 0.0,
        "mean_preprocessing_latency_ms": 0.0,
        "enabled": False,
        "experimental": True,
    }


def _apply_luminance_clahe(
    frame: NDArray[np.uint8], settings: PreprocessingSettings
) -> NDArray[np.uint8]:
    if settings.clahe_clip_limit is None or settings.clahe_tile_grid_size is None:
        return frame.copy()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=settings.clahe_clip_limit,
        tileGridSize=settings.clahe_tile_grid_size,
    )
    adjusted = clahe.apply(luminance)
    return cv2.cvtColor(
        cv2.merge((adjusted, green_red, blue_yellow)),
        cv2.COLOR_LAB2BGR,
    )


def _gamma_table(gamma: float) -> NDArray[np.uint8]:
    values = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(values, gamma) * 255.0, 0, 255).astype(np.uint8)
