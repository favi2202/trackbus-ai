from __future__ import annotations

import cv2
import numpy as np
import pytest

from trackbus.detection import Detection
from trackbus.preprocessing import (
    PreprocessingDetector,
    PreprocessingError,
    PreprocessingProfile,
    preprocess_frame,
)


class RecordingDetector:
    device_label = "cpu"
    precision = "fp32"

    def __init__(self) -> None:
        self.image: np.ndarray | None = None

    def detect(self, image: np.ndarray, *, source_view: str) -> list[Detection]:
        self.image = image.copy()
        return [Detection((1, 1, 4, 5), 0.7, 0, source_view)]


def synthetic_frames() -> dict[str, np.ndarray]:
    checkerboard = np.indices((64, 96)).sum(axis=0) % 2
    textured = np.repeat((checkerboard * 180 + 30)[:, :, None], 3, axis=2).astype(
        np.uint8
    )
    return {
        "dark": np.full((64, 96, 3), 18, dtype=np.uint8),
        "bright": np.full((64, 96, 3), 238, dtype=np.uint8),
        "blurred": cv2.GaussianBlur(textured, (15, 15), 0),
    }


@pytest.mark.parametrize("profile", list(PreprocessingProfile))
@pytest.mark.parametrize("frame_name", ("dark", "bright", "blurred"))
def test_preprocessing_is_deterministic_and_preserves_geometry(
    profile: PreprocessingProfile,
    frame_name: str,
) -> None:
    source = synthetic_frames()[frame_name]

    first = preprocess_frame(source, profile)
    second = preprocess_frame(source, profile)

    assert np.array_equal(first, second)
    assert first.shape == source.shape
    assert first.dtype == source.dtype == np.uint8


def test_none_is_disabled_and_low_light_brightens_dark_input() -> None:
    dark = synthetic_frames()["dark"]

    assert preprocess_frame(dark, "none") is dark
    enhanced = preprocess_frame(dark, "low_light")

    assert float(enhanced.mean()) > float(dark.mean())
    assert int(enhanced.max()) < 255


def test_profiles_do_not_include_sharpening_or_artificial_resize() -> None:
    detector = RecordingDetector()
    wrapped = PreprocessingDetector(detector, "contrast")
    source = synthetic_frames()["blurred"]

    detections = wrapped.detect(source, source_view="full")

    assert len(detections) == 1
    assert detector.image is not None
    assert detector.image.shape == source.shape
    assert wrapped.summary["sharpening_enabled"] is False
    assert wrapped.summary["preserves_dimensions"] is True
    assert wrapped.summary["calls"] == 1
    assert wrapped.summary["mean_preprocessing_latency_ms"] >= 0


def test_invalid_profile_and_non_uint8_input_are_rejected() -> None:
    with pytest.raises(PreprocessingError, match="must be one of"):
        preprocess_frame(synthetic_frames()["dark"], "magic")
    with pytest.raises(PreprocessingError, match="uint8"):
        preprocess_frame(np.zeros((10, 10, 3), dtype=np.float32), "contrast")
