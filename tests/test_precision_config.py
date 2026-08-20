from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trackbus.config import ConfigError, load_config
from trackbus.detector import DetectorError, UltralyticsDetector, resolve_device


def _config_yaml(model_precision: str = "") -> str:
    model_setting = f"  {model_precision}\n" if model_precision else ""
    return f"""
model:
  path: yolo11n.pt
  confidence: 0.35
  imgsz: 640
{model_setting}zones:
  outside: [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]
  inside: [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]
outputs:
  video: result.mp4
"""


def _write_config(tmp_path: Path, model_precision: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(_config_yaml(model_precision), encoding="utf-8")
    return path


def test_model_precision_defaults_to_fp32_without_a_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(_write_config(tmp_path))

    assert config.model.precision == "fp32"
    assert config.model.half is False
    assert config.model.legacy_half_configured is False
    assert caught == []


def test_canonical_fp16_configuration_loads_without_a_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(_write_config(tmp_path, "precision: fp16"))

    assert config.model.precision == "fp16"
    assert config.model.half is True
    assert config.model.legacy_half_configured is False
    assert caught == []


@pytest.mark.parametrize(
    ("legacy_value", "expected"), [(True, "fp16"), (False, "fp32")]
)
def test_legacy_half_maps_to_precision_with_one_startup_warning(
    tmp_path: Path, legacy_value: bool, expected: str
) -> None:
    rendered_value = str(legacy_value).lower()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(_write_config(tmp_path, f"half: {rendered_value}"))

    deprecations = [
        item for item in caught if "model.half is deprecated" in str(item.message)
    ]
    assert config.model.precision == expected
    assert config.model.half is legacy_value
    assert config.model.legacy_half_configured is True
    assert len(deprecations) == 1
    assert len(caught) == 1


def test_precision_and_legacy_half_cannot_be_combined(tmp_path: Path) -> None:
    setting = "precision: fp16\n  half: true"

    with pytest.raises(ConfigError, match="cannot be combined"):
        load_config(_write_config(tmp_path, setting))


def test_invalid_precision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be fp32 or fp16"):
        load_config(_write_config(tmp_path, "precision: bf16"))


class _FakePredictModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=None)]


def _install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakePredictModel:
    model = _FakePredictModel()
    fake_module = SimpleNamespace(YOLO=lambda _path: model)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    return model


def _make_detector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str | int,
    device_label: str,
    precision: str,
) -> tuple[UltralyticsDetector, _FakePredictModel]:
    model = _install_fake_ultralytics(monkeypatch)
    detector = UltralyticsDetector(
        "yolo11n.pt",
        0.35,
        640,
        device=device,
        device_label=device_label,
        precision=precision,
    )
    return detector, model


def _detect_twice(detector: UltralyticsDetector) -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    assert detector.detect(frame, source_view="full") == []
    assert detector.detect(frame, source_view="full") == []


def test_fp32_predictions_send_neither_half_nor_quantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, model = _make_detector(
        monkeypatch, device="cpu", device_label="cpu", precision="fp32"
    )

    _detect_twice(detector)

    assert detector.precision == "fp32"
    assert detector.half is False
    assert len(model.calls) == 2
    assert all("half" not in call and "quantize" not in call for call in model.calls)


def test_cuda_fp16_predictions_send_quantize_once_per_call_and_never_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, model = _make_detector(
        monkeypatch, device=0, device_label="cuda:0", precision="fp16"
    )

    _detect_twice(detector)

    assert detector.precision == "fp16"
    assert detector.half is True
    assert len(model.calls) == 2
    assert all(call.get("quantize") == 16 for call in model.calls)
    assert all("half" not in call for call in model.calls)


def test_cpu_fp16_downgrades_and_logs_only_during_initialization(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="trackbus.detector"):
        detector, model = _make_detector(
            monkeypatch, device="cpu", device_label="cpu", precision="fp16"
        )
        initialization_records = tuple(caplog.records)
        _detect_twice(detector)

    fp16_warnings = [
        record for record in caplog.records if "FP16 was requested" in record.message
    ]
    assert detector.precision == "fp32"
    assert detector.half is False
    assert len(initialization_records) == 1
    assert len(fp16_warnings) == 1
    assert len(caplog.records) == 1
    assert len(model.calls) == 2
    assert all("half" not in call and "quantize" not in call for call in model.calls)


class _FakeCuda:
    def __init__(self, *, available: bool, count: int = 0) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


def test_resolve_device_parses_cpu_and_available_cuda_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=_FakeCuda(available=True, count=2)),
    )

    assert resolve_device(" auto ") == (0, "cuda:0")
    assert resolve_device("cpu") == ("cpu", "cpu")
    assert resolve_device("CUDA") == (0, "cuda:0")
    assert resolve_device("cuda:1") == (1, "cuda:1")
    assert resolve_device("1") == (1, "cuda:1")


def test_resolve_device_falls_back_for_auto_but_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=_FakeCuda(available=False)),
    )

    assert resolve_device("auto") == ("cpu", "cpu")
    with pytest.raises(DetectorError, match="CUDA was requested"):
        resolve_device("cuda")
    with pytest.raises(DetectorError, match="CUDA device 1 is not available"):
        resolve_device("cuda:1")


def test_resolve_device_rejects_invalid_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=_FakeCuda(available=True, count=1)),
    )

    with pytest.raises(DetectorError, match="Invalid device"):
        resolve_device("gpu")
