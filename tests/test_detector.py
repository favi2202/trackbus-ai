from types import SimpleNamespace

import numpy as np

from trackbus.detector import PersonDetector, UltralyticsDetector


class FakeModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}
        self.result = SimpleNamespace()

    def track(self, **kwargs: object) -> list[SimpleNamespace]:
        self.arguments = kwargs
        return [self.result]

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.arguments = kwargs
        return [SimpleNamespace(boxes=None)]


def test_inference_settings_are_forwarded_to_yolo() -> None:
    model = FakeModel()
    detector = object.__new__(PersonDetector)
    detector.confidence = 0.2
    detector.image_size = 960
    detector._model = model

    result = detector.infer_with_tracking(
        np.zeros((24, 32, 3), dtype=np.uint8),
        tracker="configs/bytetrack_trackbus.yaml",
        device="cpu",
    )

    assert result is model.result
    assert model.arguments["classes"] == [PersonDetector.PERSON_CLASS_ID]
    assert model.arguments["conf"] == 0.2
    assert model.arguments["imgsz"] == 960
    assert model.arguments["tracker"] == "configs/bytetrack_trackbus.yaml"
    assert model.arguments["device"] == "cpu"


def test_detector_floor_is_forwarded_without_changing_confidence_reference() -> None:
    model = FakeModel()
    detector = object.__new__(UltralyticsDetector)
    detector.confidence = 0.35
    detector.detector_floor = 0.10
    detector.image_size = 640
    detector.device = "cpu"
    detector.half = False
    detector._prediction_precision_options = {}
    detector._model = model

    assert (
        detector.detect(np.zeros((24, 32, 3), dtype=np.uint8), source_view="full") == []
    )

    assert model.arguments["conf"] == 0.10
    assert detector.confidence == 0.35
