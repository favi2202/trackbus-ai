from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trackbus.config import (
    CameraConfig,
    ConfigError,
    InferenceViewConfig,
    configuration_warnings,
    load_config,
    resolve_inference_views,
)
from trackbus.detection import Detection
from trackbus.detector import DetectorError, UltralyticsDetector
from trackbus.fusion import NmsDetectionFusion, box_iou
from trackbus.interfaces import DetectorBackend
from trackbus.views import MultiViewInference, PixelInferenceView


class FakeDetector:
    def __init__(
        self,
        boxes_by_view: dict[str, tuple[tuple[float, float, float, float], ...]],
    ) -> None:
        self.boxes_by_view = boxes_by_view
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def detect(self, image: np.ndarray, *, source_view: str) -> list[Detection]:
        self.calls.append((source_view, image.shape))
        return [
            Detection(
                bounding_box=box,
                confidence=0.8,
                class_id=0,
                source_view=source_view,
                metadata={"coordinate_space": "view"},
            )
            for box in self.boxes_by_view.get(source_view, ())
        ]


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> object:
        return self.values


class FakeYoloBoxes:
    def __init__(
        self,
        coordinates: list[list[float]],
        confidences: list[float],
        classes: list[float],
    ) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(classes)
        self._length = len(coordinates)

    def __len__(self) -> int:
        return self._length


class FakePredictModel:
    def __init__(self, boxes: FakeYoloBoxes | None) -> None:
        self.arguments: dict[str, object] = {}
        self.boxes = boxes

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.arguments = kwargs
        return [SimpleNamespace(boxes=self.boxes)]


def make_detection(
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.8,
    class_id: int = 0,
    view: str = "full",
) -> Detection:
    return Detection(
        bounding_box=box,
        confidence=confidence,
        class_id=class_id,
        source_view=view,
    )


def test_fake_detector_satisfies_detector_backend_and_receives_crop() -> None:
    detector: DetectorBackend = FakeDetector({"door": ((1.0, 2.0, 5.0, 8.0),)})
    view = PixelInferenceView.from_config(
        InferenceViewConfig("door", (0.1, 0.2, 0.6, 0.8)),
        frame_width=200,
        frame_height=100,
    )

    detections = MultiViewInference(detector, (view,)).collect(
        np.zeros((100, 200, 3), dtype=np.uint8)
    )

    assert detector.calls == [("door", (60, 100, 3))]
    assert detections[0].bounding_box == (21.0, 22.0, 25.0, 28.0)
    assert detections[0].metadata["coordinate_space"] == "source"


def test_full_frame_crop_reuses_frame_and_partial_crop_has_resolved_shape() -> None:
    frame = np.zeros((101, 203, 3), dtype=np.uint8)
    full = PixelInferenceView.from_config(
        InferenceViewConfig("full", (0.0, 0.0, 1.0, 1.0)), 203, 101
    )
    partial = PixelInferenceView.from_config(
        InferenceViewConfig("partial", (0.1, 0.2, 0.6, 0.8)), 203, 101
    )

    assert full.crop(frame) is frame
    assert partial.crop(frame).shape == (61, 102, 3)
    assert (partial.left, partial.top, partial.right, partial.bottom) == (
        20,
        20,
        122,
        81,
    )


def test_translation_offsets_and_clips_to_source_frame_boundaries() -> None:
    view = PixelInferenceView.from_config(
        InferenceViewConfig("side", (0.1, 0.2, 0.6, 0.8)), 200, 100
    )
    local = make_detection((-30.0, -25.0, 500.0, 500.0), view="side")

    translated = view.translate(local, frame_width=200, frame_height=100)

    assert translated.bounding_box == (0.0, 0.0, 199.0, 99.0)
    assert translated.source_view == "side"
    assert translated.metadata == {
        "coordinate_space": "source",
        "view_bounds": (0.1, 0.2, 0.6, 0.8),
    }


def test_translation_discards_box_collapsed_outside_source_frame() -> None:
    view = PixelInferenceView.from_config(
        InferenceViewConfig("edge", (0.5, 0.0, 1.0, 1.0)),
        frame_width=200,
        frame_height=100,
    )
    local = make_detection((150.0, 10.0, 180.0, 30.0), view="edge")

    assert view.translate(local, frame_width=200, frame_height=100) is None


def test_multiview_collection_translates_every_enabled_view_once() -> None:
    detector = FakeDetector(
        {
            "full": ((10.0, 12.0, 30.0, 40.0),),
            "right": ((2.0, 3.0, 12.0, 20.0),),
        }
    )
    inference = MultiViewInference.from_configs(
        detector,
        (
            InferenceViewConfig("full", (0.0, 0.0, 1.0, 1.0)),
            InferenceViewConfig("disabled", (0.0, 0.0, 0.5, 1.0), False),
            InferenceViewConfig("right", (0.5, 0.1, 1.0, 1.0)),
        ),
        frame_width=200,
        frame_height=100,
    )

    detections = inference.collect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert detector.calls == [("full", (100, 200, 3)), ("right", (90, 100, 3))]
    assert [detection.source_view for detection in detections] == ["full", "right"]
    assert [detection.bounding_box for detection in detections] == [
        (10.0, 12.0, 30.0, 40.0),
        (102.0, 13.0, 112.0, 30.0),
    ]


def test_multiview_collection_handles_empty_detection_frames() -> None:
    detector = FakeDetector({})
    inference = MultiViewInference.from_configs(
        detector,
        (
            InferenceViewConfig("left", (0.0, 0.0, 0.6, 1.0)),
            InferenceViewConfig("right", (0.4, 0.0, 1.0, 1.0)),
        ),
        frame_width=20,
        frame_height=10,
    )

    assert inference.collect(np.zeros((10, 20, 3), dtype=np.uint8)) == []
    assert [call[0] for call in detector.calls] == ["left", "right"]


def test_multiview_requires_an_enabled_view() -> None:
    with pytest.raises(ValueError, match="at least one enabled inference view"):
        MultiViewInference.from_configs(
            FakeDetector({}),
            (InferenceViewConfig("off", (0.0, 0.0, 1.0, 1.0), False),),
            frame_width=20,
            frame_height=10,
        )


def test_ultralytics_predict_forwards_person_only_settings_and_preserves_values() -> (
    None
):
    model = FakePredictModel(
        FakeYoloBoxes(
            [[1.0, 2.0, 11.0, 22.0], [20.0, 5.0, 30.0, 25.0]],
            [0.91, 0.63],
            [0.0, 0.0],
        )
    )
    detector = object.__new__(UltralyticsDetector)
    detector._model = model
    detector.confidence = 0.25
    detector.image_size = 640
    detector.device = "cpu"
    detector.device_label = "cpu"
    detector.half = False
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    detections = detector.detect(image, source_view="left")

    assert model.arguments == {
        "source": image,
        "classes": [UltralyticsDetector.PERSON_CLASS_ID],
        "conf": 0.25,
        "imgsz": 640,
        "device": "cpu",
        "verbose": False,
    }
    assert [detection.confidence for detection in detections] == [0.91, 0.63]
    assert [detection.class_id for detection in detections] == [0, 0]
    assert all(detection.source_view == "left" for detection in detections)
    assert all(
        detection.metadata == {"coordinate_space": "view"} for detection in detections
    )


@pytest.mark.parametrize("boxes", [None, FakeYoloBoxes([], [], [])])
def test_ultralytics_predict_handles_empty_boxes(boxes: FakeYoloBoxes | None) -> None:
    detector = object.__new__(UltralyticsDetector)
    detector._model = FakePredictModel(boxes)
    detector.confidence = 0.35
    detector.image_size = 640
    detector.device = 0
    detector.device_label = "cuda:0"
    detector.half = True

    assert (
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), source_view="full") == []
    )


def test_ultralytics_predict_wraps_backend_errors() -> None:
    class FailingModel:
        def predict(self, **_kwargs: object) -> list[object]:
            raise RuntimeError("backend unavailable")

    detector = object.__new__(UltralyticsDetector)
    detector._model = FailingModel()
    detector.confidence = 0.35
    detector.image_size = 640
    detector.device = "cpu"
    detector.device_label = "cpu"
    detector.half = False

    with pytest.raises(DetectorError, match="YOLO prediction failed"):
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), source_view="full")


def test_nms_merges_identical_boxes_and_records_all_views() -> None:
    fusion = NmsDetectionFusion(iou_threshold=0.5)

    fused = fusion.fuse(
        [
            make_detection((10.0, 10.0, 30.0, 40.0), confidence=0.7, view="full"),
            make_detection((10.0, 10.0, 30.0, 40.0), confidence=0.9, view="left"),
        ]
    )

    assert len(fused) == 1
    assert fused[0].confidence == 0.9
    assert set(fused[0].source_views) == {"full", "left"}
    assert fused[0].metadata["suppressed_count"] == 1
    assert fused[0].metadata["source_detection_count"] == 2


def test_nms_merges_slightly_shifted_duplicates() -> None:
    fused = NmsDetectionFusion(iou_threshold=0.5).fuse(
        [
            make_detection((10.0, 10.0, 30.0, 40.0), view="full"),
            make_detection((12.0, 11.0, 32.0, 41.0), confidence=0.75, view="right"),
        ]
    )

    assert len(fused) == 1
    assert fused[0].bounding_box == (10.0, 10.0, 30.0, 40.0)
    assert set(fused[0].source_views) == {"full", "right"}


def test_nms_preserves_nearby_distinct_people_and_is_class_aware() -> None:
    distinct = NmsDetectionFusion(iou_threshold=0.5).fuse(
        [
            make_detection((0.0, 0.0, 10.0, 10.0), view="left"),
            make_detection((5.0, 0.0, 15.0, 10.0), view="right"),
        ]
    )
    different_classes = NmsDetectionFusion(iou_threshold=0.5).fuse(
        [
            make_detection((0.0, 0.0, 10.0, 10.0), class_id=0),
            make_detection((0.0, 0.0, 10.0, 10.0), class_id=1),
        ]
    )

    assert len(distinct) == 2
    assert box_iou(distinct[0].bounding_box, distinct[1].bounding_box) == pytest.approx(
        1 / 3
    )
    assert len(different_classes) == 2


def test_nms_large_box_does_not_erase_two_higher_confidence_people() -> None:
    large = make_detection((0.0, 0.0, 20.0, 10.0), confidence=0.6, view="full")
    left_person = make_detection((0.0, 0.0, 10.0, 10.0), confidence=0.95, view="left")
    right_person = make_detection((10.0, 0.0, 20.0, 10.0), confidence=0.9, view="right")

    fused = NmsDetectionFusion(iou_threshold=0.5).fuse(
        [large, left_person, right_person]
    )

    assert [detection.bounding_box for detection in fused] == [
        left_person.bounding_box,
        right_person.bounding_box,
    ]


def test_nms_large_high_confidence_box_does_not_erase_two_people() -> None:
    large = make_detection((0.0, 0.0, 20.0, 10.0), confidence=0.99, view="full")
    left_person = make_detection((0.0, 0.0, 10.0, 10.0), confidence=0.8, view="left")
    right_person = make_detection(
        (10.0, 0.0, 20.0, 10.0), confidence=0.75, view="right"
    )

    fused = NmsDetectionFusion(iou_threshold=0.5).fuse(
        [large, left_person, right_person]
    )

    assert [detection.bounding_box for detection in fused] == [
        left_person.bounding_box,
        right_person.bounding_box,
    ]
    assert all(
        detection.metadata["ambiguous_spanning_suppressed_count"] == 1
        for detection in fused
    )


def test_nms_never_suppresses_detections_from_the_same_view() -> None:
    first = make_detection((0.0, 0.0, 10.0, 10.0), confidence=0.9, view="full")
    second = make_detection((1.0, 0.0, 11.0, 10.0), confidence=0.8, view="full")

    fused = NmsDetectionFusion(iou_threshold=0.5).fuse([first, second])

    assert [detection.bounding_box for detection in fused] == [
        first.bounding_box,
        second.bounding_box,
    ]
    assert all(detection.metadata["suppressed_count"] == 0 for detection in fused)


def test_nms_rejects_zero_iou_threshold() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        NmsDetectionFusion(iou_threshold=0.0)


def test_prefer_full_frame_preserves_maximum_confidence() -> None:
    full = make_detection((0.0, 0.0, 10.0, 10.0), confidence=0.6, view="full")
    full = full.with_box(
        full.bounding_box,
        view_bounds=(0.0, 0.0, 1.0, 1.0),
    )
    side = make_detection((0.0, 0.0, 10.0, 10.0), confidence=0.9, view="left")

    fused = NmsDetectionFusion(
        iou_threshold=0.5,
        prefer_full_frame=True,
    ).fuse([full, side])

    assert fused[0].source_view == "full"
    assert fused[0].confidence == 0.9


def test_nms_handles_boundary_duplicates_and_empty_input() -> None:
    fusion = NmsDetectionFusion(iou_threshold=0.5)

    assert fusion.fuse([]) == []
    fused = fusion.fuse(
        [
            make_detection((0.0, 0.0, 12.0, 12.0), view="full"),
            make_detection((0.0, 0.5, 11.5, 12.0), view="left"),
        ]
    )
    assert len(fused) == 1
    assert set(fused[0].source_views) == {"full", "left"}


BASE_CONFIG = """
model:
  path: yolo11n.pt
tracking:
  minimum_zone_frames: 2
  stale_track_timeout: 30
zones:
  outside: [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]
  inside: [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]
outputs:
  video: result.mp4
"""


def write_config(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "multiview.yaml"
    path.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return path


def test_default_inference_view_is_full_frame() -> None:
    assert resolve_inference_views(CameraConfig())[0] == InferenceViewConfig(
        "full", (0.0, 0.0, 1.0, 1.0), True
    )


def test_all_documented_presets_and_test_video_config_load() -> None:
    repository = Path(__file__).resolve().parents[1]
    names = (
        "baseline_full_frame_cpu.yaml",
        "sensitive_full_frame_cpu.yaml",
        "balanced_full_frame.yaml",
        "multiview_nano.yaml",
        "multiview_small.yaml",
    )

    configs = [load_config(repository / "configs" / "presets" / name) for name in names]
    test_video = load_config(repository / "configs" / "test_video.yaml")

    assert [config.model.path for config in configs] == [
        "yolo11n.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]
    assert len(resolve_inference_views(configs[-1].camera)) == 3
    assert [view.enabled for view in test_video.camera.inference_views] == [
        True,
        False,
        False,
    ]


def test_exclusion_pathway_overlap_warns_without_collinear_false_positive(
    tmp_path: Path,
) -> None:
    overlapping = load_config(
        write_config(
            tmp_path,
            """
camera:
  doorway_lanes:
    left_lane: [[0.1, 0.45], [0.3, 0.45], [0.3, 0.55], [0.1, 0.55]]
  exclusion_polygons:
    - [[0.2, 0.46], [0.25, 0.46], [0.25, 0.54], [0.2, 0.54]]
""",
        )
    )
    separated = CameraConfig(
        left_lane=((0.1, 0.45), (0.2, 0.45), (0.2, 0.55), (0.1, 0.55)),
        exclusion_polygons=(((0.8, 0.45), (0.9, 0.45), (0.9, 0.55), (0.8, 0.55)),),
    )

    assert any(
        "passenger pathway" in item for item in configuration_warnings(overlapping)
    )
    separated_config = overlapping.__class__(
        model=overlapping.model,
        tracking=overlapping.tracking,
        zones=overlapping.zones,
        outputs=overlapping.outputs,
        camera=separated,
    )
    assert not any(
        "passenger pathway" in item for item in configuration_warnings(separated_config)
    )


def test_named_inference_views_and_fusion_settings_load(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path,
            """
camera:
  inference_views:
    - name: full
      bounds: [0, 0, 1, 1]
      enabled: true
    - name: left
      bounds: [0, 0.15, 0.62, 1]
      enabled: false
detection_fusion:
  method: nms
  iou_threshold: 0.45
  confidence_strategy: maximum
  prefer_full_frame: true
""",
        )
    )

    assert [view.name for view in config.camera.inference_views] == ["full", "left"]
    assert config.camera.inference_views[1].enabled is False
    assert config.detection_fusion.iou_threshold == 0.45
    assert config.detection_fusion.prefer_full_frame is True


@pytest.mark.parametrize(
    ("camera_yaml", "message"),
    [
        (
            """
  inference_views:
    - {name: duplicate, bounds: [0, 0, 1, 1]}
    - {name: duplicate, bounds: [0, 0, 0.5, 1]}
""",
            "names must be unique",
        ),
        (
            """
  inference_views:
    - {name: disabled, bounds: [0, 0, 1, 1], enabled: false}
""",
            "enable at least one view",
        ),
        (
            """
  detection_roi: [0, 0, 1, 1]
  inference_views:
    - {name: full, bounds: [0, 0, 1, 1]}
""",
            "cannot be combined",
        ),
        (
            """
  inference_views:
    - {name: invalid, bounds: [0.8, 0, 0.2, 1]}
""",
            "left < right",
        ),
    ],
)
def test_invalid_inference_view_configuration_is_rejected(
    tmp_path: Path, camera_yaml: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, f"\ncamera:{camera_yaml}"))


def test_sole_crop_warns_when_it_cannot_cover_both_counting_zones(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        """
camera:
  inference_views:
    - name: lower_door
      bounds: [0, 0.3, 1, 1]
""",
    )

    with pytest.warns(UserWarning, match="complete crossing history may be lost"):
        config = load_config(path)

    assert config.camera.inference_views[0].name == "lower_door"
