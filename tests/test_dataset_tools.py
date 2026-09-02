from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackbus.dataset_tools import (
    DatasetError,
    build_parser,
    create_grouped_split_plan,
    initialize_dataset,
    load_dataset_manifest,
    validate_dataset,
)


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    cv2.rectangle(image, (8, 4), (30, 28), (255 - value,) * 3, 2)
    assert cv2.imwrite(str(path), image)


def write_manifest(root: Path, records: list[dict[str, object]]) -> Path:
    path = root / "metadata" / "frames.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "test-overhead",
                "classes": ["person_full", "person_upper_body"],
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def record(
    name: str,
    *,
    camera: str = "cam-01",
    recording: str = "ride-01",
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "image": f"images/source/{name}.jpg",
        "label": f"labels/source/{name}.txt",
        "camera_id": camera,
        "recording_id": recording,
        "tags": tags or [],
    }


def prepare_record(root: Path, item: dict[str, object], value: int) -> None:
    image = root / str(item["image"])
    label = root / str(item["label"])
    write_image(image, value)
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.5 0.5 0.4 0.8\n", encoding="utf-8")


def test_dataset_initializer_creates_standard_yolo_structure_without_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trackbus-data"

    data_yaml, manifest = initialize_dataset(
        root,
        ("person_full", "person_upper_body"),
    )

    assert data_yaml.is_file()
    assert manifest.is_file()
    for split in ("train", "val", "test"):
        assert (root / "images" / split).is_dir()
        assert (root / "labels" / split).is_dir()
    with pytest.raises(DatasetError, match="not empty"):
        initialize_dataset(root, ("person",))


def test_valid_dataset_reports_classes_tags_and_no_accuracy_claim(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    records = [
        record("frame-001", tags=["occlusion", "low_light"]),
        record("frame-002", recording="ride-02", tags=["partial_upper_body"]),
    ]
    prepare_record(root, records[0], 20)
    prepare_record(root, records[1], 90)
    second_label = root / str(records[1]["label"])
    second_label.write_text("1 0.5 0.4 0.3 0.5\n", encoding="utf-8")
    manifest = load_dataset_manifest(write_manifest(root, records))

    report = validate_dataset(root, manifest)
    payload = report.to_dict()

    assert report.valid is True
    assert report.valid_images == report.valid_labels == 2
    assert report.annotated_boxes == 2
    assert report.class_counts == {"person_full": 1, "person_upper_body": 1}
    assert report.tag_counts["occlusion"] == 1
    assert payload["accuracy_claimed"] is False
    assert payload["privacy"]["uploads_performed"] is False


def test_validation_detects_exact_duplicates_corruption_and_invalid_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    records = [record("first"), record("duplicate"), record("corrupt")]
    prepare_record(root, records[0], 50)
    duplicate_image = root / str(records[1]["image"])
    duplicate_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / str(records[0]["image"]), duplicate_image)
    duplicate_label = root / str(records[1]["label"])
    duplicate_label.parent.mkdir(parents=True, exist_ok=True)
    duplicate_label.write_text("0 0.5 0.5 0.4 0.8\n", encoding="utf-8")
    corrupt_image = root / str(records[2]["image"])
    corrupt_image.parent.mkdir(parents=True, exist_ok=True)
    corrupt_image.write_bytes(b"not an image")
    corrupt_label = root / str(records[2]["label"])
    corrupt_label.parent.mkdir(parents=True, exist_ok=True)
    corrupt_label.write_text("4 0.5 0.5 1.5 0.8\n", encoding="utf-8")
    manifest = load_dataset_manifest(write_manifest(root, records))

    report = validate_dataset(root, manifest)
    codes = {issue.code for issue in report.issues}

    assert report.valid is False
    assert report.exact_duplicate_images == 1
    assert "exact_duplicate_image" in codes
    assert "corrupt_image" in codes
    assert "invalid_yolo_label" in codes


def test_split_plan_is_deterministic_and_keeps_recordings_together(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    records = [
        record("a-1", recording="ride-a"),
        record("a-2", recording="ride-a"),
        record("b-1", recording="ride-b"),
        record("c-1", camera="cam-02", recording="ride-c"),
        record("d-1", camera="cam-03", recording="ride-d"),
    ]
    manifest = load_dataset_manifest(write_manifest(root, records))

    first = create_grouped_split_plan(manifest, seed=77)
    second = create_grouped_split_plan(manifest, seed=77)

    assert first == second
    membership = {
        item["image"]: split
        for split, contents in first["splits"].items()
        for item in contents["records"]
    }
    assert membership["images/source/a-1.jpg"] == membership["images/source/a-2.jpg"]
    all_groups = [
        group for contents in first["splits"].values() for group in contents["groups"]
    ]
    assert len(all_groups) == len(set(all_groups)) == 4
    assert first["files_copied"] is False


def test_manifest_rejects_path_traversal_unknown_tags_and_random_split_ratios(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    traversal = record("frame")
    traversal["image"] = "../private.jpg"
    with pytest.raises(DatasetError, match="inside the dataset root"):
        load_dataset_manifest(write_manifest(root, [traversal]))

    unknown_tag = record("frame", tags=["face_identity"])
    with pytest.raises(DatasetError, match="unsupported values"):
        load_dataset_manifest(write_manifest(root, [unknown_tag]))

    manifest = load_dataset_manifest(write_manifest(root, [record("frame")]))
    with pytest.raises(DatasetError, match="sum to 1.0"):
        create_grouped_split_plan(
            manifest,
            train_ratio=0.8,
            validation_ratio=0.3,
            test_ratio=0.1,
        )


def test_dataset_cli_contract_requires_explicit_subcommand_and_paths() -> None:
    args = build_parser().parse_args(
        [
            "split",
            "--dataset-root",
            "datasets/trackbus",
            "--group-by",
            "camera",
            "--output",
            "split-plan.json",
        ]
    )

    assert args.command == "split"
    assert args.group_by == "camera"
    assert args.seed == 2202
