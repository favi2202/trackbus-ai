"""Privacy-conscious validation and split planning for future TrackBus data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAILURE_TAGS = {
    "occlusion",
    "edge",
    "low_light",
    "crowding",
    "compression",
    "partial_upper_body",
    "full_person",
    "small_head",
    "headwear",
    "hood",
    "hair_variation",
    "motion_blur",
}
DEFAULT_CLASSES = ("person_full", "person_upper_body")


class DatasetError(ValueError):
    """Raised when a dataset command would produce unreliable evidence."""


@dataclass(frozen=True)
class DatasetRecord:
    image: Path
    label: Path
    camera_id: str
    recording_id: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "image": self.image.as_posix(),
            "label": self.label.as_posix(),
            "camera_id": self.camera_id,
            "recording_id": self.recording_id,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class DatasetManifest:
    classes: tuple[str, ...]
    records: tuple[DatasetRecord, ...]
    dataset_name: str = "trackbus-overhead"


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    path: str | None
    message: str


@dataclass(frozen=True)
class DatasetValidationReport:
    dataset_root: str
    manifest_records: int
    valid_images: int
    valid_labels: int
    annotated_boxes: int
    class_counts: Mapping[str, int]
    tag_counts: Mapping[str, int]
    exact_duplicate_images: int
    issues: tuple[DatasetIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "workflow": "trackbus_dataset_validation",
            "valid": self.valid,
            "dataset_root": self.dataset_root,
            "manifest_records": self.manifest_records,
            "valid_images": self.valid_images,
            "valid_labels": self.valid_labels,
            "annotated_boxes": self.annotated_boxes,
            "class_counts": dict(self.class_counts),
            "tag_counts": dict(self.tag_counts),
            "exact_duplicate_images": self.exact_duplicate_images,
            "issues": [asdict(issue) for issue in self.issues],
            "privacy": {
                "uploads_performed": False,
                "biometric_identity_required": False,
                "persistent_passenger_identity_allowed": False,
                "images_may_contain_personal_data": True,
            },
            "accuracy_claimed": False,
        }


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Load the explicit frame manifest used for grouping and validation."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"Could not read dataset manifest '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Dataset manifest is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetError("Dataset manifest root must be an object.")
    _reject_unknown(
        document,
        "dataset manifest",
        {"schema_version", "dataset_name", "classes", "records", "notes"},
    )
    if document.get("schema_version") != 1:
        raise DatasetError("Dataset manifest schema_version must be 1.")
    dataset_name = _non_empty_text(
        document.get("dataset_name", "trackbus-overhead"),
        "dataset_name",
    )
    raw_classes = document.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise DatasetError("classes must be a non-empty list.")
    classes = tuple(
        _non_empty_text(value, f"classes[{index}]")
        for index, value in enumerate(raw_classes)
    )
    if len(set(classes)) != len(classes):
        raise DatasetError("classes must be unique.")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise DatasetError("records must be a list.")
    records = tuple(
        _parse_record(value, index) for index, value in enumerate(raw_records)
    )
    image_paths = [record.image for record in records]
    if len(set(image_paths)) != len(image_paths):
        raise DatasetError("records contain duplicate image paths.")
    return DatasetManifest(classes, records, dataset_name)


def validate_dataset(
    dataset_root: Path,
    manifest: DatasetManifest,
) -> DatasetValidationReport:
    """Validate images and YOLO labels without retaining decoded frames."""

    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"Dataset root does not exist: {root}")
    issues: list[DatasetIssue] = []
    valid_images = 0
    valid_labels = 0
    annotated_boxes = 0
    class_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    hashes: dict[str, Path] = {}
    duplicate_images = 0

    for record in manifest.records:
        tag_counts.update(record.tags)
        image_path = root / record.image
        label_path = root / record.label
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            issues.append(
                DatasetIssue(
                    "error",
                    "unsupported_image_extension",
                    record.image.as_posix(),
                    f"Supported image extensions: {', '.join(sorted(IMAGE_SUFFIXES))}.",
                )
            )
        elif not image_path.is_file():
            issues.append(
                DatasetIssue(
                    "error",
                    "missing_image",
                    record.image.as_posix(),
                    "Manifest image does not exist.",
                )
            )
        else:
            decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                issues.append(
                    DatasetIssue(
                        "error",
                        "corrupt_image",
                        record.image.as_posix(),
                        "OpenCV could not decode the image.",
                    )
                )
            else:
                valid_images += 1
                digest = _sha256(image_path)
                previous = hashes.get(digest)
                if previous is None:
                    hashes[digest] = record.image
                else:
                    duplicate_images += 1
                    issues.append(
                        DatasetIssue(
                            "error",
                            "exact_duplicate_image",
                            record.image.as_posix(),
                            f"Exact duplicate of {previous.as_posix()}.",
                        )
                    )

        if not label_path.is_file():
            issues.append(
                DatasetIssue(
                    "error",
                    "missing_label",
                    record.label.as_posix(),
                    "Manifest label does not exist.",
                )
            )
            continue
        label_errors, box_classes = _validate_yolo_label(
            label_path,
            manifest.classes,
        )
        if label_errors:
            issues.extend(
                DatasetIssue(
                    "error",
                    "invalid_yolo_label",
                    record.label.as_posix(),
                    message,
                )
                for message in label_errors
            )
            continue
        valid_labels += 1
        annotated_boxes += len(box_classes)
        class_counts.update(box_classes)

    if not manifest.records:
        issues.append(
            DatasetIssue(
                "warning",
                "empty_manifest",
                None,
                "No records are available to validate or split.",
            )
        )
    return DatasetValidationReport(
        dataset_root=str(root),
        manifest_records=len(manifest.records),
        valid_images=valid_images,
        valid_labels=valid_labels,
        annotated_boxes=annotated_boxes,
        class_counts=dict(sorted(class_counts.items())),
        tag_counts=dict(sorted(tag_counts.items())),
        exact_duplicate_images=duplicate_images,
        issues=tuple(issues),
    )


def create_grouped_split_plan(
    manifest: DatasetManifest,
    *,
    group_by: str = "recording",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 2202,
) -> dict[str, Any]:
    """Assign complete cameras or recordings to deterministic dataset splits."""

    ratios = {
        "train": _ratio(train_ratio, "train_ratio"),
        "validation": _ratio(validation_ratio, "validation_ratio"),
        "test": _ratio(test_ratio, "test_ratio"),
    }
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise DatasetError("train, validation, and test ratios must sum to 1.0.")
    if group_by not in {"camera", "recording"}:
        raise DatasetError("group_by must be camera or recording.")
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in manifest.records:
        group = (
            record.camera_id
            if group_by == "camera"
            else f"{record.camera_id}/{record.recording_id}"
        )
        grouped.setdefault(group, []).append(record)

    split_records: dict[str, list[DatasetRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_groups: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    train_boundary = ratios["train"]
    validation_boundary = train_boundary + ratios["validation"]
    for group in sorted(grouped):
        score = _stable_group_score(group, seed)
        split = (
            "train"
            if score < train_boundary
            else "validation"
            if score < validation_boundary
            else "test"
        )
        split_groups[split].append(group)
        split_records[split].extend(grouped[group])

    return {
        "schema_version": 1,
        "workflow": "trackbus_grouped_split_plan",
        "dataset_name": manifest.dataset_name,
        "strategy": "grouped_not_neighboring_frames",
        "group_by": group_by,
        "seed": seed,
        "ratios": ratios,
        "classes": list(manifest.classes),
        "splits": {
            split: {
                "groups": split_groups[split],
                "record_count": len(split_records[split]),
                "records": [
                    record.to_dict()
                    for record in sorted(
                        split_records[split], key=lambda item: item.image.as_posix()
                    )
                ],
            }
            for split in ("train", "validation", "test")
        },
        "files_copied": False,
        "accuracy_claimed": False,
        "privacy_review_required_before_training": True,
    }


def initialize_dataset(root: Path, classes: Sequence[str]) -> tuple[Path, Path]:
    """Create an empty standard YOLO structure without overwriting existing data."""

    resolved = root.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise DatasetError(f"Dataset directory is not empty: {resolved}")
    parsed_classes = tuple(
        _non_empty_text(value, f"classes[{index}]")
        for index, value in enumerate(classes)
    )
    if not parsed_classes or len(set(parsed_classes)) != len(parsed_classes):
        raise DatasetError("At least one unique dataset class is required.")
    for split in ("train", "val", "test"):
        (resolved / "images" / split).mkdir(parents=True, exist_ok=True)
        (resolved / "labels" / split).mkdir(parents=True, exist_ok=True)
    metadata = resolved / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    data_yaml = resolved / "data.yaml"
    manifest_path = metadata / "frames.json"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(resolved),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(parsed_classes)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": resolved.name,
                "classes": list(parsed_classes),
                "records": [],
                "notes": (
                    "Use temporary frame identifiers only; never store passenger "
                    "names or biometric identity."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_yaml, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trackbus.dataset_tools",
        description=(
            "Prepare and validate local YOLO data without training, uploading, or "
            "claiming accuracy."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Create an empty YOLO structure")
    initialize.add_argument("--dataset-root", required=True, type=Path)
    initialize.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated future detector classes",
    )

    validate = commands.add_parser("validate", help="Validate images and labels")
    _add_dataset_arguments(validate)
    validate.add_argument("--output", type=Path, help="Optional JSON report")

    split = commands.add_parser("split", help="Create a group-safe split plan")
    _add_dataset_arguments(split)
    split.add_argument(
        "--group-by",
        choices=("camera", "recording"),
        default="recording",
    )
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--validation", type=float, default=0.20)
    split.add_argument("--test", type=float, default=0.10)
    split.add_argument("--seed", type=int, default=2202)
    split.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            classes = tuple(value.strip() for value in args.classes.split(","))
            data_yaml, manifest_path = initialize_dataset(args.dataset_root, classes)
            print(f"YOLO dataset structure created: {data_yaml.parent}")
            print(f"Manifest template: {manifest_path}")
            return 0

        manifest_path = args.manifest or args.dataset_root / "metadata" / "frames.json"
        manifest = load_dataset_manifest(manifest_path)
        report = validate_dataset(args.dataset_root, manifest)
        if args.command == "validate":
            if args.output is not None:
                _write_json(args.output, report.to_dict())
            _print_validation(report)
            return 0 if report.valid else 1
        if not report.valid:
            raise DatasetError(
                "Dataset validation failed; fix errors before creating a split plan."
            )
        plan = create_grouped_split_plan(
            manifest,
            group_by=args.group_by,
            train_ratio=args.train,
            validation_ratio=args.validation,
            test_ratio=args.test,
            seed=args.seed,
        )
        _write_json(args.output, plan)
        print(f"Grouped split plan written: {args.output}")
        print("No images or labels were copied or uploaded.")
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"TrackBus dataset error: {exc}", file=sys.stderr)
        return 2


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)


def _parse_record(value: object, index: int) -> DatasetRecord:
    name = f"records[{index}]"
    if not isinstance(value, dict):
        raise DatasetError(f"{name} must be an object.")
    _reject_unknown(
        value,
        name,
        {"image", "label", "camera_id", "recording_id", "tags"},
    )
    raw_tags = value.get("tags", [])
    if not isinstance(raw_tags, list):
        raise DatasetError(f"{name}.tags must be a list.")
    tags = tuple(
        _non_empty_text(tag, f"{name}.tags[{tag_index}]")
        for tag_index, tag in enumerate(raw_tags)
    )
    unknown_tags = sorted(set(tags) - FAILURE_TAGS)
    if unknown_tags:
        raise DatasetError(
            f"{name}.tags contains unsupported values: {', '.join(unknown_tags)}."
        )
    return DatasetRecord(
        image=_relative_path(value.get("image"), f"{name}.image"),
        label=_relative_path(value.get("label"), f"{name}.label"),
        camera_id=_non_empty_text(value.get("camera_id"), f"{name}.camera_id"),
        recording_id=_non_empty_text(value.get("recording_id"), f"{name}.recording_id"),
        tags=tuple(dict.fromkeys(tags)),
    )


def _validate_yolo_label(
    path: Path,
    classes: Sequence[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    box_classes: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"Could not read label text: {exc}"], []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number} must contain five values")
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {line_number} contains non-numeric values")
            continue
        if not 0 <= class_id < len(classes):
            errors.append(f"line {line_number} class id is outside configured classes")
            continue
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number} contains non-finite coordinates")
            continue
        if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
            errors.append(f"line {line_number} center must be normalized")
            continue
        if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
            errors.append(f"line {line_number} width and height must be in (0, 1]")
            continue
        if (
            center_x - width / 2 < -1e-6
            or center_x + width / 2 > 1.0 + 1e-6
            or center_y - height / 2 < -1e-6
            or center_y + height / 2 > 1.0 + 1e-6
        ):
            errors.append(f"line {line_number} box extends outside the image")
            continue
        box_classes.append(classes[class_id])
    return errors, box_classes


def _stable_group_score(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object, name: str) -> Path:
    text = _non_empty_text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise DatasetError(f"{name} must remain inside the dataset root.")
    return path


def _non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{name} must be non-empty text.")
    return value.strip()


def _ratio(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise DatasetError(f"{name} must be greater than 0 and below 1.")
    return parsed


def _reject_unknown(value: Mapping[str, object], name: str, allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DatasetError(f"{name} contains unknown keys: {', '.join(unknown)}.")


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _print_validation(report: DatasetValidationReport) -> None:
    status = "VALID" if report.valid else "INVALID"
    print(f"TrackBus dataset: {status}")
    print(
        f"records={report.manifest_records} images={report.valid_images} "
        f"labels={report.valid_labels} boxes={report.annotated_boxes}"
    )
    for issue in report.issues:
        location = f" [{issue.path}]" if issue.path else ""
        print(f"{issue.severity.upper()} {issue.code}{location}: {issue.message}")
    print("No files were uploaded and no passenger identity was processed.")


if __name__ == "__main__":
    sys.exit(main())
