# TrackBus overhead dataset preparation

This workflow prepares future representative bus-door footage for review. It
does not train a model, upload images, identify passengers, or claim an accuracy
improvement.

## Privacy and collection boundary

Use footage only when collection, retention, and annotation are authorized.
Frames may contain personal data even though TrackBus uses no face recognition.
Keep the dataset local, assign temporary frame identifiers, remove unnecessary
audio and metadata, define a retention period, and restrict access. Never store
passenger names, face embeddings, payment identity, or persistent person IDs.

The repository ignores `/datasets/` so images and labels are not committed by
accident.

## Standard directory structure

From the repository root, create an empty structure:

```powershell
python -m trackbus.dataset_tools init `
  --dataset-root ".\datasets\trackbus-overhead" `
  --classes "person_full,person_upper_body"
```

The command refuses to overwrite a non-empty directory and creates:

```text
datasets/trackbus-overhead/
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
  metadata/frames.json
```

`person_full` is a visible whole-person box. `person_upper_body` is a deliberately
separate future class for a head-and-shoulders/upper-torso observation where a
reliable full-person box cannot be annotated. Do not use a face-only class.

## Frame manifest

Each record in `metadata/frames.json` names one image and YOLO label plus the
camera and recording group:

```json
{
  "schema_version": 1,
  "dataset_name": "trackbus-overhead-pilot-01",
  "classes": ["person_full", "person_upper_body"],
  "records": [
    {
      "image": "images/source/cam01-ride07-000120.jpg",
      "label": "labels/source/cam01-ride07-000120.txt",
      "camera_id": "cam01",
      "recording_id": "ride07",
      "tags": ["occlusion", "crowding"]
    }
  ]
}
```

Supported diagnostic tags are `occlusion`, `edge`, `low_light`, `crowding`,
`compression`, `partial_upper_body`, and `full_person`. They describe failure
conditions; they are not passenger identity.

## Validate before splitting

```powershell
python -m trackbus.dataset_tools validate `
  --dataset-root ".\datasets\trackbus-overhead" `
  --output ".\data\output\dataset-validation.json"
```

Validation checks image decoding, exact duplicate images, missing files, UTF-8
labels, class ranges, normalized YOLO boxes, boxes extending outside the image,
class totals, and failure-tag totals. A valid report is evidence of dataset
structure—not detector quality or annotation correctness.

## Group-safe split plan

Neighboring frames from the same ride must not be randomly divided across
training and evaluation. That leaks near-identical scenes and produces an
optimistic benchmark. Create a deterministic recording-level plan instead:

```powershell
python -m trackbus.dataset_tools split `
  --dataset-root ".\datasets\trackbus-overhead" `
  --group-by recording `
  --train 0.70 `
  --validation 0.20 `
  --test 0.10 `
  --seed 2202 `
  --output ".\data\output\dataset-split-plan.json"
```

Use `--group-by camera` when the evaluation must prove generalization to unseen
camera geometry. The command writes a plan only; it does not copy, move, delete,
or upload files. Review the group distribution before applying it.

## Future training—not yet an accuracy claim

After annotations and splits are independently reviewed, the bounded starting
configuration is `configs/training/yolo11n_overhead.yaml`:

```powershell
yolo cfg=".\configs\training\yolo11n_overhead.yaml"
```

Start with YOLO11n. Record dataset hash/version, GPU, package versions, effective
configuration, per-class precision/recall, validation camera groups, test camera
groups, and runtime. Keep the current production detector until held-out
detection and end-to-end event benchmarks show a real improvement.
