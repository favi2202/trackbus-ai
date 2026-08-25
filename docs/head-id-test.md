# HEAD_id_test: anonymous head-detector experiment

This branch tests whether a doorway-specific head detector preserves observations
better than the current full-person detector on crowded overhead footage. It is
an experiment, not a production replacement and not an accuracy claim.

TrackBus still uses temporary anonymous tracking IDs. `H12` means only “head
track 12 in this process.” It is not a passenger identity. The experiment does
not recognize faces or classify hair, baldness, caps, hoods, age, gender, or any
other passenger attribute. Those appearances are annotation diversity for one
class named `head`.

## 1. Check out the isolated branch

From the repository root in Windows PowerShell:

```powershell
git fetch origin
git switch HEAD_id_test
git pull --ff-only
git status --short
```

The production branch remains `agent/trackbus-foundation`.

## 2. Extract local frames without inventing labels

This command samples up to 200 images from the known 589-frame test clip. It
does not upload footage and does not generate pseudo-labels:

```powershell
& ".\.venv\Scripts\python.exe" -m trackbus.extract_head_frames `
  --video ".\data\input\bus-test-1.mp4" `
  --output ".\datasets\head-raw\bus-test-1" `
  --every 3 `
  --max-frames 200 `
  --jpeg-quality 95
```

The output includes `extraction.json` with the exact source frames and privacy
contract. The command refuses to overwrite a non-empty folder.

Create the final one-class dataset structure separately:

```powershell
& ".\.venv\Scripts\python.exe" -m trackbus.dataset_tools init `
  --dataset-root ".\datasets\trackbus-head" `
  --classes "head"
```

Manually annotate one tight box around the visible head, including natural hair,
caps, or hoods inside that same box. Use only class `head`. Include difficult
examples—crowding, blur, partial occlusion, image edges, low light—and negative
frames containing handles, seats, bags, and reflections. Do not create face,
hair, cap, hood, or identity classes.

Do not train and test on neighboring frames from one ride. Gather multiple
authorized recordings and split complete recordings between train, validation,
and test. The repository ignores `/datasets/`, so raw passenger imagery is not
committed.

The aerial-person dataset discussed earlier is not a drop-in head dataset: its
person boxes can help a separate person experiment, but they do not provide the
required bus-door head boxes.

## 3. Validate and train

After placing images and YOLO labels under `datasets/trackbus-head`, update its
metadata manifest and run:

```powershell
& ".\.venv\Scripts\python.exe" -m trackbus.dataset_tools validate `
  --dataset-root ".\datasets\trackbus-head" `
  --output ".\data\output\head-dataset-validation.json"

& ".\.venv\Scripts\yolo.exe" `
  "cfg=.\configs\training\yolo11s_head_id_test.yaml"
```

The bounded starter configuration uses YOLO11s, 960-pixel input, batch 4,
Windows-safe worker count 2, deterministic seed 2202, and local output under
`data/output/training/yolo11s-head-id-test`. Tune only after saving a baseline.

The expected weights path is:

```text
data/output/training/yolo11s-head-id-test/weights/best.pt
```

## 4. Verify the model label before a full replay

Run a short detector-only smoke test. Head mode checks the model metadata and
fails immediately unless class 0 is exactly `head`:

```powershell
& ".\.venv\Scripts\python.exe" -m trackbus.benchmark_detection `
  --video ".\data\input\bus-test-1.mp4" `
  --model ".\data\output\training\yolo11s-head-id-test\weights\best.pt" `
  --detection-target head `
  --imgsz 960 `
  --detector-floor 0.05 `
  --device cuda `
  --max-frames 60 `
  --output ".\data\output\head-detector-smoke.json"
```

Passing `yolo11s.pt` with `--detection-target head` must fail because its class 0
is `person`; that safety check prevents a misleading experiment.

## 5. Run the same clip twice

Use identical video, zones, image size, threshold, tracker, counter settings, and
preprocessing. Change only the target and weights.

Person baseline:

```powershell
& ".\.venv\Scripts\python.exe" ".\showcase.py" `
  --video ".\data\input\bus-test-1.mp4" `
  --zones ".\configs\bus-test-1-zones.yaml" `
  --model "yolo11s.pt" `
  --detection-target person `
  --imgsz 960 `
  --confidence 0.30 `
  --detector-floor 0.05 `
  --tracker ".\configs\tracking_occlusion.yaml" `
  --minimum-zone-frames 2 `
  --preprocessing-profile contrast `
  --device cuda `
  --headless `
  --queue-dir ".\data\offline-queue-head-test\person" `
  --trajectory-log ".\data\output\head-id-test-person.jsonl"
```

Head candidate:

```powershell
& ".\.venv\Scripts\python.exe" ".\showcase.py" `
  --video ".\data\input\bus-test-1.mp4" `
  --zones ".\configs\bus-test-1-zones.yaml" `
  --model ".\data\output\training\yolo11s-head-id-test\weights\best.pt" `
  --detection-target head `
  --imgsz 960 `
  --confidence 0.30 `
  --detector-floor 0.05 `
  --tracker ".\configs\tracking_occlusion.yaml" `
  --minimum-zone-frames 2 `
  --preprocessing-profile contrast `
  --device cuda `
  --headless `
  --queue-dir ".\data\offline-queue-head-test\head" `
  --trajectory-log ".\data\output\head-id-test-head.jsonl"
```

These runs intentionally omit `--api-url`, so experimental events are not sent
to the hosted D1 database.

## 6. Compare against the known 6 IN / 2 OUT truth

```powershell
& ".\.venv\Scripts\python.exe" -m trackbus.compare_head_test `
  --person-log ".\data\output\head-id-test-person.jsonl" `
  --head-log ".\data\output\head-id-test-head.jsonl" `
  --ground-truth-in 6 `
  --ground-truth-out 2 `
  --output ".\data\output\head-id-test-comparison.json"

Get-Content ".\data\output\head-id-test-comparison.json"
```

The comparison is rejected if the source or processed-frame count differs. The
winner is the lower value of:

```text
|predicted IN - 6| + |predicted OUT - 2|
```

One clip can decide whether the idea deserves more work; it cannot establish a
production accuracy figure. Keep the person pipeline unless the head candidate
also wins on independent rides, detector box annotations, runtime, and false
event review.

## 7. Run branch verification

```powershell
& ".\.venv\Scripts\python.exe" -m ruff check showcase.py trackbus tests
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" ".\showcase.py" --help
& ".\.venv\Scripts\python.exe" -m trackbus.extract_head_frames --help
& ".\.venv\Scripts\python.exe" -m trackbus.compare_head_test --help
```
