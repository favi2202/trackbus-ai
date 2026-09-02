# Camera-zone calibration

`trackbus.calibrate_camera` is a local, interactive editor for TrackBus camera
geometry. It draws on a decoded source frame only: it does not run detection,
infer crossing events, read ground truth, or change counting thresholds.

Calibrations are camera-specific. Recalibrate whenever the camera position,
rotation, crop, or source aspect ratio changes.

## Start the editor

Run the command from the repository root in a graphical desktop session:

```powershell
python -m trackbus.calibrate_camera `
  --input data/input/bus_door_02.mp4 `
  --output-config configs/cameras/bus_door_02.yaml `
  --frame 350
```

`--frame` selects the initial representative frame. If the output YAML already
exists, the editor loads its zones and preserves all unrelated settings when it
saves. If it does not exist, draw both required counting zones before saving.
Saving uses an atomic replacement; cancelling leaves the existing file intact.
The output path may never be the source-video path.

Choose a frame that clearly shows the doorway structure. It is often useful to
inspect several frames to make sure doors, shadows, and passengers do not hide a
boundary.

## Controls

| Control | Action |
| --- | --- |
| `1` | Select the `INSIDE` polygon |
| `2` | Select the `OUTSIDE` polygon |
| `3` | Select the optional crossing-corridor polygon |
| `4` | Select exclusion polygons; each finished draft adds one polygon |
| Left click | Add a draft vertex, or move a selected existing vertex |
| Right click | Select the nearest existing vertex for editing |
| `Enter` | Finish the current polygon |
| `U` | Undo the latest edit |
| `C` | Clear the selected layer; this clears all exclusions in layer 4 |
| `R` | Reset geometry and anchor mode to their startup values |
| `M` | Cycle `center`, `bottom_center`, and `top_center` anchor previews |
| `[` / `]` | Move the representative frame backward/forward by one frame |
| `J` / `K` | Move backward/forward by approximately one second |
| `S` | Validate, save, and close |
| `Q` / `Esc` | Cancel without saving |

The magenta dot on the sample box previews the selected zone anchor. This box is
only a visualization; the calibrator never creates detections.

## Geometry

Coordinates are saved in normalized source-frame space: `[0, 0]` is the top
left and `[1, 1]` is the bottom right. Tracking and counting later scale these
coordinates to the actual decoded frame.

The editor owns these YAML fields:

```yaml
zones:
  inside:
    - [0.20, 0.10]
    - [0.80, 0.10]
    - [0.80, 0.38]
    - [0.20, 0.38]
  outside:
    - [0.20, 0.65]
    - [0.80, 0.65]
    - [0.80, 0.95]
    - [0.20, 0.95]

tracking:
  zone_anchor: bottom_center

camera:
  crossing_corridor:
    - [0.30, 0.25]
    - [0.70, 0.25]
    - [0.70, 0.78]
    - [0.30, 0.78]
  exclusion_polygons: []
```

The example coordinates are illustrative, not a universal bus-door preset.

`INSIDE` and `OUTSIDE` must represent opposite sides of the crossing. Leave a
real neutral region between them so a valid event has spatial history rather
than a one-frame boundary flip. The optional corridor should follow the
intended passenger path, intersect both counting zones, and include some of the
neutral region.

Exclusion polygons are only for known static structures that the detector
mistakes for people. Never place one across a passenger pathway. Exclusions can
hide real detections and therefore require review with diagnostic output after
calibration.

## Save-time validation

The editor refuses to save when:

- either required counting zone is missing;
- a polygon has fewer than three points, an out-of-range coordinate, or zero
  area;
- `INSIDE` and `OUTSIDE` overlap by more than 10% of the smaller zone;
- no usable neutral component connects the two zones;
- a configured corridor does not intersect both zones; or
- a configured corridor contains no usable neutral area; or
- an exclusion polygon overlaps either counting zone or the crossing corridor.

Validation checks geometry, not semantic correctness. After saving, run normal
TrackBus diagnostics and event evaluation against independently annotated
ground truth. Calibration must not be judged from aggregate totals alone.
