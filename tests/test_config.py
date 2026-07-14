from pathlib import Path

import pytest

from trackbus.config import ConfigError, load_config

VALID_CONFIG = """
model:
  path: yolo11n.pt
  confidence: 0.4
tracking:
  minimum_zone_frames: 2
  stale_track_timeout: 30
zones:
  outside: [[0, 0], [1, 0], [1, 0.4], [0, 0.4]]
  inside: [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]
outputs:
  video: result.mp4
capacity: 40
initial_occupancy: 0
"""


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_configuration_loads(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.capacity == 40
    assert config.model.confidence == 0.4
    assert config.zones.outside[2] == (1.0, 0.4)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("capacity: 40", "capacity: 0", "capacity"),
        ("confidence: 0.4", "confidence: 1.5", "confidence"),
        ("initial_occupancy: 0", "initial_occupancy: -1", "initial_occupancy"),
        ("stale_track_timeout: 30", "stale_track_timeout: 0", "stale"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = write_config(tmp_path, VALID_CONFIG.replace(old, new))

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_out_of_range_normalized_coordinate_is_rejected(tmp_path: Path) -> None:
    invalid = VALID_CONFIG.replace("[1, 0.4]", "[1.2, 0.4]")

    with pytest.raises(ConfigError, match="normalized"):
        load_config(write_config(tmp_path, invalid))


def test_missing_configuration_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
