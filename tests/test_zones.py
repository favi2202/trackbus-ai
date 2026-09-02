import pytest

from trackbus.config import ZonesConfig
from trackbus.detection import AnchorMode, Detection, TrackedDetection, anchor_for_box
from trackbus.zones import PixelZones, ZoneMembership


@pytest.fixture
def zones() -> PixelZones:
    config = ZonesConfig(
        outside=((0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)),
        inside=((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)),
    )
    return PixelZones.from_normalized(config, frame_width=100, frame_height=100)


def test_zone_membership(zones: PixelZones) -> None:
    assert zones.membership((50, 20)) is ZoneMembership.OUTSIDE
    assert zones.membership((50, 80)) is ZoneMembership.INSIDE
    assert zones.membership((50, 50)) is ZoneMembership.TRANSITION


def test_normalized_one_maps_inside_frame(zones: PixelZones) -> None:
    assert zones.inside[:, 0].max() == 99
    assert zones.inside[:, 1].max() == 99


def test_invalid_frame_dimensions_are_rejected() -> None:
    config = ZonesConfig(
        outside=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        inside=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="positive"):
        PixelZones.from_normalized(config, 0, 100)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (AnchorMode.CENTER, (20.0, 40.0)),
        (AnchorMode.BOTTOM_CENTER, (20.0, 60.0)),
        (AnchorMode.TOP_CENTER, (20.0, 20.0)),
    ],
)
def test_configurable_box_anchor_modes(
    mode: AnchorMode, expected: tuple[float, float]
) -> None:
    box = (10.0, 20.0, 30.0, 60.0)
    detection = Detection(box, 0.9, 0, "full")
    tracked = TrackedDetection(7, box, 0.9)

    assert anchor_for_box(box, mode) == expected
    assert detection.anchor_for(mode) == expected
    assert tracked.anchor_for(mode) == expected
    assert detection.anchor == tracked.anchor == (20.0, 60.0)


def test_invalid_anchor_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="anchor mode"):
        anchor_for_box((0.0, 0.0, 10.0, 10.0), "feet")


def test_zone_boundary_hysteresis_turns_only_shallow_hits_neutral(
    zones: PixelZones,
) -> None:
    shallow = zones.classify((50.0, 38.0), boundary_hysteresis=0.03)
    deep = zones.classify((50.0, 30.0), boundary_hysteresis=0.03)

    assert shallow.raw is ZoneMembership.OUTSIDE
    assert shallow.effective is ZoneMembership.TRANSITION
    assert shallow.boundary_jitter is True
    assert deep.raw is deep.effective is ZoneMembership.OUTSIDE


@pytest.mark.parametrize("hysteresis", [-0.01, 0.11])
def test_invalid_zone_boundary_hysteresis_is_rejected(
    zones: PixelZones, hysteresis: float
) -> None:
    with pytest.raises(ValueError, match="between 0 and 0.10"):
        zones.classify((50.0, 20.0), hysteresis)
