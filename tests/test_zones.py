import pytest

from trackbus.config import ZonesConfig
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
