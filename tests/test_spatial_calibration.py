import pytest

from menipy.common.spatial_calibration import px_per_mm_from_known_distance


def test_known_distance_accepts_a_tilted_segment():
    assert px_per_mm_from_known_distance(((0, 0), (3, 4)), 0.5) == pytest.approx(10.0)


@pytest.mark.parametrize("points,distance", [(((1, 1), (1, 1)), 1.0), (((0, 0), (2, 0)), 0.0)])
def test_known_distance_rejects_degenerate_input(points, distance):
    with pytest.raises(ValueError):
        px_per_mm_from_known_distance(points, distance)
