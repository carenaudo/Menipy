"""Skip iteration geometry only when forces are independent of normals."""

import numpy as np
import pytest

from menipy.math import active_contour as snake
from tests.test_active_contour import make_synthetic_circle_image


def previous_normals(xy, closed, config=None):
    return snake._contour_directions(xy, closed)[1]


@pytest.mark.parametrize("boundary", list(snake.SnakeBoundaryCondition))
@pytest.mark.parametrize("flux,balloon", [(0, 0), (0.2, 0), (0, -0.3), (0.2, -0.3)])
def test_exact_evolution(boundary, flux, balloon, monkeypatch):
    image = make_synthetic_circle_image()
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    cfg = snake.ActiveContourConfig(max_iterations=25, convergence=0.0,
                                    w_flux=flux, w_balloon=balloon)
    kwargs = {"config": cfg, "boundary_condition": boundary, "substrate_line": 90.0,
              "pinned_endpoints": (tuple(points[0]), tuple(points[-1]))}
    actual = snake.evolve_active_contour(image, points, **kwargs)
    monkeypatch.setattr(snake, "_evolution_normals", previous_normals)
    expected = snake.evolve_active_contour(image, points, **kwargs)
    for field, value in vars(actual).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(expected, field))
        else:
            assert value == getattr(expected, field)


@pytest.mark.parametrize("flux,balloon", [(0, 0), (0.2, 0), (0, -0.3), (0.2, -0.3)])
def test_geometry_call_count(flux, balloon, monkeypatch):
    calls = []
    original = snake._contour_directions

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(snake, "_contour_directions", counted)
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    result = snake.evolve_active_contour(make_synthetic_circle_image(), points,
        config=snake.ActiveContourConfig(max_iterations=25, convergence=0.0,
                                         w_flux=flux, w_balloon=balloon))
    assert len(calls) == (result.iterations + 1 if flux or balloon else 1)


@pytest.mark.parametrize("flux,balloon", [(0.2, 0), (0, -0.3)])
def test_missing_required_normals(flux, balloon):
    with pytest.raises(ValueError, match="Normals are required"):
        snake.compute_external_forces(np.zeros((10, 10)), np.ones((4, 2)), None,
            snake.ActiveContourConfig(w_flux=flux, w_balloon=balloon))


def test_configuration_change_rechecks_need():
    points = np.ones((4, 2))
    cfg = snake.ActiveContourConfig()
    assert snake._evolution_normals(points, False, cfg) is None
    cfg.w_balloon = 0.1
    np.testing.assert_array_equal(snake._evolution_normals(points, False, cfg),
                                  previous_normals(points, False))
