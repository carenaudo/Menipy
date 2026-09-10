"""Independent pre-optimization force reference and full evolution checks."""

from itertools import product

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from menipy.math import active_contour as snake


def original_forces(image, xy, normals, config, precomputed_gradients=None):
    h, w = image.shape[:2]
    gradients = precomputed_gradients
    if gradients is None:
        gradients = snake.precompute_image_gradients(image, config)
    egx, egy, gx, gy = gradients
    px = np.clip(xy[:, 0], 0.0, w - 1.0)
    py = np.clip(xy[:, 1], 0.0, h - 1.0)
    coords = [py, px]
    fx = np.zeros(len(xy), dtype=float)
    fy = np.zeros(len(xy), dtype=float)
    if config.w_edge != 0.0:
        fx += config.w_edge * map_coordinates(egx, coords, order=1, mode="nearest")
        fy += config.w_edge * map_coordinates(egy, coords, order=1, mode="nearest")
    if config.w_line != 0.0:
        fx += config.w_line * map_coordinates(gx, coords, order=1, mode="nearest")
        fy += config.w_line * map_coordinates(gy, coords, order=1, mode="nearest")
    if config.w_flux != 0.0:
        samp_gx = map_coordinates(gx, coords, order=1, mode="nearest")
        samp_gy = map_coordinates(gy, coords, order=1, mode="nearest")
        flux = samp_gx * normals[:, 0] + samp_gy * normals[:, 1]
        fx += config.w_flux * flux * normals[:, 0]
        fy += config.w_flux * flux * normals[:, 1]
    if config.w_balloon != 0.0:
        fx += config.w_balloon * normals[:, 0]
        fy += config.w_balloon * normals[:, 1]
    return np.column_stack([fx, fy])


@pytest.mark.parametrize("weights", list(product([0.0, -0.3], repeat=4)))
@pytest.mark.parametrize("precomputed", [False, True])
def test_exact_forces(weights, precomputed):
    rng = np.random.default_rng(52)
    image = rng.integers(0, 256, (73, 87), dtype=np.uint8)
    points = rng.uniform(-20, 100, (150, 2))
    normals = rng.normal(size=points.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    config = snake.ActiveContourConfig(**dict(zip(
        ("w_edge", "w_line", "w_flux", "w_balloon"), weights
    )))
    gradients = snake.precompute_image_gradients(image, config) if precomputed else None
    copies = [value.copy() for value in (image, points, normals, *(gradients or ()))]
    expected = original_forces(image, points, normals, config, gradients)
    actual = snake.compute_external_forces(image, points, normals, config, gradients)
    np.testing.assert_array_equal(actual, expected)
    for actual_input, before in zip((image, points, normals, *(gradients or ())), copies):
        np.testing.assert_array_equal(actual_input, before)


@pytest.mark.parametrize("boundary", list(snake.SnakeBoundaryCondition))
@pytest.mark.parametrize("line,flux", [(0, 0), (0.2, 0), (0, 0.3), (0.2, 0.3)])
def test_exact_evolution(boundary, line, flux, monkeypatch):
    from tests.test_active_contour import make_synthetic_circle_image

    image = make_synthetic_circle_image()
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    config = snake.ActiveContourConfig(max_iterations=15, w_line=line, w_flux=flux)
    kwargs = {"config": config, "boundary_condition": boundary, "substrate_line": 90.0,
              "pinned_endpoints": (tuple(points[0]), tuple(points[-1]))}
    actual = snake.evolve_active_contour(image, points, **kwargs)
    monkeypatch.setattr(snake, "compute_external_forces", original_forces)
    expected = snake.evolve_active_contour(image, points, **kwargs)
    for name, value in vars(actual).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(expected, name))
        else:
            assert value == getattr(expected, name)


def test_shared_samples(monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return map_coordinates(*args, **kwargs)

    monkeypatch.setattr(snake, "map_coordinates", counted)
    config = snake.ActiveContourConfig(w_line=0.2, w_flux=0.3)
    image = np.zeros((20, 20))
    gradients = snake.precompute_image_gradients(image, config)
    snake.compute_external_forces(image, np.ones((10, 2)), np.ones((10, 2)),
                                  config, gradients)
    assert len(calls) == 4
    assert all(sum(value is field for value in calls) == 1 for field in gradients)
