"""Exact reference for deferred snake curvature calculation."""

import numpy as np
import pytest

from menipy.math import active_contour as snake


def original_geometry(
    xy: np.ndarray,
    closed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit tangent vectors, unit outward normals, and local curvatures.

    For an open drop silhouette ordered from left-contact to right-contact,
    the outward normal points away from the drop interior (into the background).

    Args:
        xy: (N, 2) array of coordinates.
        closed: Whether the contour is a closed loop.

    Returns:
        tangents: (N, 2) unit tangent vectors.
        normals: (N, 2) unit outward normal vectors.
        curvatures: (N,) local signed curvatures.
    """
    n = len(xy)
    if n < 3:
        zeros_2d = np.zeros((n, 2), dtype=float)
        return zeros_2d, zeros_2d, np.zeros(n, dtype=float)

    if closed:
        prev_xy = np.roll(xy, 1, axis=0)
        next_xy = np.roll(xy, -1, axis=0)
    else:
        prev_xy = np.vstack([xy[0], xy[:-1]])
        next_xy = np.vstack([xy[1:], xy[-1]])

    # Central difference tangent
    dx = next_xy[:, 0] - prev_xy[:, 0]
    dy = next_xy[:, 1] - prev_xy[:, 1]
    lengths = np.hypot(dx, dy)
    lengths[lengths < 1e-9] = 1.0

    tx = dx / lengths
    ty = dy / lengths
    tangents = np.column_stack([tx, ty])

    # Outward normal is perpendicular to tangent: (ty, -tx)
    normals = np.column_stack([ty, -tx])

    # Arc-length step ds between adjacent points
    ds = lengths / 2.0
    ds[ds < 1e-9] = 1.0

    if closed:
        prev_t = np.roll(tangents, 1, axis=0)
        next_t = np.roll(tangents, -1, axis=0)
    else:
        prev_t = np.vstack([tangents[0], tangents[:-1]])
        next_t = np.vstack([tangents[1:], tangents[-1]])

    dtx = (next_t[:, 0] - prev_t[:, 0]) / (2.0 * ds)
    dty = (next_t[:, 1] - prev_t[:, 1]) / (2.0 * ds)
    curvatures = dtx * ty - dty * tx

    return tangents, normals, curvatures




def original_iteration_normals(xy, closed, config=None):
    return original_geometry(xy, closed)[1]


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 80, 300])
@pytest.mark.parametrize('degenerate', [False, True])
def test_exact_geometry(closed, n, degenerate):
    points = np.random.default_rng(25).normal(size=(n, 2))
    if degenerate:
        points[:] = 1.0
    before = points.copy()
    for actual, expected in zip(snake.compute_contour_normals(points, closed),
                                original_geometry(points, closed)):
        np.testing.assert_array_equal(actual, expected)
    if n >= 3:
        np.testing.assert_array_equal(snake._evolution_normals(points, closed),
                                      original_iteration_normals(points, closed))
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize('boundary', list(snake.SnakeBoundaryCondition))
@pytest.mark.parametrize('convergence', [0.0, 100.0])
@pytest.mark.parametrize('resample', [0, 10])
def test_exact_evolution(boundary, convergence, resample, monkeypatch):
    from tests.test_active_contour import make_synthetic_circle_image

    image = make_synthetic_circle_image()
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    cfg = snake.ActiveContourConfig(max_iterations=25, convergence=convergence,
                                    resample_interval=resample)
    kwargs = {'config': cfg, 'boundary_condition': boundary, 'substrate_line': 90.0,
              'pinned_endpoints': (tuple(points[0]), tuple(points[-1]))}
    actual = snake.evolve_active_contour(image, points, **kwargs)
    monkeypatch.setattr(snake, '_evolution_normals', original_iteration_normals)
    expected = snake.evolve_active_contour(image, points, **kwargs)
    for field, value in vars(actual).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(expected, field))
        else:
            assert value == getattr(expected, field)


def test_curvature_only_computed_for_final_result(monkeypatch):
    from tests.test_active_contour import make_synthetic_circle_image

    geometry = snake.compute_contour_normals
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return geometry(*args, **kwargs)

    monkeypatch.setattr(snake, 'compute_contour_normals', counted)
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    result = snake.evolve_active_contour(make_synthetic_circle_image(), points,
        config=snake.ActiveContourConfig(max_iterations=25, convergence=0.0))
    assert result.iterations == 25
    assert len(calls) == 1
