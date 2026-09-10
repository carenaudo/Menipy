"""Exact matrix and complete evolution equivalence for repeated snake settings."""

import numpy as np
import pytest

from menipy.math import active_contour as snake


def uncached_inverse(n, config, boundary):
    matrix = snake.build_pentadiagonal_matrix(n, config.alpha, config.beta, boundary)
    return np.linalg.inv(matrix + config.gamma * np.eye(n, dtype=float))


@pytest.mark.parametrize("boundary", list(snake.SnakeBoundaryCondition))
@pytest.mark.parametrize("n", [20, 80, 300, 301])
def test_exact_matrix_and_bound(n, boundary):
    snake._cached_shape_inverse.cache_clear()
    cfg = snake.ActiveContourConfig()
    actual = snake._shape_inverse(n, cfg, boundary)
    np.testing.assert_array_equal(actual, uncached_inverse(n, cfg, boundary))
    if n <= 300:
        assert actual is snake._shape_inverse(n, cfg, boundary)
        assert not actual.flags.writeable
    else:
        assert snake._cached_shape_inverse.cache_info().currsize == 0


@pytest.mark.parametrize("boundary", list(snake.SnakeBoundaryCondition))
def test_exact_evolution(boundary, monkeypatch):
    from tests.test_active_contour import make_synthetic_circle_image

    image = make_synthetic_circle_image()
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
    cfg = snake.ActiveContourConfig(max_iterations=15)
    kwargs = {
        "config": cfg,
        "boundary_condition": boundary,
        "substrate_line": 90.0,
        "pinned_endpoints": (tuple(points[0]), tuple(points[-1])),
    }
    actual = snake.evolve_active_contour(image, points, **kwargs)
    monkeypatch.setattr(snake, "_shape_inverse", uncached_inverse)
    expected = snake.evolve_active_contour(image, points, **kwargs)
    for name, value in vars(actual).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(expected, name))
        else:
            assert value == getattr(expected, name)


def test_settings_and_builder_changes_invalidate(monkeypatch):
    snake._cached_shape_inverse.cache_clear()
    cfg = snake.ActiveContourConfig()
    boundary = snake.SnakeBoundaryCondition.PERIODIC
    initial = snake._shape_inverse(20, cfg, boundary)
    for index in range(12):
        cfg.alpha = 0.01 + index * 0.001
        np.testing.assert_array_equal(
            snake._shape_inverse(20, cfg, boundary), uncached_inverse(20, cfg, boundary)
        )
    assert snake._cached_shape_inverse.cache_info().currsize == 8
    builder = snake.build_pentadiagonal_matrix
    monkeypatch.setattr(
        snake, "build_pentadiagonal_matrix", lambda *args: builder(*args) * 2
    )
    assert not np.array_equal(snake._shape_inverse(20, cfg, boundary), initial)
    np.testing.assert_array_equal(
        snake._shape_inverse(20, cfg, boundary), uncached_inverse(20, cfg, boundary)
    )


@pytest.mark.parametrize("field", ["alpha", "beta", "gamma"])
def test_each_coefficient_invalidates(field):
    snake._cached_shape_inverse.cache_clear()
    config = snake.ActiveContourConfig()
    boundary = snake.SnakeBoundaryCondition.PERIODIC
    before = snake._shape_inverse(20, config, boundary)
    setattr(config, field, getattr(config, field) * 2)
    after = snake._shape_inverse(20, config, boundary)
    assert after is not before
    np.testing.assert_array_equal(after, uncached_inverse(20, config, boundary))


def test_singular_failure_is_not_cached():
    snake._cached_shape_inverse.cache_clear()
    config = snake.ActiveContourConfig(alpha=0, beta=0, gamma=0)
    with pytest.raises(np.linalg.LinAlgError):
        snake._shape_inverse(20, config, snake.SnakeBoundaryCondition.PERIODIC)
    assert snake._cached_shape_inverse.cache_info().currsize == 0
