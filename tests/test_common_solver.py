"""Tests for common solver residual helpers."""

import numpy as np

from menipy.common.solver import _residuals_normal_projection


def _legacy_normal_projection(obs_xy: np.ndarray, model_xy: np.ndarray) -> np.ndarray:
    """Reference copy of the previous linear-scan nearest-neighbor residual."""
    t = np.gradient(model_xy, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n = np.column_stack([-t[:, 1], t[:, 0]])

    res = []
    for p in obs_xy:
        d = np.sum((model_xy - p) ** 2, axis=1)
        j = int(np.argmin(d))
        res.append(np.dot((p - model_xy[j]), n[j]))
    return np.asarray(res, dtype=float)


def test_normal_projection_residuals_match_legacy_nearest_neighbor():
    theta = np.linspace(0.0, np.pi, 41)
    model_xy = np.column_stack([np.cos(theta), np.sin(theta)])
    obs_xy = model_xy[::4] + np.array([0.025, -0.015])

    residuals = _residuals_normal_projection(obs_xy, model_xy)
    legacy_residuals = _legacy_normal_projection(obs_xy, model_xy)

    assert residuals.shape == (len(obs_xy),)
    np.testing.assert_allclose(residuals, legacy_residuals, rtol=1e-12, atol=1e-12)
