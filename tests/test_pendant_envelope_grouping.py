"""Exact reference comparisons for grouped pendant envelope construction."""

import numpy as np
import pytest

from menipy.common.cancellation import check_cancelled
from menipy.pipelines.pendant import strict_young_laplace as strict
from menipy.pipelines.pendant.strict_young_laplace import _axis_frame


def legacy_envelope(
    contour_px: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
    bin_px: float = 1.0,
    axis_origin_px: tuple[float, float] | None = None,
    axis_direction_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Collapse a pendant contour into a radial ``(r_mm, z_mm)`` envelope."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if xy.shape[0] < 3 or px_per_mm <= 0:
        return np.empty((0, 2), dtype=float)

    bin_px = max(float(bin_px), 1.0)
    origin, basis = _axis_frame(
        axis_x_px=axis_x_px,
        apex_y_px=apex_y_px,
        axis_origin_px=axis_origin_px,
        axis_direction_xy=axis_direction_xy,
    )
    local = (xy - origin) @ basis
    row_keys = np.rint(local[:, 1] / bin_px).astype(int)
    rows: list[tuple[float, float]] = []
    for row in np.unique(row_keys):
        check_cancelled()
        pts = xy[row_keys == row]
        if pts.size == 0:
            continue
        local_pts = local[row_keys == row]
        z_mm = float(np.mean(local_pts[:, 1])) / float(px_per_mm)
        if z_mm < -0.5 / float(px_per_mm):
            continue
        xs = local_pts[:, 0]
        if xs.size >= 2:
            r_mm = (float(np.max(xs)) - float(np.min(xs))) / (2.0 * px_per_mm)
        else:
            r_mm = float(np.max(np.abs(xs))) / float(px_per_mm)
        if np.isfinite(z_mm) and np.isfinite(r_mm) and r_mm >= 0:
            rows.append((r_mm, max(0.0, z_mm)))

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 1])]
    merged: list[tuple[float, float]] = []
    for z in np.unique(arr[:, 1]):
        check_cancelled()
        r = float(np.max(arr[arr[:, 1] == z, 0]))
        merged.append((r, float(z)))
    profile = np.asarray(merged, dtype=float)
    if profile.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    if profile[0, 1] > 1e-9:
        profile = np.vstack([[0.0, 0.0], profile])
    else:
        profile[0, 1] = 0.0
        profile[0, 0] = min(
            profile[0, 0], profile[1, 0] if profile.shape[0] > 1 else 0.0
        )
    return profile


@pytest.mark.parametrize("count", [0, 2, 3, 100, 1000, 10000])
@pytest.mark.parametrize("direction", [None, (0.3, -0.9), (0.0, 0.0)])
def test_exact_envelope(count, direction):
    rng = np.random.default_rng(42)
    contour = rng.uniform([-100, -1500], [100, 2], (count, 2))
    original = contour.copy()
    for bin_px in (1, 3.7):
        kwargs = {
            "axis_x_px": 0,
            "apex_y_px": 0,
            "px_per_mm": 100,
            "axis_direction_xy": direction,
            "bin_px": bin_px,
        }
        np.testing.assert_array_equal(
            strict.build_pendant_profile_envelope_mm(contour, **kwargs),
            legacy_envelope(contour, **kwargs),
        )
    np.testing.assert_array_equal(contour, original)


def test_clamped_and_singleton_rows():
    contour = np.array([[1, 0.4], [-2, 0.3], [3, -0.6], [-5, -0.7], [6, -2], [7, -3]])
    kwargs = {"axis_x_px": 0, "apex_y_px": 0, "px_per_mm": 100}
    for points in (contour, contour[::-1], np.repeat(contour, 4, axis=0)):
        np.testing.assert_array_equal(
            strict.build_pendant_profile_envelope_mm(points, **kwargs),
            legacy_envelope(points, **kwargs),
        )


def test_cancel_during_group_processing(monkeypatch):
    from menipy.common.cancellation import AnalysisCancelled

    calls = []

    def cancel():
        calls.append(True)
        if len(calls) == 3:
            raise AnalysisCancelled()

    monkeypatch.setattr(strict, "check_cancelled", cancel)
    with pytest.raises(AnalysisCancelled):
        strict.build_pendant_profile_envelope_mm(
            np.array([[0, 0], [1, -1], [2, -2]]),
            axis_x_px=0,
            apex_y_px=0,
            px_per_mm=100,
        )
    assert len(calls) == 3


@pytest.mark.parametrize("noise", [0, 0.2, 10])
def test_full_fit_exact(noise, monkeypatch):
    import json

    from tests.test_pendant_fit_cache import fit_input

    inputs = fit_input(noise)
    actual = strict.fit_pendant_young_laplace_strict(inputs)
    monkeypatch.setattr(strict, "build_pendant_profile_envelope_mm", legacy_envelope)
    expected = strict.fit_pendant_young_laplace_strict(inputs)
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
