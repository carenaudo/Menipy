"""Exact cold-table construction with bounded, cancellable derivative reuse."""

import numpy as np
import pytest

from menipy.common.cancellation import (
    AnalysisCancelled,
    CancellationToken,
    cancellation_scope,
)
from menipy.pipelines.pendant import approximations as approx
from menipy.pipelines.pendant import strict_young_laplace as strict


def uncached_integrate(*args, **kwargs):
    kwargs.pop("_derivative_cache", None)
    return strict.integrate_young_laplace_profile_mm(*args, **kwargs)


def assert_tables_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for plane in actual:
        for left, right in zip(actual[plane], expected[plane]):
            np.testing.assert_array_equal(left, right)


def test_complete_grid_exact(monkeypatch):
    actual = approx._build_selected_plane_lookup()
    monkeypatch.setattr(approx, "integrate_young_laplace_profile_mm", uncached_integrate)
    expected = approx._build_selected_plane_lookup()
    assert_tables_equal(actual, expected)


@pytest.mark.parametrize("beta", [0.03, 0.8, 2.5])
def test_profiles_and_events_exact(beta):
    cache = {}
    for height in [0.8, 1.8, 4.5]:
        kwargs = {"target_height_mm": height, "needle_radius_mm": 0.3,
                  "max_step": 0.08, "return_metadata": True}
        expected, expected_meta = strict.integrate_young_laplace_profile_mm(1, beta, **kwargs)
        actual, meta = strict.integrate_young_laplace_profile_mm(1, beta, _derivative_cache=cache, **kwargs)
        np.testing.assert_array_equal(actual, expected)
        assert meta == expected_meta
        assert len(cache) <= 16384


def test_bound_and_cancellation_on_hit(monkeypatch):
    token = CancellationToken()
    cache = {}
    original = strict.solve_ivp

    def inspect(fun, *args, **kwargs):
        for i in range(16400):
            fun(0, np.array([1.0, i / 10000.0, 0.2]))
        assert len(cache) == 16384
        state = np.array([1.0, 0.2, 0.3])
        fun(0, state)
        token.cancel()
        with pytest.raises(AnalysisCancelled):
            fun(0, state)
        raise AnalysisCancelled()

    monkeypatch.setattr(strict, "solve_ivp", inspect)
    with pytest.raises(AnalysisCancelled), cancellation_scope(token):
        strict.integrate_young_laplace_profile_mm(1, 0.3, _derivative_cache=cache)
    monkeypatch.setattr(strict, "solve_ivp", original)


def test_beta_identity_is_part_of_key():
    cache = {}
    for beta in [0.3, 0.4, 0.3]:
        actual = strict.integrate_young_laplace_profile_mm(1, beta, _derivative_cache=cache)
        expected = strict.integrate_young_laplace_profile_mm(1, beta)
        np.testing.assert_array_equal(actual, expected)
