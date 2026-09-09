"""Exact fit equivalence and cooperative cancellation with shape reuse."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from menipy.common.cancellation import AnalysisCancelled, CancellationToken
from menipy.pipelines.pendant import strict_young_laplace as strict


def fit_input(noise=0.0, refraction=1.0):
    model = strict.integrate_young_laplace_profile_mm(1.2, 0.6, target_height_mm=2.0)
    model[:, 0] *= refraction
    model += np.array([0.025, 0.015])
    contour = strict.model_mm_to_pendant_px(
        model, axis_x_px=250, apex_y_px=300, px_per_mm=100
    )
    contour += np.random.default_rng(42).normal(0, noise, contour.shape)
    return strict.PendantStrictFitInput(
        contour_px=contour,
        axis_x_px=250,
        apex_y_px=300,
        px_per_mm=100,
        r0_seed_mm=1.1,
        beta_seed=0.5,
        physics={
            "rho1": 1000,
            "rho2": 1.2,
            "g": 9.80665,
            "refraction_index_liquid": refraction,
        },
    )


@pytest.mark.parametrize("noise,refraction", [(0, 1), (0.2, 1), (0.2, 1.33), (10, 1)])
def test_cached_and_uncached_fit_are_exactly_equal(noise, refraction):
    inputs = fit_input(noise, refraction)
    integrator = strict.integrate_young_laplace_profile_mm
    with patch.object(
        strict, "integrate_young_laplace_profile_mm", wraps=integrator
    ) as calls:
        cached = strict.fit_pendant_young_laplace_strict(inputs)
        cached_count = calls.call_count
    with patch.object(strict, "lru_cache", lambda **kwargs: lambda function: function):
        with patch.object(
            strict, "integrate_young_laplace_profile_mm", wraps=integrator
        ) as calls:
            uncached = strict.fit_pendant_young_laplace_strict(inputs)
            uncached_count = calls.call_count
    assert json.dumps(cached, sort_keys=True) == json.dumps(uncached, sort_keys=True)
    assert cached_count < uncached_count


def test_cancel_before_offset_only_cache_hit(monkeypatch):
    inputs = fit_input()
    token = CancellationToken()

    def optimize(residuals, *, x0, **kwargs):
        residuals(x0)
        token.cancel()
        offset = x0.copy()
        offset[2] += 1e-8
        residuals(offset)
        pytest.fail("Cancellation was ignored on a reused physical shape")

    monkeypatch.setattr(strict, "least_squares", optimize)
    with patch.object(
        strict,
        "integrate_young_laplace_profile_mm",
        wraps=strict.integrate_young_laplace_profile_mm,
    ) as calls:
        with pytest.raises(AnalysisCancelled):
            strict.fit_pendant_young_laplace_strict(inputs, check_cancelled=token.check)
        assert calls.call_count == 1


def test_cache_preserves_optimizer_failure(monkeypatch):
    inputs = fit_input(0.2)
    optimize = strict.least_squares

    def stop_early(*args, **kwargs):
        kwargs["max_nfev"] = 1
        return optimize(*args, **kwargs)

    monkeypatch.setattr(strict, "least_squares", stop_early)
    cached = strict.fit_pendant_young_laplace_strict(inputs)
    with patch.object(strict, "lru_cache", lambda **kwargs: lambda fn: fn):
        uncached = strict.fit_pendant_young_laplace_strict(inputs)
    assert not cached["strict_fit_success"]
    assert cached["strict_fit_warning"] == "optimizer_failed"
    assert json.dumps(cached, sort_keys=True) == json.dumps(uncached, sort_keys=True)
