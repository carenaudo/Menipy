"""Exact adaptive-profile regression for ODE callback reuse."""

from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from menipy.common.cancellation import check_cancelled
from menipy.pipelines.pendant import strict_young_laplace as strict


def legacy_integrate(
    r0_mm: float,
    beta: float,
    *,
    target_height_mm: float | None = None,
    needle_radius_mm: float | None = None,
    max_step: float = 0.02,
    branch: str = "full",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Integrate a symmetric pendant Young-Laplace profile in millimetres."""
    r0_mm = float(r0_mm)
    beta = float(beta)
    if not np.isfinite(r0_mm) or not np.isfinite(beta) or r0_mm <= 0:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "invalid_parameters"}
        return (profile, meta) if return_metadata else profile

    z_target = None
    if target_height_mm is not None and target_height_mm > 0:
        z_target = max(float(target_height_mm) / r0_mm, 0.2)

    r_needle_target = None
    if needle_radius_mm is not None and needle_radius_mm > 0:
        r_needle_target = float(needle_radius_mm) / r0_mm

    def ode(_s: float, y: np.ndarray) -> list[float]:
        check_cancelled()
        r, z, psi = y
        if abs(r) < 1e-10:
            sin_psi_over_r = 1.0
        else:
            sin_psi_over_r = float(np.sin(psi) / r)
        return [
            float(np.cos(psi)),
            float(np.sin(psi)),
            float(2.0 - beta * z - sin_psi_over_r),
        ]

    def hit_axis(s: float, y: np.ndarray) -> float:
        if s <= 0.1:
            return 1.0
        return float(y[0] - 1e-6)

    hit_axis.terminal = True
    hit_axis.direction = -1

    events = [hit_axis]
    if z_target is not None:

        def hit_target_height(_s: float, y: np.ndarray) -> float:
            return float(y[1] - z_target)

        hit_target_height.terminal = True
        hit_target_height.direction = 1
        events.append(hit_target_height)

    if r_needle_target is not None:

        def hit_needle_radius_after_equator(_s: float, y: np.ndarray) -> float:
            r, _z, psi = y
            if psi <= (np.pi / 2.0):
                return 1.0
            return float(r - r_needle_target)

        hit_needle_radius_after_equator.terminal = True
        hit_needle_radius_after_equator.direction = -1
        events.append(hit_needle_radius_after_equator)

    s_max = max(8.0, (z_target or 4.0) * 3.0 + 2.0)
    sol = solve_ivp(
        ode,
        (0.0, s_max),
        [0.0, 0.0, 0.0],
        method="RK45",
        events=events,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success or sol.y.shape[1] < 3:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "solver_failed", "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    stop_reason = "s_max"
    if sol.t_events:
        event_names = ["axis_return"]
        if z_target is not None:
            event_names.append("height_cutoff")
        if r_needle_target is not None:
            event_names.append("needle_radius")
        for name, events_for_name in zip(event_names, sol.t_events):
            check_cancelled()
            if len(events_for_name) > 0:
                stop_reason = name
                break

    r_right = sol.y[0] * r0_mm
    z_right = sol.y[1] * r0_mm
    if branch == "right":
        profile = np.column_stack([r_right, z_right])
        meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    r_left = -r_right[::-1]
    z_left = z_right[::-1]
    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])
    profile = np.column_stack([r_full, z_full])
    meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
    return (profile, meta) if return_metadata else profile


@pytest.mark.parametrize(
    "r0,beta", [(1.2, 0), (1.2, 0.6), (0.05, 4), (5, 0.02), (-1, 0.6)]
)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"target_height_mm": 2.0},
        {"needle_radius_mm": 0.2},
        {"target_height_mm": 2.0, "needle_radius_mm": 0.3, "branch": "right"},
    ],
)
def test_exact_profiles_and_events(r0, beta, options):
    actual, meta = strict.integrate_young_laplace_profile_mm(
        r0, beta, return_metadata=True, **options
    )
    expected, reference_meta = legacy_integrate(
        r0, beta, return_metadata=True, **options
    )
    np.testing.assert_array_equal(actual, expected)
    assert meta == reference_meta


@pytest.mark.parametrize("noise", [0, 0.2, 10])
def test_exact_full_fit(noise, monkeypatch):
    import json

    from tests.test_pendant_fit_cache import fit_input

    inputs = fit_input(noise)
    actual = strict.fit_pendant_young_laplace_strict(inputs)
    monkeypatch.setattr(strict, "integrate_young_laplace_profile_mm", legacy_integrate)
    expected = strict.fit_pendant_young_laplace_strict(inputs)
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
