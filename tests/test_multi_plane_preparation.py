"""Exact comparison of shared-profile multi-plane estimation."""

from typing import Any

import numpy as np
import pytest

from menipy.pipelines.pendant import approximations as approx
from menipy.pipelines.pendant.approximations import (
    DEFAULT_SELECTED_PLANES,
    _beta_from_gamma,
    _physics,
    _r0_mm_from_context,
    _selected_plane_estimate,
)


def legacy_multi(
    ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]
) -> dict[str, Any]:
    """Approximate IFT from a median over multiple selected planes."""
    estimates = []
    plane_rows = []
    for k in DEFAULT_SELECTED_PLANES:
        raw = _selected_plane_estimate(ctx, profile_mm, physics, k=k)
        gamma = raw.get("approx_selected_plane_surface_tension_mN_m")
        status = raw.get("approx_selected_plane_status")
        plane_rows.append(
            {
                "k": k,
                "status": status,
                "surface_tension_mN_m": gamma,
                "s": raw.get("approx_selected_plane_s"),
            }
        )
        if status == "ok" and gamma is not None and np.isfinite(float(gamma)):
            estimates.append(float(gamma))

    prefix = "approx_multi_selected_plane"
    out: dict[str, Any] = {
        f"{prefix}_planes": plane_rows,
        f"{prefix}_n": len(estimates),
    }
    if not estimates:
        out[f"{prefix}_status"] = "no_valid_planes"
        return out

    gamma_mn_m = float(np.median(estimates))
    delta_rho, g, _rho1 = _physics(physics)
    r0_mm = _r0_mm_from_context(ctx)
    beta = _beta_from_gamma(delta_rho, g, r0_mm, gamma_mn_m / 1000.0)
    out.update(
        {
            f"{prefix}_status": "ok",
            f"{prefix}_surface_tension_mN_m": gamma_mn_m,
            f"{prefix}_std_mN_m": float(np.std(estimates)),
            f"{prefix}_beta": beta,
        }
    )
    return out


@pytest.mark.parametrize(
    "kind", ["normal", "reverse", "duplicates", "invalid", "short", "empty"]
)
def test_exact_estimates(kind, monkeypatch):
    from types import SimpleNamespace

    z = np.linspace(0, 3, 300)
    profile = np.column_stack([np.sin(z), z])
    if kind == "reverse":
        profile = profile[::-1]
    elif kind == "duplicates":
        profile = np.repeat(profile, 2, axis=0)
    elif kind == "invalid":
        profile = np.vstack([profile, [np.nan, 1], [1, np.inf], [-1, 1]])
    elif kind == "short":
        profile = profile[:2]
    elif kind == "empty":
        profile = np.empty((0, 2))
    # Deterministic lookup: exercise successful and unavailable-plane paths.
    monkeypatch.setattr(approx, "_lookup_h_from_s", lambda k, s, h: (0.6, "ok"))
    ctx = SimpleNamespace(results={"r0_mm": 1.2})
    expected = legacy_multi(ctx, profile, {})
    original = approx._profile
    calls = []

    def prepare(points):
        calls.append(True)
        return original(points)

    monkeypatch.setattr(approx, "_profile", prepare)
    assert approx.multi_selected_plane(ctx, profile, {}) == expected
    assert len(calls) == 1


def test_real_lookup_exact():
    from types import SimpleNamespace

    from menipy.pipelines.pendant.strict_young_laplace import (
        integrate_young_laplace_profile_mm,
    )

    profile = integrate_young_laplace_profile_mm(1.2, 0.6, target_height_mm=2.0)
    ctx = SimpleNamespace(results={"r0_mm": 1.2})
    assert approx.multi_selected_plane(ctx, profile, {}) == legacy_multi(
        ctx, profile, {}
    )
