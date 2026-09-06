"""Unit tests for surface free energy algorithms (OWRK & Wu)."""

from __future__ import annotations

import pytest

from menipy.common.liquid_db import ProbeLiquid, get_liquid
from menipy.math.surface_energy import (
    OWRKResult,
    WuResult,
    compute_owrk,
    compute_surface_energy,
    compute_wu,
    validate_sfe_inputs,
)


@pytest.fixture
def standard_liquids() -> list[ProbeLiquid]:
    """Water, Diiodomethane, and Glycerol at 20 °C."""
    w = get_liquid("water", 20.0)
    d = get_liquid("diiodomethane", 20.0)
    g = get_liquid("glycerol", 20.0)
    assert w is not None and d is not None and g is not None
    return [w, d, g]


def test_owrk_three_liquids_standard(standard_liquids):
    """OWRK regression with 3 standard liquids on realistic test angles."""
    # Angles representative of PMMA / treated polymer:
    # Water: 65.3°, Diiodomethane: 42.1°, Glycerol: 58.0°
    angles = [65.3, 42.1, 58.0]
    res = compute_owrk(standard_liquids, angles)

    assert isinstance(res, OWRKResult)
    assert res.gamma_s_d > 0.0
    assert res.gamma_s_p > 0.0
    assert pytest.approx(res.gamma_s_total, rel=1e-4) == res.gamma_s_d + res.gamma_s_p
    assert res.r_squared is not None
    assert 0.90 <= res.r_squared <= 1.0  # highly linear for consistent simulated substrate
    assert len(res.data_points if hasattr(res, "data_points") else res.x_coords) == 3
    assert len(res.residuals) == 3
    assert res.se_slope is not None
    assert res.se_intercept is not None
    assert not res.warnings


def test_owrk_two_liquids(standard_liquids):
    """OWRK with exactly 2 liquids should compute SFE with r_squared=None."""
    two_liquids = standard_liquids[:2]
    angles = [65.3, 42.1]
    res = compute_owrk(two_liquids, angles)

    assert res.gamma_s_d > 0.0
    assert res.gamma_s_p > 0.0
    assert res.r_squared is None  # 0 degrees of freedom
    assert res.se_slope is None


def test_owrk_negative_slope_constraint(standard_liquids):
    """Superhydrophobic/nonpolar scenario yielding negative slope is constrained."""
    # Water contact angle very high, diiodomethane low -> negative slope
    angles = [120.0, 30.0, 115.0]
    res = compute_owrk(standard_liquids, angles)

    # Negative slope constrained: polar component set to 0
    assert res.gamma_s_p == 0.0
    assert res.slope == 0.0
    assert any("negative owrk slope" in w.lower() for w in res.warnings)


def test_owrk_invalid_inputs(standard_liquids):
    """Invalid input configurations must raise ValueError."""
    # Less than 2 liquids
    with pytest.raises(ValueError, match="at least 2"):
        compute_owrk(standard_liquids[:1], [60.0])

    # Mismatch in lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_owrk(standard_liquids, [60.0, 45.0])


def test_wu_harmonic_mean(standard_liquids):
    """Wu harmonic mean method with 2 liquids."""
    two_liquids = standard_liquids[:2]
    angles = [65.3, 42.1]
    res = compute_wu(two_liquids, angles)

    assert isinstance(res, WuResult)
    assert res.gamma_s_d > 0.0
    assert res.gamma_s_p > 0.0
    assert pytest.approx(res.gamma_s_total, rel=1e-4) == res.gamma_s_d + res.gamma_s_p


def test_wu_more_than_two_liquids(standard_liquids):
    """Wu with 3 liquids emits warning and uses first 2."""
    angles = [65.3, 42.1, 58.0]
    res = compute_wu(standard_liquids, angles)

    assert any("uses exactly 2 liquids" in w for w in res.warnings)
    assert len(res.liquid_names) == 2


def test_validate_sfe_inputs(standard_liquids):
    """Input validator detects out-of-range angles, duplicates, and polarity gaps."""
    # Valid inputs -> no warnings
    assert not validate_sfe_inputs(standard_liquids, [65.0, 40.0, 55.0])

    # Out of range angle
    w_bad_angle = validate_sfe_inputs(standard_liquids, [195.0, 40.0, 55.0])
    assert any("outside valid range" in w for w in w_bad_angle)

    # Duplicate liquid
    dup_liquids = [standard_liquids[0], standard_liquids[0]]
    w_dup = validate_sfe_inputs(dup_liquids, [65.0, 60.0])
    assert any("duplicate liquid" in w.lower() for w in w_dup)


def test_compute_surface_energy_facade(standard_liquids):
    """Combined facade method computes both OWRK and Wu and serializes cleanly."""
    angles = [65.3, 42.1, 58.0]
    sfe = compute_surface_energy(
        standard_liquids, angles, method="both", substrate_name="PET_Sample"
    )

    assert sfe.owrk is not None
    assert sfe.wu is not None
    assert sfe.substrate_name == "PET_Sample"

    d = sfe.to_dict()
    assert d["substrate"] == "PET_Sample"
    assert "owrk" in d
    assert "wu" in d
    assert d["owrk"]["gamma_s_total_mN_m"] > 0.0
    assert d["wu"]["gamma_s_total_mN_m"] > 0.0
