"""Unit tests for the probe liquid library (menipy.common.liquid_db)."""

from __future__ import annotations

import pytest

from menipy.common.liquid_db import (
    ProbeLiquid,
    format_liquid_table,
    get_builtin_liquids,
    get_liquid,
    list_liquid_names,
)


def test_builtin_library_integrity():
    """All built-in liquids must satisfy physical consistency rules."""
    liquids = get_builtin_liquids()
    assert len(liquids) >= 15, "Expected at least 15 probe liquid records"

    for liq in liquids:
        assert isinstance(liq, ProbeLiquid)
        assert liq.name
        assert liq.formula
        assert liq.cas
        assert liq.gamma_total > 0.0
        assert liq.gamma_d >= 0.0
        assert liq.gamma_p >= 0.0
        # Dispersive + polar should match total within 0.15 mN/m
        assert abs(liq.gamma_total - (liq.gamma_d + liq.gamma_p)) <= 0.15
        assert liq.temperature_c in (20.0, 25.0)
        assert liq.reference
        # Validation method should produce no warnings for built-ins
        warnings = liq.validate()
        assert not warnings, f"Validation warning for {liq.name}: {warnings}"


def test_owrk_x_coordinate():
    """OWRK x-coordinate √(γ_p / γ_d) must match analytical expectation."""
    water = get_liquid("water", 20.0)
    assert water is not None
    expected_water_x = (51.0 / 21.8) ** 0.5
    assert pytest.approx(water.owrk_x, rel=1e-3) == expected_water_x

    diiodo = get_liquid("diiodomethane", 20.0)
    assert diiodo is not None
    assert diiodo.owrk_x == 0.0  # purely dispersive


def test_get_liquid_name_variations():
    """Lookups should handle case, hyphens, and underscores robustly."""
    water1 = get_liquid("Water")
    water2 = get_liquid("water")
    water3 = get_liquid("WATER")
    assert water1 == water2 == water3

    eg1 = get_liquid("ethylene_glycol")
    eg2 = get_liquid("ethylene glycol")
    eg3 = get_liquid("ethylene-glycol")
    assert eg1 == eg2 == eg3
    assert eg1 is not None and eg1.name == "Ethylene glycol"

    # Non-existent liquid
    assert get_liquid("non_existent_chemical") is None


def test_temperature_lookup():
    """Liquid lookups must return correct temperature variant."""
    w20 = get_liquid("Water", 20.0)
    w25 = get_liquid("Water", 25.0)
    assert w20 is not None and w25 is not None
    assert w20.temperature_c == 20.0
    assert w25.temperature_c == 25.0
    assert w20.gamma_total == 72.8
    assert w25.gamma_total == 72.0

    d20 = get_liquid("Diiodomethane", 20.0)
    d25 = get_liquid("Diiodomethane", 25.0)
    assert d20 is not None and d25 is not None
    assert d20.gamma_total == 50.8
    assert d25.gamma_total == 50.0


def test_list_liquid_names():
    """Unique liquid names must include key probe liquids."""
    names = list_liquid_names()
    assert "Water" in names
    assert "Diiodomethane" in names
    assert "Glycerol" in names
    assert "Ethylene glycol" in names
    assert "Formamide" in names
    assert len(names) >= 10


def test_format_liquid_table():
    """Table formatting must produce readable output without exceptions."""
    all_table = format_liquid_table()
    assert "Water" in all_table
    assert "Diiodomethane" in all_table

    t20_table = format_liquid_table(temperature_c=20.0)
    assert "Water" in t20_table
    assert "72.8" in t20_table
