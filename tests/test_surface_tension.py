"""Tests for test surface tension.

Unit tests."""

import numpy as np
import pytest

from menipy.models.surface_tension import (
    bond_number,
    jennings_pallas_beta,
    surface_tension,
    volume_from_contour,
)


def test_jennings_pallas_beta_edges():
    expected = {0.5: 0.48006475, 2.0: 1.761466}
    for s1, ref in expected.items():
        val = jennings_pallas_beta(s1)
        assert pytest.approx(val, rel=1e-2) == ref


def test_surface_tension_and_bond_number():
    beta = jennings_pallas_beta(1.0)
    gamma = surface_tension(999.0, 9.80665, 2.0, beta)
    bo = bond_number(999.0, 9.80665, 2.0, gamma)
    assert gamma > 0
    assert bo > 0


def test_surface_tension_returns_newtons_per_metre_for_mn_conversion():
    beta = jennings_pallas_beta(1.0)
    gamma_n_m = surface_tension(998.8, 9.80665, 1.0, beta)
    gamma_mn_m = gamma_n_m * 1000.0
    assert gamma_n_m == pytest.approx(0.0135592573)
    assert gamma_mn_m == pytest.approx(13.5592573)


def test_volume_from_contour_circle():
    R = 1.0
    y = np.linspace(0, 2 * R, 100)
    r = np.sqrt(np.maximum(0.0, R**2 - (y - R) ** 2))
    contour = np.stack([r, y], axis=1)
    vol = volume_from_contour(contour)
    expected = 4.0 / 3.0 * np.pi * R**3
    assert np.isclose(vol, expected, rtol=1e-2)
