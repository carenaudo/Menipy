import numpy as np
import pytest

from src.models.surface_tension import (
    jennings_pallas_beta,
    surface_tension,
    bond_number,
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


def test_volume_from_contour_circle():
    R = 1.0
    y = np.linspace(0, 2 * R, 100)
    r = np.sqrt(np.maximum(0.0, R**2 - (y - R) ** 2))
    contour = np.stack([r, y], axis=1)
    vol = volume_from_contour(contour)
    expected = 4.0 / 3.0 * np.pi * R**3
    assert np.isclose(vol, expected, rtol=1e-2)
