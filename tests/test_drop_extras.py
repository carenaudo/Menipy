import numpy as np
import pytest

from menipy.models.drop_extras import (
    vmax_uL,
    worthington_number,
    apex_curvature_m_inv,
    surface_area_mm2,
)


def test_worthington_number_unity():
    vmax = vmax_uL(0.072, 1.0, 1000.0)
    wo = worthington_number(vmax, vmax)
    assert pytest.approx(wo, rel=1e-6) == 1.0


def test_apex_curvature():
    kappa = apex_curvature_m_inv(10.0)
    assert pytest.approx(kappa, rel=1e-2) == 200.0


def test_surface_area_sphere():
    R = 2.0
    theta = np.linspace(0, np.pi, 50)
    r = R * np.sin(theta)
    z = R * (1 - np.cos(theta))
    contour_px = np.stack([r * 10.0, z * 10.0], axis=1)  # px, assume 10 px/mm
    area = surface_area_mm2(contour_px, px_per_mm=10.0)
    expected = 4.0 * np.pi * R ** 2
    assert np.isclose(area, expected, rtol=1e-2)
