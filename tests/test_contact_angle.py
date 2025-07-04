import pytest
import numpy as np

from src import contact_angle as ca


def test_height_width_ca():
    angle = ca.height_width_ca(1.0, 2.0)
    assert angle == pytest.approx(90.0, rel=1e-6)


def test_circle_fit_ca():
    angle = ca.circle_fit_ca(1.0, 2.0)
    assert angle == pytest.approx(30.0, rel=1e-6)


def test_tangent_line_ca():
    angle = ca.tangent_line_ca(1.0)
    assert angle == pytest.approx(45.0, rel=1e-6)


def test_spline_derivative_ca():
    angle = ca.spline_derivative_ca(1.0)
    assert angle == pytest.approx(45.0, rel=1e-6)


def test_derived_parameters():
    r_b = 1.0
    assert ca.footprint_area_mm2(r_b) == pytest.approx(np.pi, rel=1e-6)

    contour = np.array([[1.0, 0.0], [1.0, 2.0]])
    vol = ca.drop_volume_uL(contour)
    assert vol == pytest.approx(2 * np.pi, rel=1e-3)

    area = ca.surface_area_mm2(contour)
    assert area == pytest.approx(4 * np.pi, rel=1e-3)

    kappa = ca.apex_curvature_m_inv(10.0)
    assert kappa == pytest.approx(200.0, rel=1e-6)

    bo = ca.bond_number(1000.0, 9.81, 1.0, 0.072)
    assert bo == pytest.approx(0.13625, rel=1e-3)

    weight = ca.apparent_weight_mN(10.0, 1000.0)
    assert weight == pytest.approx(0.0981, rel=1e-3)
