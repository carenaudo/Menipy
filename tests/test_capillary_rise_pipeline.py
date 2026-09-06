import math

import numpy as np

from menipy.math.jurin import (
    jurin_contact_angle,
    jurin_surface_tension,
    rayleigh_corrected_capillary_height,
    wilhelmy_meniscus_contact_angle,
)
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.pipelines.capillary_rise.stages import CapillaryRisePipeline


def test_jurin_surface_tension_and_inversion():
    delta_rho = 998.2
    g = 9.80665
    r_tube = 0.0005  # 0.5 mm radius
    theta_rad = math.radians(15.0)

    gamma_expected = 0.0728
    h_m = (2.0 * gamma_expected * math.cos(theta_rad)) / (delta_rho * g * r_tube)

    gamma_calc = jurin_surface_tension(
        h_m=h_m,
        rho_kg_m3=delta_rho,
        g=g,
        tube_radius_m=r_tube,
        contact_angle_rad=theta_rad,
    )
    assert math.isclose(gamma_calc, gamma_expected, rel_tol=1e-5)

    theta_calc = jurin_contact_angle(
        h_m=h_m,
        gamma_n_m=gamma_expected,
        rho_kg_m3=delta_rho,
        g=g,
        tube_radius_m=r_tube,
    )
    assert math.isclose(theta_calc, theta_rad, rel_tol=1e-5)


def test_rayleigh_corrected_capillary_height():
    h_m = 0.030
    r_m = 0.001

    h_eff = rayleigh_corrected_capillary_height(h_m, r_m)
    assert h_eff > h_m
    assert math.isclose(h_eff, 0.03033, abs_tol=1e-4)


def test_wilhelmy_meniscus_contact_angle():
    gamma = 0.0728
    delta_rho = 998.2
    g = 9.80665
    l_c = math.sqrt(gamma / (delta_rho * g))
    theta_deg = 30.0
    theta_rad = math.radians(theta_deg)

    h_m = math.sqrt(2.0) * l_c * math.sqrt(1.0 - math.sin(theta_rad))

    theta_calc = wilhelmy_meniscus_contact_angle(h_m, gamma, delta_rho, g)
    assert math.isclose(math.degrees(theta_calc), theta_deg, abs_tol=1e-3)


def test_capillary_rise_pipeline_execution():
    ctx = Context()
    xs = np.linspace(100, 200, 50)
    ys = 150.0 + 20.0 * ((xs - 150.0) / 50.0) ** 2
    baseline_pts = np.array([[200, 450], [100, 450]])
    contour_pts = np.vstack([np.column_stack([xs, ys]), baseline_pts])

    ctx.contour = Contour(xy=contour_pts)
    ctx.scale = {"px_per_mm": 10.0}
    ctx.tube_diameter_mm = 2.0
    ctx.physics = {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
    ctx.contact_angle_deg = 0.0

    pipeline = CapillaryRisePipeline()
    ctx = pipeline.do_geometric_features(ctx)
    ctx = pipeline.do_calibration(ctx)
    ctx = pipeline.do_physics(ctx)
    ctx = pipeline.do_compute_metrics(ctx)

    assert "h_mm" in ctx.results
    assert math.isclose(ctx.results["h_mm"], 30.0, abs_tol=0.1)
    assert "r_tube_mm" in ctx.results
    assert math.isclose(ctx.results["r_tube_mm"], 1.0, abs_tol=0.01)
    assert "gamma_mN_m" in ctx.results
    assert ctx.results["gamma_mN_m"] > 0
