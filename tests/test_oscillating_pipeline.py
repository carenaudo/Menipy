import math

import numpy as np

from menipy.math.rheology import (
    dilational_viscoelasticity,
    harmonic_sine_fit,
    rayleigh_lamb_surface_tension,
)
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.pipelines.oscillating.stages import OscillatingPipeline


def test_harmonic_sine_fit():
    fps = 100.0
    t = np.arange(200) / fps
    f_true = 5.0
    amp_true = 2.5
    y0_true = 10.0
    phi_true = math.radians(45.0)

    y = y0_true + amp_true * np.sin(2.0 * np.pi * f_true * t + phi_true)

    y0_fit, amp_fit, f_fit, phi_fit = harmonic_sine_fit(t, y, frequency_hz=f_true)

    assert math.isclose(y0_fit, y0_true, abs_tol=1e-3)
    assert math.isclose(amp_fit, amp_true, abs_tol=1e-3)
    assert math.isclose(f_fit, f_true, abs_tol=1e-3)


def test_dilational_viscoelasticity():
    A0 = 50.0  # mm^2
    delta_A = 2.5  # 5% strain
    delta_gamma = 5.0  # mN/m
    phase_shift_rad = math.radians(30.0)  # 30 deg
    f0 = 2.0  # Hz

    # |E| = 5.0 / (2.5 / 50.0) = 5.0 / 0.05 = 100.0 mN/m
    # E' = 100 * cos(30) = 86.6025 mN/m
    # E'' = 100 * sin(30) = 50.0 mN/m
    # eta_d = E'' / (2 * pi * 2.0) = 50.0 / (4 * pi) = 3.97887 mN*s/m
    res = dilational_viscoelasticity(
        A0=A0,
        delta_A=delta_A,
        delta_gamma=delta_gamma,
        phase_shift_rad=phase_shift_rad,
        frequency_hz=f0,
    )

    assert math.isclose(res["modulus_E"], 100.0, rel_tol=1e-4)
    assert math.isclose(res["storage_modulus_E_prime"], 100.0 * math.cos(phase_shift_rad), rel_tol=1e-4)
    assert math.isclose(res["loss_modulus_E_double_prime"], 100.0 * math.sin(phase_shift_rad), rel_tol=1e-4)
    assert math.isclose(res["phase_shift_deg"], 30.0, abs_tol=1e-3)
    assert res["dilational_viscosity"] > 0


def test_rayleigh_lamb_surface_tension():
    gamma_expected = 0.0728  # 72.8 mN/m (water)
    r_m = 0.001  # 1 mm radius
    rho1 = 1000.0
    rho2 = 1.2

    # omega_2 = sqrt(24 * gamma / (3 * rho1 * R^3 + 2 * rho2 * R^3))
    omega_2 = math.sqrt((24.0 * gamma_expected) / (3.0 * rho1 * (r_m**3) + 2.0 * rho2 * (r_m**3)))
    f_2 = omega_2 / (2.0 * math.pi)

    gamma_calc = rayleigh_lamb_surface_tension(
        frequency_hz=f_2,
        radius_m=r_m,
        rho_drop_kg_m3=rho1,
        rho_medium_kg_m3=rho2,
        mode_n=2,
    )

    assert math.isclose(gamma_calc, gamma_expected, rel_tol=1e-4)


def test_oscillating_pipeline_execution():
    ctx = Context()
    fps = 100.0
    t = np.arange(150) / fps
    f0 = 4.0
    # Synthetic radius series oscillating between 95 and 105 px around 100 px
    r_series = 100.0 + 5.0 * np.sin(2.0 * np.pi * f0 * t)
    ctx.r_eq_series_px = list(r_series)
    ctx.r0_eq_px = 100.0
    ctx.scale = {"px_per_mm": 50.0}  # 50 px/mm -> R0 = 2.0 mm
    ctx.physics = {"fps": fps, "rho1": 1000.0, "rho2": 1.2}
    ctx.oscillation_phase_deg = 20.0

    # Provide a dummy contour
    theta = np.linspace(0, 2 * np.pi, 50)
    ctx.contour = Contour(xy=np.column_stack([100.0 + 50.0 * np.cos(theta), 100.0 + 50.0 * np.sin(theta)]))

    pipeline = OscillatingPipeline()
    ctx = pipeline.do_geometric_features(ctx)
    ctx = pipeline.do_calibration(ctx)
    ctx = pipeline.do_physics(ctx)
    ctx = pipeline.do_profile_fitting(ctx)
    ctx = pipeline.do_compute_metrics(ctx)

    assert "f0_Hz" in ctx.results
    assert math.isclose(ctx.results["f0_Hz"], f0, abs_tol=0.2)
    assert "gamma_mN_m" in ctx.results
    assert ctx.results["gamma_mN_m"] > 0
    assert "modulus_E" in ctx.results
    assert "storage_modulus_E_prime" in ctx.results
    assert "loss_modulus_E_double_prime" in ctx.results
    assert "dilational_viscosity" in ctx.results
    assert math.isclose(ctx.results["phase_shift_deg"], 20.0, abs_tol=1e-2)
