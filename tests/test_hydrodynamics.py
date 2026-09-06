"""Unit and physical validation tests for hydrodynamics and tilting plate models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from menipy.math.hydrodynamics import (
    analyze_tilting_plate,
    cox_voinov_extrapolation,
    furmidge_retention_force,
)
from menipy.models.temporal import TemporalFrameResult


def test_cox_voinov_recovers_known_static_contact_angle():
    """Verify Cox-Voinov extrapolation recovers exact static angle from moving contact line."""
    theta_0_true_deg = 65.0
    theta_0_rad = math.radians(theta_0_true_deg)
    k_true = 0.05  # rad^3 / (mm/s)

    velocities = np.linspace(0.05, 1.0, 15)
    theta_cubed = theta_0_rad**3 + k_true * velocities
    angles_deg = np.degrees(theta_cubed ** (1.0 / 3.0))

    # Add small noise
    rng = np.random.default_rng(42)
    noise_deg = rng.normal(0, 0.05, len(angles_deg))
    angles_noisy = angles_deg + noise_deg

    res = cox_voinov_extrapolation(
        velocities,
        angles_noisy,
        viscosity_pa_s=0.001,  # water: 1 mPa*s
        surface_tension_mN_m=72.8,  # water: 72.8 mN/m
    )

    assert res["theta_0_deg"] == pytest.approx(theta_0_true_deg, abs=0.2)
    assert res["r_squared"] > 0.95
    assert res["n_points"] == 15
    assert res["ln_L_over_lm"] is not None
    assert res["ln_L_over_lm"] > 0
    assert res["mean_capillary_number"] is not None
    assert res["mean_capillary_number"] > 0


def test_cox_voinov_insufficient_velocity_variation():
    """Verify graceful handling when contact line is pinned or has zero velocity variation."""
    velocities = [0.0, 0.0, 0.0, 0.0]
    angles = [90.0, 90.0, 90.0, 90.0]

    res = cox_voinov_extrapolation(velocities, angles)
    assert res["theta_0_deg"] == 90.0
    assert res["r_squared"] == 0.0
    assert res["ln_L_over_lm"] is None


def test_furmidge_retention_force_analytical():
    """Verify Furmidge equation matches analytical retention force and sliding angle."""
    gamma = 72.8  # mN/m
    theta_adv = 115.0  # degrees
    theta_rec = 95.0  # degrees
    width = 2.5  # mm
    droplet_mass_mg = 20.0  # 20 mg (~20 uL water drop)

    res = furmidge_retention_force(
        theta_adv_deg=theta_adv,
        theta_rec_deg=theta_rec,
        contact_width_mm=width,
        surface_tension_mN_m=gamma,
        droplet_mass_mg=droplet_mass_mg,
    )

    cos_R = math.cos(math.radians(95.0))
    cos_A = math.cos(math.radians(115.0))
    expected_cos_diff = cos_R - cos_A
    expected_f_line = gamma * expected_cos_diff
    expected_f_total_uN = gamma * width * expected_cos_diff
    expected_sin_alpha = expected_f_total_uN / (droplet_mass_mg * 9.80665)
    expected_alpha_deg = math.degrees(math.asin(expected_sin_alpha))

    assert res["cos_diff"] == pytest.approx(expected_cos_diff, rel=1e-5)
    assert res["retention_force_per_width_mN_m"] == pytest.approx(expected_f_line, rel=1e-5)
    assert res["total_retention_force_uN"] == pytest.approx(expected_f_total_uN, rel=1e-5)
    assert res["critical_sliding_angle_deg"] == pytest.approx(expected_alpha_deg, rel=1e-4)


def test_analyze_tilting_plate_detects_sliding_onset():
    """Verify tilting plate analysis identifies sliding onset frame and critical angle."""
    frames = []
    total_frames = 25
    # Substrate tilts from 0 deg to 24 deg (1 deg per frame)
    for i in range(total_frames):
        tilt_rad = math.radians(float(i))
        # Line from (0, 100) along tilt
        x1, y1 = 0.0, 100.0
        x2, y2 = 200.0 * math.cos(tilt_rad), 100.0 + 200.0 * math.sin(tilt_rad)

        # Droplet remains pinned until frame 10 (10 deg), then starts sliding
        state = "pinned" if i < 10 else "advancing"
        vel = 0.0 if i < 10 else (0.1 * (i - 9))

        f = TemporalFrameResult(
            frame_index=i,
            timestamp_s=i * 0.1,
            accepted=True,
            state=state,
            baseline=((x1, y1), (x2, y2)),
            contacts=((80.0, 100.0), (120.0, 100.0)),
            theta_left_deg=105.0 if i < 10 else (115.0 + i * 0.5),  # downhill
            theta_right_deg=95.0 if i < 10 else (90.0 - i * 0.3),   # uphill
            half_width_mm=1.5,
            contact_velocity_mm_s=vel,
        )
        frames.append(f)

    res = analyze_tilting_plate(frames, surface_tension_mN_m=72.8, droplet_mass_mg=25.0)

    assert res["is_tilting"] is True
    assert res["tilt_range_deg"] == pytest.approx(24.0, abs=0.5)
    assert res["critical_frame_index"] == 10
    assert res["critical_sliding_angle_deg"] == pytest.approx(10.0, abs=0.5)
    assert res["theta_advancing_critical_deg"] > res["theta_receding_critical_deg"]
    assert res["retention_metrics"] is not None
    assert res["retention_metrics"]["total_retention_force_uN"] > 0
