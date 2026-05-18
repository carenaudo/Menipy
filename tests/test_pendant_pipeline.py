"""Regression tests for pendant contour and calibrated diameter behavior."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.models.context import Context
from menipy.models.geometry import Contour, Geometry
from menipy.models.surface_tension import jennings_pallas_beta, surface_tension
from menipy.pipelines.pendant.stages import PendantPipeline
from menipy.pipelines.pendant.strict_young_laplace import (
    integrate_young_laplace_profile_mm,
    model_mm_to_pendant_px,
)


@pytest.mark.parametrize("opencv_shape", [False, True])
def test_pendant_contour_extraction_clips_needle_shaft(opencv_shape):
    contour = np.array(
        [
            [90, 0],
            [110, 0],
            [110, 100],
            [150, 130],
            [170, 180],
            [100, 240],
            [30, 180],
            [50, 130],
            [90, 100],
        ],
        dtype=float,
    )
    if opencv_shape:
        contour = contour.reshape(-1, 1, 2)

    contact_points = ((90, 100), (110, 100))
    ctx = Context(drop_contour=contour, contact_points=contact_points)

    PendantPipeline().do_contour_extraction(ctx)

    xy = np.asarray(ctx.contour.xy, dtype=float)
    assert xy.ndim == 2
    assert xy.shape[1] == 2
    assert np.min(xy[:, 1]) >= 100
    assert any(np.allclose(point, [90, 100]) for point in xy)
    assert any(np.allclose(point, [110, 100]) for point in xy)


def test_pendant_diameter_uses_calibrated_contour_width_not_fit_radius():
    theta = np.linspace(0.0, np.pi, 81)
    contour = np.column_stack(
        [
            100.0 + 20.0 * np.cos(theta),
            160.0 + 20.0 * np.sin(theta),
        ]
    )
    ctx = Context(
        contour=Contour(xy=contour),
        contact_points=((80, 100), (120, 100)),
        scale={"px_per_mm": 20.0},
        fit={
            "param_names": ["r0_mm", "beta"],
            "params": [99.0, 0.4],
            "residuals": {},
        },
    )
    ctx.geometry = Geometry(axis_x=100.0, apex_xy=(100.0, 180.0))

    PendantPipeline().do_compute_metrics(ctx)

    assert ctx.results["fit_r0_mm"] == 99.0
    assert ctx.results["fit_beta"] == 0.4
    assert ctx.results["r0_mm"] == pytest.approx(1.0)
    assert ctx.results["diameter_px"] == pytest.approx(40.0)
    assert ctx.results["diameter_mm"] == pytest.approx(2.0)
    assert ctx.results["diameter_mm"] != pytest.approx(198.0)
    assert ctx.results["surface_tension_method"] == "jennings_pallas_geometric"


def test_pendant_surface_tension_uses_geometric_mn_per_m_units():
    theta = np.linspace(0.0, np.pi, 81)
    contour = np.column_stack(
        [
            100.0 + 20.0 * np.cos(theta),
            160.0 + 20.0 * np.sin(theta),
        ]
    )
    ctx = Context(
        contour=Contour(xy=contour),
        scale={"px_per_mm": 20.0},
        physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
        fit={
            "param_names": ["r0_mm", "beta"],
            "params": [99.0, 0.2],
            "residuals": {"rmse": 500.0},
        },
    )
    ctx.geometry = Geometry(axis_x=100.0, apex_xy=(100.0, 180.0))

    PendantPipeline().do_compute_metrics(ctx)

    expected_beta = jennings_pallas_beta(1.0)
    expected_n_m = surface_tension(998.8, 9.80665, 1.0, expected_beta)
    assert ctx.results["s1"] == pytest.approx(1.0)
    assert ctx.results["beta"] == pytest.approx(expected_beta)
    assert ctx.results["surface_tension_mN_m"] == pytest.approx(expected_n_m * 1000.0)
    assert ctx.results["surface_tension_mN_m"] != pytest.approx(expected_n_m)
    assert ctx.results["fit_warning"] == "young_laplace_fit_unreliable"


def test_strict_young_laplace_recovers_synthetic_calibrated_contour():
    r0_mm = 1.2
    beta = 0.6
    px_per_mm = 100.0
    axis_x = 250.0
    apex_y = 300.0
    model_mm = integrate_young_laplace_profile_mm(
        r0_mm, beta, target_height_mm=2.0
    )
    model_mm = model_mm[model_mm[:, 1] <= 2.0]
    contour_px = model_mm_to_pendant_px(
        model_mm,
        axis_x_px=axis_x,
        apex_y_px=apex_y,
        px_per_mm=px_per_mm,
    )
    ctx = Context(
        contour=Contour(xy=contour_px),
        scale={"px_per_mm": px_per_mm},
        physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
    )
    ctx.geometry = Geometry(axis_x=axis_x, apex_xy=(axis_x, apex_y))

    pipeline = PendantPipeline()
    pipeline.do_profile_fitting(ctx)
    pipeline.do_compute_metrics(ctx)

    expected_gamma = surface_tension(998.8, 9.80665, r0_mm, beta) * 1000.0
    assert ctx.results["surface_tension_method"] == "young_laplace_strict"
    assert ctx.results["strict_fit_success"] is True
    assert ctx.results["r0_mm"] == pytest.approx(r0_mm, rel=1e-6)
    assert ctx.results["beta"] == pytest.approx(beta, rel=1e-6)
    assert ctx.results["surface_tension_mN_m"] == pytest.approx(expected_gamma, rel=1e-6)
    assert ctx.fit["residuals"]["units"] == "mm"


def test_strict_young_laplace_recovers_shifted_noisy_contour_offsets():
    rng = np.random.default_rng(12)
    r0_mm = 1.2
    beta = 0.6
    px_per_mm = 100.0
    axis_x = 250.0
    apex_y = 300.0
    shift_mm = np.array([0.04, -0.03])
    model_mm = integrate_young_laplace_profile_mm(
        r0_mm, beta, target_height_mm=2.0
    )
    model_mm = model_mm[model_mm[:, 1] <= 2.0] + shift_mm
    contour_px = model_mm_to_pendant_px(
        model_mm,
        axis_x_px=axis_x,
        apex_y_px=apex_y,
        px_per_mm=px_per_mm,
    )
    contour_px += rng.normal(0.0, 0.4, contour_px.shape)
    ctx = Context(
        contour=Contour(xy=contour_px),
        scale={"px_per_mm": px_per_mm},
        physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
    )
    ctx.geometry = Geometry(axis_x=axis_x, apex_xy=(axis_x, apex_y))

    pipeline = PendantPipeline()
    pipeline.do_profile_fitting(ctx)
    pipeline.do_compute_metrics(ctx)

    assert ctx.results["surface_tension_method"] == "young_laplace_strict"
    assert ctx.results["strict_fit_success"] is True
    assert ctx.results["r0_mm"] == pytest.approx(r0_mm, rel=0.01)
    assert ctx.results["beta"] == pytest.approx(beta, rel=0.03)
    assert ctx.results["strict_x_offset_mm"] == pytest.approx(shift_mm[0], abs=0.02)
    assert ctx.results["strict_z_offset_mm"] == pytest.approx(shift_mm[1], abs=0.02)


def test_strict_young_laplace_failure_falls_back_to_geometric_metrics():
    theta = np.linspace(0.0, np.pi, 81)
    contour = np.column_stack(
        [
            100.0 + 20.0 * np.cos(theta),
            160.0 + 20.0 * np.sin(theta),
        ]
    )
    ctx = Context(
        contour=Contour(xy=contour),
        scale={"px_per_mm": 20.0},
        physics={"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
        fit={
            "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
            "params": [5.0, 0.5, 0.0, 0.0],
            "residuals": {"rmse": 10.0, "units": "mm"},
            "strict_fit_success": False,
            "strict_fit_warning": "residual_gate_failed",
            "strict_surface_tension_mN_m": 999.0,
            "strict_r0_mm": 5.0,
            "strict_beta": 0.5,
            "strict_rmse_mm": 10.0,
        },
    )
    ctx.geometry = Geometry(axis_x=100.0, apex_xy=(100.0, 180.0))

    PendantPipeline().do_compute_metrics(ctx)

    expected_beta = jennings_pallas_beta(1.0)
    expected_gamma = surface_tension(998.8, 9.80665, 1.0, expected_beta) * 1000.0
    assert ctx.results["surface_tension_method"] == "jennings_pallas_geometric"
    assert ctx.results["surface_tension_mN_m"] == pytest.approx(expected_gamma)
    assert ctx.results["surface_tension_mN_m"] != 999.0
    assert ctx.results["fit_warning"] == "young_laplace_fit_unreliable"
    assert ctx.results["strict_fit_success"] is False


def test_sample_pendant_uses_detected_drop_not_roi_border_and_scales_diameter():
    sample = Path("data/samples/gota pendiente 1.png")
    image = cv2.imread(str(sample))
    assert image is not None

    calibration = AutoCalibrator(image, "pendant").detect_all()
    assert calibration.needle_rect is not None
    assert calibration.drop_contour is not None
    assert calibration.contact_points is not None
    assert calibration.roi_rect is not None

    pipeline = PendantPipeline()
    ctx = pipeline.run_with_plan(
        only=["contour_extraction"],
        image=str(sample),
        drop_contour=calibration.drop_contour,
        contact_points=calibration.contact_points,
        apex_point=calibration.apex_point,
        needle_rect=calibration.needle_rect,
        roi_rect=calibration.roi_rect,
        scale={"px_per_mm": calibration.needle_rect[2] / 1.83},
        calibration_params={
            "needle_diameter_mm": 1.83,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2,
        },
    )

    xy = np.asarray(ctx.contour.xy, dtype=float)
    contact_y = min(p[1] for p in calibration.contact_points)
    roi_x, roi_y, roi_w, roi_h = calibration.roi_rect

    assert np.min(xy[:, 1]) >= contact_y
    assert not (
        int(np.min(xy[:, 0])) == roi_x
        and int(np.min(xy[:, 1])) == roi_y
        and int(np.max(xy[:, 0])) == roi_x + roi_w
        and int(np.max(xy[:, 1])) == roi_y + roi_h
    )

    diameters = []
    for needle_diameter_mm in (1.0, 2.0):
        metrics_ctx = Context(
            contour=ctx.contour,
            contact_points=calibration.contact_points,
            scale={
                "px_per_mm": calibration.needle_rect[2] / needle_diameter_mm,
            },
            fit={
                "param_names": ["r0_mm", "beta"],
                "params": [20.0, 0.3],
                "residuals": {},
            },
        )
        metrics_ctx.geometry = Geometry(axis_x=float(np.median(xy[:, 0])))
        pipeline.do_compute_metrics(metrics_ctx)
        diameters.append(metrics_ctx.results["diameter_mm"])

    assert diameters[1] > diameters[0]


def test_sample_pendant_surface_tension_reports_geometric_result_not_bad_fit():
    sample = Path("data/samples/gota pendiente 1.png")
    image = cv2.imread(str(sample))
    assert image is not None

    calibration = AutoCalibrator(image, "pendant").detect_all()
    assert calibration.needle_rect is not None
    assert calibration.drop_contour is not None

    ctx = PendantPipeline().run_with_plan(
        only=["compute_metrics"],
        image=str(sample),
        drop_contour=calibration.drop_contour,
        contact_points=calibration.contact_points,
        apex_point=calibration.apex_point,
        needle_rect=calibration.needle_rect,
        roi_rect=calibration.roi_rect,
        calibration_params={
            "needle_diameter_mm": 1.83,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2,
        },
    )

    assert ctx.results["surface_tension_method"] in {
        "young_laplace_strict",
        "jennings_pallas_geometric",
    }
    assert ctx.results["surface_tension_mN_m"] < 100.0
    assert ctx.results["surface_tension_mN_m"] != pytest.approx(17307.2, rel=1e-2)
    if ctx.results["surface_tension_method"] == "young_laplace_strict":
        assert ctx.results["strict_fit_success"] is True
        assert "fit_warning" not in ctx.results
    else:
        assert ctx.results["fit_warning"] == "young_laplace_fit_unreliable"
    assert "strict_r0_mm" in ctx.results
    assert "geometric_surface_tension_mN_m" in ctx.results
