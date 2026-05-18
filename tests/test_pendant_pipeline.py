"""Regression tests for pendant contour and calibrated diameter behavior."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.models.context import Context
from menipy.models.geometry import Contour, Geometry
from menipy.pipelines.pendant.stages import PendantPipeline


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
    ctx = Context(
        contour=Contour(
            xy=np.array(
                [
                    [50, 100],
                    [150, 100],
                    [40, 130],
                    [160, 130],
                    [55, 160],
                    [145, 160],
                ],
                dtype=float,
            )
        ),
        contact_points=((50, 100), (150, 100)),
        scale={"px_per_mm": 20.0},
        fit={
            "param_names": ["r0_mm", "beta"],
            "params": [99.0, 0.4],
            "residuals": {},
        },
    )
    ctx.geometry = Geometry(axis_x=100.0, apex_xy=(100.0, 180.0))

    PendantPipeline().do_compute_metrics(ctx)

    assert ctx.results["r0_mm"] == 99.0
    assert ctx.results["diameter_px"] == 120.0
    assert ctx.results["diameter_mm"] == pytest.approx(6.0)
    assert ctx.results["diameter_mm"] != pytest.approx(198.0)


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
