import numpy as np

from menipy.models.context import Context
from menipy.pipelines.captive_bubble.physics import compute_physics
from menipy.pipelines.captive_bubble.stages import CaptiveBubblePipeline


def test_captive_bubble_physics():
    # Test gamma and capillary length calculations
    config = {"rho1": 1000.0, "rho2": 1.2, "g": 9.8}
    r0_mm = 2.0
    beta = 0.5

    gamma, cl_mm = compute_physics(config, r0_mm, beta)

    assert gamma is not None
    assert cl_mm is not None
    assert gamma > 0
    assert cl_mm > 0

    # beta = (d_rho * g * r0_m^2) / gamma
    # gamma = (1000 - 1.2) * 9.8 * 0.002^2 / 0.5 = 998.8 * 9.8 * 0.000004 / 0.5 = 0.07830592 N/m = 78.3 mN/m
    assert abs(gamma - 78.30592) < 0.001


def test_captive_bubble_pipeline_integration(monkeypatch):
    # Mock edged run
    import menipy.common.edge_detection as edged

    monkeypatch.setattr(edged, "run", lambda *args, **kwargs: None)

    # Give a dummy contour
    ctx = Context()
    from menipy.models.context import Contour
    from menipy.models.geometry import CaptiveBubbleGeometry

    ctx.contour = Contour(xy=np.array([[0, 0], [10, 0], [5, 10]], dtype=float))

    pipeline = CaptiveBubblePipeline()
    ctx = pipeline.do_geometric_features(ctx)

    assert ctx.geometry is not None
    assert isinstance(ctx.geometry, CaptiveBubbleGeometry)
    assert ctx.geometry.cap_depth_px == 10.0


def test_captive_bubble_metrics_and_contract():
    # Synthetic parabolic bubble: ceiling at y=10, apex at (100, 110), depth = 100px, diameter = 200px
    xs = np.linspace(0, 200, 60)
    ys = 110.0 - 100.0 * (1.0 - ((xs - 100.0) / 100.0) ** 2)
    ceiling_pts = np.array([[200, 10], [0, 10]])
    contour_pts = np.vstack([np.column_stack([xs, ys]), ceiling_pts])

    ctx = Context()
    from menipy.models.geometry import Contour
    ctx.contour = Contour(xy=contour_pts)
    ctx.scale = {"px_per_mm": 50.0}  # 50 px/mm -> depth_mm = 2.0 mm, diameter_mm = 4.0 mm
    ctx.physics = {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}

    pipeline = CaptiveBubblePipeline()
    ctx = pipeline.do_geometric_features(ctx)
    ctx = pipeline.do_calibration(ctx)
    ctx = pipeline.do_physics(ctx)
    ctx = pipeline.do_profile_fitting(ctx)
    ctx = pipeline.do_compute_metrics(ctx)

    assert "depth_mm" in ctx.results
    assert abs(ctx.results["depth_mm"] - 2.0) < 0.1
    assert "diameter_mm" in ctx.results
    assert abs(ctx.results["diameter_mm"] - 4.0) < 0.1
    assert "r0_mm" in ctx.results
    assert ctx.results["r0_mm"] > 0
    assert "surface_tension_mN_m" in ctx.results
    assert ctx.results["surface_tension_mN_m"] > 0
    assert "volume_uL" in ctx.results
    assert ctx.results["volume_uL"] > 0
    assert "theta_bubble_deg" in ctx.results
    assert "theta_liquid_deg" in ctx.results
    assert abs(ctx.results["theta_bubble_deg"] + ctx.results["theta_liquid_deg"] - 180.0) < 1e-4

