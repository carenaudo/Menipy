import pytest
import numpy as np

from menipy.models.context import Context
from menipy.pipelines.captive_bubble.stages import CaptiveBubblePipeline
from menipy.pipelines.captive_bubble.physics import compute_physics

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
    from menipy.models.geometry import CaptiveBubbleGeometry
    from menipy.models.context import Contour
    ctx.contour = Contour(xy=np.array([[0, 0], [10, 0], [5, 10]], dtype=float))
    
    pipeline = CaptiveBubblePipeline()
    ctx = pipeline.do_geometric_features(ctx)
    
    assert ctx.geometry is not None
    assert isinstance(ctx.geometry, CaptiveBubbleGeometry)
    assert ctx.geometry.cap_depth_px == 10.0
