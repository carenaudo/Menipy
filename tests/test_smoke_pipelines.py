# tests/test_smoke_pipelines.py
from __future__ import annotations

import math
import numpy as np
import pytest

# import the pipelines (plural layout)
from src.menipy.pipelines.sessile import SessilePipeline
from src.menipy.pipelines.pendant import PendantPipeline
from src.menipy.pipelines.oscillating import OscillatingPipeline
from src.menipy.pipelines.capillary_rise import CapillaryRisePipeline
from src.menipy.pipelines.captive_bubble import CaptiveBubblePipeline

# Optional: overlay uses OpenCV in your codepath; preview checks are skipped if cv2 missing
try:
    import cv2  # type: ignore
    _CV_OK = True
except Exception:
    _CV_OK = False


def _circle_xy(cx: float, cy: float, r: float, n: int = 256) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]).astype(float)


def _mk_frame(h: int = 512, w: int = 512) -> np.ndarray:
    """Make a plain BGR frame as the drawing surface for overlay."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "PipelineClass, edge_setup",
    [
        # sessile: one contour
        (SessilePipeline, lambda ctx: _set_single_contour(ctx, _circle_xy(256, 256, 140))),
        # pendant: one contour
        (PendantPipeline, lambda ctx: _set_single_contour(ctx, _circle_xy(256, 256, 150))),
        # oscillating: series of contours with slightly varying radius
        (OscillatingPipeline, lambda ctx: _set_series_contours(ctx, base_r=150, amp=8, n=24)),
        # capillary rise: one contour
        (CapillaryRisePipeline, lambda ctx: _set_single_contour(ctx, _circle_xy(256, 256, 135))),
        # captive bubble: one contour
        (CaptiveBubblePipeline, lambda ctx: _set_single_contour(ctx, _circle_xy(256, 256, 120))),
    ],
)
def test_pipeline_smoke(PipelineClass, edge_setup):
    # construct
    p = PipelineClass()

    # --- patch acquisition to inject a synthetic frame ---
    def do_acq(ctx):
        ctx.frames = [_mk_frame()]
        return ctx

    p.do_acquisition = do_acq  # type: ignore[attr-defined]

    # --- patch edge detection to avoid cv2 dependency ---
    if edge_setup is not None:
        def do_edge(ctx):
            return edge_setup(ctx)
        p.do_edge_detection = do_edge  # type: ignore[attr-defined]
    else:
        # fallback: single contour
        def do_edge(ctx):
            return _set_single_contour(ctx, _circle_xy(256, 256, 140))
        p.do_edge_detection = do_edge  # type: ignore[attr-defined]

    # --- patch solver to avoid SciPy and keep test deterministic ---
    def do_solver(ctx):
        # Minimal "fit" that downstream stages expect
        ctx.fit = {
            "param_names": ["R0_mm"],
            "params": [42.0],
            "residuals": {"rss": 0.0},
            "solver": {"success": True},
        }
        return ctx

    p.do_solver = do_solver  # type: ignore[attr-defined]

    # run
    ctx = p.run()

    # results must exist
    assert isinstance(ctx.results, dict) and len(ctx.results) > 0
    assert "R0_mm" in ctx.results  # injected by our patched solver

    # overlay preview exists only if cv2 is available (your overlay uses cv2.addWeighted)
    if _CV_OK:
        assert getattr(ctx, "preview", None) is not None
        assert isinstance(ctx.preview, np.ndarray)
        assert ctx.preview.ndim == 3 and ctx.preview.shape[2] == 3
    else:
        pytest.skip("OpenCV not available; preview composition is cv2-dependent.")


# ------------------------ helpers for edge stage -----------------------------

def _set_single_contour(ctx, xy: np.ndarray):
    C = type("Contour", (), {})()
    C.xy = xy
    ctx.contour = C
    return ctx

def _set_series_contours(ctx, base_r: float, amp: float, n: int = 16):
    contours = []
    for k in range(n):
        r = base_r + amp * math.sin(2.0 * math.pi * k / n)
        xy = _circle_xy(256, 256, r)
        C = type("Contour", (), {})()
        C.xy = xy
        contours.append(C)
    ctx.contours_by_frame = contours
    ctx.contour = contours[0]
    return ctx
