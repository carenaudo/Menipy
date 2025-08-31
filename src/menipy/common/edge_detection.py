# src/adsa/common/edge_detection.py
from __future__ import annotations
from typing import Callable
import numpy as np
from .registry import EDGE_DETECTORS
# Plugin loader (optional): discover alternative detectors via entry points.
def _load_plugin(name: str) -> Callable | None:
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="menipy.edge_detection"):
            if ep.name == name:
                return ep.load()
    except Exception:
        pass
    return None

def _fallback_canny(img: np.ndarray) -> np.ndarray:
    """Return Nx2 contour (dummy placeholder)."""
    # TODO: implement with cv2.Canny + cv2.findContours; return the biggest contour
    h, w = img.shape[:2]
    theta = np.linspace(0, 2*np.pi, 200)
    r = min(h, w) * 0.25
    x = w/2 + r*np.cos(theta)
    y = h/2 + r*np.sin(theta)
    return np.column_stack([x, y]).astype(float)

def get_contour_detector(name: str = "canny"):
    return EDGE_DETECTORS.get(name, _fallback_canny)

def run(ctx, method: str = "canny"):
    """
    Run a contour detector on the current frame(s) and update the ContourObj in Context.

    Args:
        ctx: Context object
        method: Name of the contour detector to use (default: "canny")

    Returns:
        ctx with the detected contour stored in ctx.contour.xy
    """
    img = ctx.frames[0] if isinstance(ctx.frames, list) else ctx.frames
    detector = get_contour_detector(method)
    ctx.contour = type("ContourObj", (), {})()
    ctx.contour.xy = detector(img)
    return ctx