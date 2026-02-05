"""
Parametric Bezier-like curve generator for testing edge detection integration.
"""
import numpy as np


def bezier_like(img):
    h, w = img.shape[:2]
    t = np.linspace(0, 2 * np.pi, 300)
    r = min(h, w) * (0.2 + 0.05 * np.sin(3 * t))
    return np.column_stack([w / 2 + r * np.cos(t), h / 2 + r * np.sin(t)])


EDGE_DETECTORS = {"bezier": bezier_like}
