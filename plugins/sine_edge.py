import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# Wavy / sine-modulated radius contour generator

def _fallback_sine(h, w, waves=4, amplitude=0.07, points=300):
    base_r = min(h, w) * 0.22
    t = np.linspace(0.0, 2*np.pi, points)
    r = base_r * (1.0 + amplitude * np.sin(waves * t))
    return np.column_stack([w/2 + r*np.cos(t), h/2 + r*np.sin(t)])


def sine_like(img, waves=4, amplitude=0.07, use_canny: bool = False):
    """Return a sine-modulated circular contour.

    If cv2 is available and use_canny=True, attempt to get a contour from an
    edge map. Otherwise return a parametric sine-modulated circle.
    """
    if hasattr(img, 'shape'):
        h, w = img.shape[:2]
    else:
        h, w = img

    if use_canny and cv2 is not None and hasattr(img, 'shape'):
        try:
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                best = max(contours, key=cv2.contourArea)
                pts = best.reshape(-1, 2).astype(float)
                return pts
        except Exception:
            pass

    return _fallback_sine(h, w, waves=waves, amplitude=amplitude)


EDGE_DETECTORS = {"sine": sine_like}

# Auto-register when imported
try:
    from menipy.common.registry import register_edge
    register_edge("sine", sine_like)
except Exception:
    pass
