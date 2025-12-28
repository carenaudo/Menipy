import numpy as np

# Try to import OpenCV; it's optional for environments without cv2
try:
    import cv2
except Exception:
    cv2 = None


# Simple circular contour generator plugin
# API: define a function that accepts an image (or shape) and returns Nx2 contour points


def _fallback_circle(h, w, points=200):
    r = min(h, w) * 0.25
    t = np.linspace(0.0, 2 * np.pi, points)
    return np.column_stack([w / 2 + r * np.cos(t), h / 2 + r * np.sin(t)])


def circle_like(img, use_canny: bool = False):
    """Return a centered circle contour sized relative to the image.

    If cv2 is available and use_canny=True, perform a simple Canny edge
    extraction and attempt to extract the largest contour. Otherwise return a
    parametric circle as fallback.
    """
    if hasattr(img, "shape"):
        h, w = img.shape[:2]
    else:
        h, w = img

    if use_canny and cv2 is not None and hasattr(img, "shape"):
        try:
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if contours:
                # pick the largest contour by length/area
                best = max(contours, key=cv2.contourArea)
                pts = best.reshape(-1, 2).astype(float)
                return pts
        except Exception:
            # fall through to parametric fallback
            pass

    return _fallback_circle(h, w)


EDGE_DETECTORS = {"circle": circle_like}

# Auto-register when imported so discovery/registry can pick this up
try:
    from menipy.common.registry import register_edge

    register_edge("circle", circle_like)
except Exception:
    # registry may not be available in some contexts; fail silently
    pass
