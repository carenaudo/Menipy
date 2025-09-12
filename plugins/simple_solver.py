import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# Minimal solver plugin: generates a parabola profile as an example solver output

def _estimate_scale_from_image(img):
    """If cv2 is available, compute a rough scale factor based on contours.

    Returns a positive float scale (defaults to 1.0 on failure).
    """
    if cv2 is None or not hasattr(img, 'shape'):
        return 1.0
    try:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1.0
        # use median area to infer scale
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1]
        if not areas:
            return 1.0
        median_area = float(np.median(np.array(areas)))
        # heuristic: larger area -> larger scale; normalize
        return max(0.1, min(10.0, np.sqrt(median_area) / 50.0))
    except Exception:
        return 1.0


def parabola_solver(params, physics=None, geometry=None, img=None):
    """Return a simple parabola profile using a size parameter.

    If an image is provided and cv2 is available, use a rough estimated scale
    from the image to modulate the solver output.
    """
    a = float(params[0]) if len(params) > 0 else 1.0
    if img is not None:
        scale = _estimate_scale_from_image(img)
    else:
        scale = 1.0
    x = np.linspace(-1.0 * scale, 1.0 * scale, 200)
    y = a * (x**2)
    return np.column_stack([x, y])


SOLVERS = {"parabola": parabola_solver}

# Auto-register when imported
try:
    from menipy.common.registry import register_solver
    register_solver("parabola", parabola_solver)
except Exception:
    pass
