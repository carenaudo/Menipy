import cv2
import numpy as np


def detect_vertical_edges(img_roi: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Detect the main needle axis from an ROI.

    Parameters
    ----------
    img_roi:
        Image region containing the needle.

    Returns
    -------
    tuple
        ``(top_pt, bottom_pt, length_px)`` where points are ``(x, y)`` in
        ROI coordinates.

    Raises
    ------
    ValueError
        If no suitable vertical edges are found.
    """
    if img_roi.ndim == 3:
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_roi

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    lines = cv2.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=max(5, gray.shape[0] // 2),
        maxLineGap=5,
    )

    if lines is None or len(lines) < 2:
        raise ValueError("Needle edges not detected")

    xs: list[int] = []
    y_top = gray.shape[0]
    y_bottom = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) <= 3 and abs(y2 - y1) > 0:
            xs.append(int(round(0.5 * (x1 + x2))))
            y_top = min(y_top, y1, y2)
            y_bottom = max(y_bottom, y1, y2)

    if len(xs) < 2:
        raise ValueError("Needle edges not detected")

    left_x = min(xs)
    right_x = max(xs)
    axis_x = int(round((left_x + right_x) / 2))
    top_pt = (axis_x, int(y_top))
    bottom_pt = (axis_x, int(y_bottom))
    length_px = float(right_x - left_x)
    return top_pt, bottom_pt, length_px
