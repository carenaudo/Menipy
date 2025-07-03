import cv2
import numpy as np

from ..models import droplet_volume, contact_angle_from_mask, estimate_surface_tension
from ..models.geometry import horizontal_intersections


def extract_external_contour(img_roi: np.ndarray) -> np.ndarray:
    """Return the largest external contour of ``img_roi``.

    Parameters
    ----------
    img_roi:
        Cropped region containing the droplet.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with contour points in ROI coordinates.

    Raises
    ------
    ValueError
        If no external contour is detected.
    """
    if img_roi.ndim == 3:
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_roi

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No external contour found")

    contour = max(contours, key=cv2.contourArea).squeeze(1).astype(float)
    return contour


def _max_horizontal_diameter(contour: np.ndarray) -> tuple[int, float, int, int]:
    """Return y-position, width and edge x-coordinates of the widest horizontal slice.

    Parameters
    ----------
    contour:
        External contour points ``(x, y)``.

    Returns
    -------
    tuple
        ``(y, width, x_left, x_right)`` values in pixel units.
    """
    x_min = int(np.floor(contour[:, 0].min()))
    y_min = int(np.floor(contour[:, 1].min()))
    x_max = int(np.ceil(contour[:, 0].max()))
    y_max = int(np.ceil(contour[:, 1].max()))

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = np.round(contour - [x_min, y_min]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    left_edges = np.argmax(mask > 0, axis=1)
    right_edges = mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0, axis=1)
    widths = right_edges - left_edges + 1
    widths[mask.sum(axis=1) == 0] = 0
    i = int(widths.argmax())
    y = y_min + i
    width = float(widths[i])
    x_left = x_min + int(left_edges[i])
    x_right = x_min + int(right_edges[i])
    return y, width, x_left, x_right


def compute_drop_metrics(contour: np.ndarray, px_per_mm: float, mode: str) -> dict:
    """Return basic geometric metrics for a droplet contour.

    Parameters
    ----------
    contour:
        External contour points ``(x, y)``.
    px_per_mm:
        Calibration factor in pixels per millimetre.
    mode:
        ``"pendant"`` or ``"contact-angle"`` specifying orientation.

    Returns
    -------
    dict
        Dictionary containing height, diameter, apex, volume, contact angle,
        surface tension and Wo number.

    Raises
    ------
    ValueError
        If ``contour`` shape or ``px_per_mm`` is invalid, or if ``mode`` is unknown.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    if mode not in {"pendant", "contact-angle"}:
        raise ValueError("mode must be 'pendant' or 'contact-angle'")

    x_min = float(contour[:, 0].min())
    x_max = float(contour[:, 0].max())
    y_min = float(contour[:, 1].min())
    y_max = float(contour[:, 1].max())

    y_diam, diam_px, x_left, x_right = _max_horizontal_diameter(contour)

    height_mm = (y_max - y_min) / px_per_mm
    diameter_mm = diam_px / px_per_mm

    apex_idx = int(np.argmax(contour[:, 1])) if mode == "pendant" else int(np.argmin(contour[:, 1]))
    apex = (int(round(contour[apex_idx, 0])), int(round(contour[apex_idx, 1])))

    xi = int(np.floor(x_min))
    yi = int(np.floor(y_min))
    w = int(np.ceil(x_max) - xi + 1)
    h = int(np.ceil(y_max) - yi + 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = np.round(contour - [xi, yi]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    px_to_mm = 1.0 / px_per_mm
    vol_mm3 = droplet_volume(mask, px_to_mm=px_to_mm)
    volume_uL = vol_mm3 / 1000.0 if vol_mm3 is not None else None

    angle = contact_angle_from_mask(mask)
    ift = estimate_surface_tension(mask, 1.0, 1000.0, px_to_mm=px_to_mm)

    y_apex = float(contour[:, 1].max())
    radius_apex_mm = (y_apex - y_diam) / px_per_mm

    xs_contact = horizontal_intersections(contour, y_min)
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    if xs_contact.size >= 2:
        x_c_left = int(round(xs_contact.min()))
        x_c_right = int(round(xs_contact.max()))
        contact_line = ((x_c_left, int(round(y_min))), (x_c_right, int(round(y_min))))

    return {
        "height_mm": float(height_mm),
        "diameter_mm": float(diameter_mm),
        "apex": apex,
        "volume_uL": float(volume_uL) if volume_uL is not None else None,
        "contact_angle_deg": float(angle),
        "ift_mN_m": float(ift) if ift is not None else None,
        "wo": 0.0,
        "diameter_px": float(diam_px),
        "diameter_line": ((x_left, y_diam), (x_right, y_diam)),
        "radius_apex_mm": float(radius_apex_mm),
        "contact_line": contact_line,
    }
