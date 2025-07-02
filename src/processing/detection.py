from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class Droplet:
    """Result of droplet detection."""

    apex_px: Tuple[int, int]
    contact_px: Tuple[int, int, int, int]
    contour_px: np.ndarray
    projected_area_mm2: float
    r_max_mm: float
    height_mm: float
    mask: np.ndarray


def detect_droplet(frame: np.ndarray, roi: tuple[int, int, int, int], px_to_mm: float) -> Droplet:
    """Detect droplet silhouette inside ``roi`` and return geometric metrics.

    Parameters
    ----------
    frame:
        Original image array.
    roi:
        (x0, y0, w, h) bounding box in image coordinates.
    px_to_mm:
        Pixel-to-millimetre calibration factor.

    Returns
    -------
    Droplet
        Dataclass with contour, apex, contact point and metrics.

    Raises
    ------
    ValueError
        If no droplet contour is found inside the ROI.
    """
    x0, y0, w, h = roi
    crop = frame[y0 : y0 + h, x0 : x0 + w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv = cv2.bitwise_not(thresh)
    contours_a, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_b, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_a = max((cv2.contourArea(c) for c in contours_a), default=0)
    area_b = max((cv2.contourArea(c) for c in contours_b), default=0)
    if max(area_a, area_b) >= 0.95 * (w * h) and min(area_a, area_b) == 0:
        raise ValueError("No droplet found in ROI")
    if area_a == 0:
        mask = inv
    elif area_b == 0:
        mask = thresh
    else:
        mask = thresh if area_a < area_b else inv

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    ys, xs = np.nonzero(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No droplet found in ROI")

    areas = [cv2.contourArea(c) for c in contours]
    cnt = contours[int(np.argmax(areas))].astype(np.float32).reshape(-1, 2)
    cnt[:, 0] += x0
    cnt[:, 1] += y0

    top_points = cnt[np.isclose(cnt[:, 1], y0)]
    bottom_points = cnt[np.isclose(cnt[:, 1], y0 + h - 1)]
    if top_points.size:
        orientation = "pendant"
        contact_line_pts = top_points
        apex_band = cnt[np.isclose(cnt[:, 1], cnt[:, 1].max())]
        apex_x = 0.5 * (apex_band[:, 0].min() + apex_band[:, 0].max())
        apex_y = cnt[:, 1].max()
    elif bottom_points.size:
        orientation = "sessile"
        contact_line_pts = bottom_points
        apex_band = cnt[np.isclose(cnt[:, 1], cnt[:, 1].min())]
        apex_x = 0.5 * (apex_band[:, 0].min() + apex_band[:, 0].max())
        apex_y = cnt[:, 1].min()
    else:
        raise ValueError("No droplet found in ROI")

    left = int(round(contact_line_pts[:, 0].min()))
    right = int(round(contact_line_pts[:, 0].max()))
    y_contact = int(round(contact_line_pts[0, 1]))
    contact_px = (left, y_contact, right, y_contact)
    apex_px = (int(round(apex_x)), int(round(apex_y)))

    height_mm = abs(contact_px[1] - apex_px[1]) * px_to_mm
    width_px = xs.max() - xs.min() + 1
    r_max_mm = 0.5 * width_px * px_to_mm
    projected_area_mm2 = float(np.count_nonzero(mask) * (px_to_mm ** 2))

    return Droplet(
        apex_px=apex_px,
        contact_px=contact_px,
        contour_px=cnt,
        projected_area_mm2=projected_area_mm2,
        r_max_mm=r_max_mm,
        height_mm=height_mm,
        mask=mask,
    )
