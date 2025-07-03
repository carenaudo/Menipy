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


@dataclass
class SessileDroplet:
    """Output of the robust sessile drop detector."""

    contour_px: np.ndarray
    substrate_px: Tuple[int, int, int, int]
    contact_px: Tuple[int, int, int, int]
    apex_px: Tuple[int, int]
    height_mm: float
    r_max_mm: float
    projected_area_mm2: float
    mask: np.ndarray


@dataclass
class PendantDroplet:
    """Output of the robust pendant drop detector."""

    contour_px: np.ndarray
    needle_px: Tuple[int, int, int, int]
    contact_px: Tuple[int, int, int, int]
    apex_px: Tuple[int, int]
    height_mm: float
    r_max_mm: float
    projected_area_mm2: float
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


def _fit_line_ransac(points: np.ndarray, iterations: int = 50) -> tuple[float, float, float]:
    """Fit a 2D line ``ax + by + c = 0`` to ``points`` using a simple RANSAC."""

    if points.shape[0] < 2:
        raise ValueError("not enough points for line fit")

    rng = np.random.default_rng()
    best_inliers: np.ndarray | None = None
    best_err = np.inf
    for _ in range(iterations):
        idx = rng.choice(points.shape[0], 2, replace=False)
        (x1, y1), (x2, y2) = points[idx]
        a = y2 - y1
        b = x1 - x2
        c = -(a * x1 + b * y1)
        norm = np.hypot(a, b)
        if norm == 0:
            continue
        d = np.abs(a * points[:, 0] + b * points[:, 1] + c) / norm
        inliers = d < 1.0
        err = d[inliers].sum()
        if err < best_err and inliers.sum() >= 2:
            best_err = float(err)
            best_inliers = inliers

    if best_inliers is None:
        raise ValueError("Surface line not detected")

    inlier_pts = points[best_inliers]
    A = np.c_[inlier_pts[:, 0], np.ones_like(inlier_pts[:, 0])]
    bvec = inlier_pts[:, 1]
    coeffs, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
    m, k = coeffs
    return m, -1.0, k


def detect_sessile_droplet(
    frame: np.ndarray, roi: tuple[int, int, int, int], px_to_mm: float
) -> SessileDroplet:
    """Detect a sessile droplet on a possibly tilted surface.

    Parameters
    ----------
    frame:
        Original image array.
    roi:
        (x0, y0, w, h) region of interest in image coordinates.
    px_to_mm:
        Pixel-to-millimetre calibration factor.

    Returns
    -------
    SessileDroplet
        Detected geometric information.

    Raises
    ------
    ValueError
        If the surface line cannot be detected or no contour found.
    """

    x0, y0, w, h = roi
    crop = frame[y0 : y0 + h, x0 : x0 + w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    edges = cv2.Canny(mask, 50, 150)
    min_len = int(0.6 * w)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_len, maxLineGap=5)
    if lines is None or len(lines) < 2:
        raise ValueError("Surface line not detected")

    best = None
    best_len = -1.0
    for x1, y1, x2, y2 in lines[:, 0]:
        mean_y = (y1 + y2) / 2
        length = float(np.hypot(x2 - x1, y2 - y1))
        if mean_y >= 0.5 * h and length > best_len:
            best_len = length
            best = (x1 + x0, y1 + y0, x2 + x0, y2 + y0)

    if best is None:
        raise ValueError("Surface line not detected")

    # gather edge points around the chosen segment for RANSAC
    ys, xs = np.nonzero(edges)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    pts = pts[pts[:, 1] >= y0 + 0.5 * h]

    a, b, c = _fit_line_ransac(pts)

    def substrate_y_at(x: float) -> float:
        return -(a * x + c) / b

    x1 = x0
    y1 = substrate_y_at(x1)
    x2 = x0 + w - 1
    y2 = substrate_y_at(x2)
    substrate_px = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No droplet found in ROI")

    cnt = max(contours, key=cv2.contourArea).astype(np.float32)
    cnt[:, 0, 0] += x0
    cnt[:, 0, 1] += y0
    cnt = cnt.reshape(-1, 2)

    apex_idx = int(np.argmin(cnt[:, 1]))
    apex = tuple(int(round(v)) for v in cnt[apex_idx])

    d = np.abs(a * cnt[:, 0] + b * cnt[:, 1] + c) / np.hypot(a, b)
    idx = np.where(d < 1.0)[0]
    if idx.size == 0:
        raise ValueError("Surface line not detected")

    idx2 = np.sort(np.concatenate([idx, idx + len(cnt)]))
    splits = np.where(np.diff(idx2) > 1)[0] + 1
    segments = [s for s in np.split(idx2, splits) if s.size > 0]
    seg = max(segments, key=len)
    seg = seg % len(cnt)
    contact_pts = cnt[seg]
    contact_left = tuple(int(round(v)) for v in contact_pts[0])
    contact_right = tuple(int(round(v)) for v in contact_pts[-1])
    contact_px = (*contact_left, *contact_right)

    height_mm = (substrate_y_at(apex[0]) - apex[1]) * px_to_mm
    width_px = contact_right[0] - contact_left[0]
    r_max_mm = 0.5 * width_px * px_to_mm
    projected_area_mm2 = float(np.count_nonzero(mask) * (px_to_mm ** 2))

    return SessileDroplet(
        contour_px=cnt,
        substrate_px=substrate_px,
        contact_px=contact_px,
        apex_px=apex,
        height_mm=height_mm,
        r_max_mm=r_max_mm,
        projected_area_mm2=projected_area_mm2,
        mask=mask,
    )


def detect_pendant_droplet(
    frame: np.ndarray, roi: tuple[int, int, int, int], px_to_mm: float
) -> PendantDroplet:
    """Detect a pendant droplet hanging from a needle line.

    Parameters
    ----------
    frame:
        Original image array.
    roi:
        (x0, y0, w, h) region of interest in image coordinates.
    px_to_mm:
        Pixel-to-millimetre calibration factor.

    Returns
    -------
    PendantDroplet
        Detected geometric information.

    Raises
    ------
    ValueError
        If the needle line cannot be detected or no contour found.
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

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    edges = cv2.Canny(mask, 50, 150)
    ys, xs = np.nonzero(edges)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    pts = pts[pts[:, 1] <= y0 + 0.5 * h]
    if pts.shape[0] < 2:
        raise ValueError("Needle line not detected")

    a, b, c = _fit_line_ransac(pts)

    def needle_y_at(x: float) -> float:
        return -(a * x + c) / b

    x1 = x0
    y1 = needle_y_at(x1)
    x2 = x0 + w - 1
    y2 = needle_y_at(x2)
    needle_px = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No droplet found in ROI")

    cnt = max(contours, key=cv2.contourArea).astype(np.float32)
    cnt[:, 0, 0] += x0
    cnt[:, 0, 1] += y0
    cnt = cnt.reshape(-1, 2)

    apex_idx = int(np.argmax(cnt[:, 1]))
    apex = tuple(int(round(v)) for v in cnt[apex_idx])

    d = np.abs(a * cnt[:, 0] + b * cnt[:, 1] + c) / np.hypot(a, b)
    idx = np.where(d < 2.0)[0]
    if idx.size == 0:
        raise ValueError("Needle line not detected")

    idx2 = np.sort(np.concatenate([idx, idx + len(cnt)]))
    splits = np.where(np.diff(idx2) > 1)[0] + 1
    segments = [s for s in np.split(idx2, splits) if s.size > 0]
    seg = max(segments, key=len)
    seg = seg % len(cnt)
    contact_pts = cnt[seg]
    contact_left = tuple(int(round(v)) for v in contact_pts[0])
    contact_right = tuple(int(round(v)) for v in contact_pts[-1])
    contact_px = (*contact_left, *contact_right)

    height_mm = (apex[1] - needle_y_at(apex[0])) * px_to_mm
    width_px = contact_right[0] - contact_left[0]
    r_max_mm = 0.5 * width_px * px_to_mm
    projected_area_mm2 = float(np.count_nonzero(mask) * (px_to_mm ** 2))

    return PendantDroplet(
        contour_px=cnt,
        needle_px=needle_px,
        contact_px=contact_px,
        apex_px=apex,
        height_mm=height_mm,
        r_max_mm=r_max_mm,
        projected_area_mm2=projected_area_mm2,
        mask=mask,
    )
