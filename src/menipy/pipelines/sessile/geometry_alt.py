from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import cv2

from ...analysis import compute_sessile_metrics_alt
from ...physics.contact_geom import line_params
import math


def filter_contours_by_size(
    contours: list[np.ndarray],
    min_area: float,
    max_area: float,
) -> list[np.ndarray]:
    """Return contours with area within ``min_area`` and ``max_area``."""
    result: list[np.ndarray] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if min_area <= area <= max_area:
            result.append(c)
    return result


def exclude_near_line(
    contours: list[np.ndarray],
    line: tuple[tuple[int, int], tuple[int, int]],
    p1: tuple[int, int],
    p2: tuple[int, int],
    radius: float = 5.0,
) -> list[np.ndarray]:
    """Return contours not touching ``line`` outside a tolerance around P1/P2."""
    a, b, c = line_params(line[0], line[1])
    allowed = [np.array(p1, float), np.array(p2, float)]
    res: list[np.ndarray] = []
    for cts in contours:
        d = np.abs(a * cts[:, 0] + b * cts[:, 1] + c) / (a * a + b * b) ** 0.5
        if d.min() >= radius:
            res.append(cts)
            continue
        mask = d < radius
        pts = cts[mask]
        if len(pts) == 0:
            res.append(cts)
            continue
        keep = True
        for p in pts:
            if all(np.linalg.norm(p - q) > radius for q in allowed):
                keep = False
                break
        if keep:
            res.append(cts)
    return res


def extract_outer_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external contour in ``mask``."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")
    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze(1).astype(float)


def project_onto_line(
    pts: np.ndarray, line: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and foot of each point projected onto ``line``."""
    p1 = np.asarray(line[0], float)
    p2 = np.asarray(line[1], float)
    d = p2 - p1
    denom = float(d.dot(d))
    if denom == 0:
        raise ValueError("invalid line")
    t = np.clip(((pts - p1) @ d) / denom, 0.0, 1.0)
    foot = p1 + t[:, None] * d
    dist = np.linalg.norm(pts - foot, axis=1)
    return dist, foot


def find_contact_points(
    contour: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    guess1: tuple[int, int],
    guess2: tuple[int, int],
    tol: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered contact points nearest to ``guess1`` and ``guess2``."""
    from ...detectors.geometry_alt import polyline_contour_intersections

    poly = np.array([line[0], line[1]], float)
    pts = polyline_contour_intersections(poly, contour)
    if len(pts) < 2:
        raise ValueError("line does not intersect contour twice")
    arr = np.array(pts)
    g1 = np.array(guess1, float)
    g2 = np.array(guess2, float)
    d1 = np.linalg.norm(arr - g1, axis=1)
    d2 = np.linalg.norm(arr - g2, axis=1)
    i1 = int(d1.argmin())
    i2 = int(d2.argmin())
    if d1[i1] > tol:
        i1 = 0
    if d2[i2] > tol:
        i2 = len(arr) - 1
    if i1 > i2:
        i1, i2 = i2, i1
    return arr[i1], arr[i2]


def compute_apex(
    contour: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Return apex point between ``p1`` and ``p2``."""
    _, foots = project_onto_line(contour, line)
    line_pt = np.asarray(line[0], float)
    line_dir = np.asarray(line[1], float) - line_pt
    denom = float(line_dir.dot(line_dir))
    t_all = ((foots - line_pt) @ line_dir) / denom
    t1 = float(((p1 - line_pt) @ line_dir) / denom)
    t2 = float(((p2 - line_pt) @ line_dir) / denom)
    if t1 > t2:
        t1, t2 = t2, t1
    mask = (t_all >= t1) & (t_all <= t2)
    if not np.any(mask):
        idx = int(np.argmax(np.linalg.norm(contour - line_pt, axis=1)))
        return contour[idx]
    seg = contour[mask]
    dist = np.abs(np.cross(line_dir, seg - line_pt)) / np.linalg.norm(line_dir)
    m = dist.max()
    idxs = np.where(dist == m)[0]
    apex = seg[idxs[len(idxs) // 2]]
    return apex


def annotate_results(
    image: np.ndarray,
    contour: np.ndarray,
    line: tuple[tuple[int, int], tuple[int, int]],
    p1: tuple[int, int],
    p2: tuple[int, int],
    apex: tuple[int, int],
) -> np.ndarray:
    """Return image annotated with contour, contact line and apex."""
    from ...gui.overlay import draw_drop_overlay

    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    axis = (center, apex)
    return draw_drop_overlay(
        image,
        contour,
        diameter_line=(p1, p2),
        axis_line=axis,
        contact_line=(p1, p2),
        apex=apex,
        contact_pts=(p1, p2),
    )


def _nearest_index(contour: np.ndarray, pt: np.ndarray) -> int:
    """Return index of contour point closest to ``pt``."""
    d = np.linalg.norm(contour - pt, axis=1)
    return int(d.argmin())


def compute_contact_angles(
    contour: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    base_diameter_mm: float,
    apex_height_mm: float,
    line: tuple[tuple[float, float], tuple[float, float]],
    window: int = 5,
) -> dict[str, float]:
    """Return outside contact angles and local slopes at ``p1`` and ``p2``."""

    a = base_diameter_mm / 2.0
    h = apex_height_mm
    if h <= 0:
        R = float("inf")
    else:
        R = (a ** 2 + h ** 2) / (2 * h)

    theta_sph = math.degrees(math.acos(max(min((R - h) / R, 1.0), -1.0))) if R != float("inf") else 0.0

    idx1 = _nearest_index(contour, p1)
    idx2 = _nearest_index(contour, p2)
    pts1 = np.vstack([
        contour[(idx1 + i) % len(contour)] for i in range(-window, window + 1)
    ])
    pts2 = np.vstack([
        contour[(idx2 + i) % len(contour)] for i in range(-window, window + 1)
    ])
    vx1, vy1, _, _ = cv2.fitLine(pts1.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx2, vy2, _, _ = cv2.fitLine(pts2.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

    slope1 = float(vy1) / float(vx1) if float(vx1) != 0 else float("inf")
    slope2 = float(vy2) / float(vx2) if float(vx2) != 0 else float("inf")

    theta_slope1 = math.degrees(math.atan(abs(slope1)))
    theta_slope2 = math.degrees(math.atan(abs(slope2)))

    return {
        "theta_spherical_p1": theta_sph,
        "theta_spherical_p2": theta_sph,
        "slope_p1": slope1,
        "slope_p2": slope2,
        "theta_slope_p1": theta_slope1,
        "theta_slope_p2": theta_slope2,
    }


@dataclass
class HelperBundle:
    px_per_mm: float


@dataclass
class SessileMetrics:
    contour: np.ndarray
    apex: tuple[int, int]
    diameter_line: tuple[tuple[int, int], tuple[int, int]]
    p1: tuple[int, int]
    p2: tuple[int, int]
    substrate_line: tuple[tuple[int, int], tuple[int, int]]
    derived: dict[str, float]


def analyze(
    frame: np.ndarray,
    helpers: HelperBundle,
    substrate: tuple[tuple[int, int], tuple[int, int]],
    drop_side: Literal["left", "right", "auto"] = "auto",
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> SessileMetrics:
    """Return sessile-drop metrics and geometry from ``frame``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c.squeeze(1) for c in contours if c.size > 0]
    area_est = mask.shape[0] * mask.shape[1]
    contours = filter_contours_by_size(contours, 10.0, area_est)
    p1_guess = contact_points[0] if contact_points else substrate[0]
    p2_guess = contact_points[1] if contact_points else substrate[1]
    contours = exclude_near_line(contours, substrate, p1_guess, p2_guess)
    if not contours:
        raise ValueError("no droplet region detected")

    mask_clean = np.zeros_like(mask)
    cv2.drawContours(mask_clean, [contours[0].astype(np.int32)], -1, 255, -1)
    clean_contour = extract_outer_contour(mask_clean)

    cp1, cp2 = find_contact_points(clean_contour, substrate, p1_guess, p2_guess)
    apex_pt = compute_apex(clean_contour, substrate, cp1, cp2)

    metrics = compute_sessile_metrics_alt(
        clean_contour.astype(float), helpers.px_per_mm, substrate_line=substrate
    )
    metrics["contact_line"] = (
        (int(round(cp1[0])), int(round(cp1[1]))),
        (int(round(cp2[0])), int(round(cp2[1]))),
    )
    extra = compute_contact_angles(
        clean_contour.astype(float),
        cp1,
        cp2,
        metrics.get("w_mm", 0.0),
        metrics.get("h_mm", 0.0),
        substrate,
    )
    metrics.update(extra)
    return SessileMetrics(
        contour=clean_contour.astype(float),
        apex=(int(round(apex_pt[0])), int(round(apex_pt[1]))),
        diameter_line=metrics["diameter_line"],
        p1=metrics["contact_line"][0],
        p2=metrics["contact_line"][1],
        substrate_line=substrate,
        derived=metrics,
    )


__all__ = [
    "analyze",
    "SessileMetrics",
    "HelperBundle",
    "filter_contours_by_size",
    "exclude_near_line",
    "extract_outer_contour",
    "project_onto_line",
    "find_contact_points",
    "compute_apex",
    "annotate_results",
    "compute_contact_angles",
]
