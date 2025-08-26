from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import cv2

from ...analysis import compute_sessile_metrics_alt
#from ...physics.contact_geom import line_params
from ...detectors.geometry_alt import split_contour_by_line
import math

def line_params(p1_px: tuple[float, float], p2_px: tuple[float, float]) -> tuple[float, float, float]:
    """Return (a, b, c) for line ax + by + c = 0 normalised."""
    x1, y1 = p1_px
    x2, y2 = p2_px
    a, b = y1 - y2, x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm == 0:
        return 0.0, 0.0, 0.0
    return a / norm, b / norm, c / norm


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
    """Return contours not touching ``line`` outside ``p1``/``p2`` segment."""

    a, b, c = line_params(line[0], line[1])
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    seg = p2 - p1
    seg_len_sq = float(seg.dot(seg)) or 1.0
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

        t = ((pts - p1) @ seg) / seg_len_sq
        outside = (t < 0.0) | (t > 1.0)
        if np.any(outside):
            continue

        res.append(cts)

    return res


def extract_outer_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external contour in ``mask``."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")
    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze(1).astype(float)


def clean_droplet_contour(
    binary_image: np.ndarray,
    substrate_y: int,
    min_area: int = 100,
    min_dist: int = 5,
    kernel_size: tuple[int, int] = (3, 3),
    aspect_ratio_range: tuple[float, float] = (0.5, 3.0),
) -> list[np.ndarray]:
    """Return clean droplet contours above ``substrate_y``.

    The input ``binary_image`` is assumed to contain a rough droplet mask. This
    helper removes any pixels touching or below the substrate line, performs
    morphological cleanup, keeps only the largest connected component and
    filters the resulting contours by distance from the substrate, area and
    aspect ratio.
    """

    if binary_image.ndim == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.ones_like(binary, dtype=np.uint8) * 255
    mask[substrate_y:, :] = 0
    cleaned = cv2.bitwise_and(binary, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels <= 1:
        return []
    largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    cleaned = (labels == largest).astype(np.uint8) * 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours: list[np.ndarray] = []
    ar_min, ar_max = aspect_ratio_range
    for cnt in contours:
        if np.min(cnt[:, :, 1]) >= substrate_y - min_dist:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0.0
        if not (ar_min <= ar <= ar_max):
            continue
        final_contours.append(cnt.squeeze(1).astype(float))

    return final_contours


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
    tol: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered contact points along ``line``.

    Contour points more than ``tol`` pixels below the substrate line are
    discarded. The remaining points closest to the user provided guesses
    ``guess1`` and ``guess2`` are selected, prioritizing minimal vertical
    distance to the line and then horizontal proximity to the guesses.
    """

    contour = np.asarray(contour, float)
    p1 = np.asarray(line[0], float)
    p2 = np.asarray(line[1], float)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        slope = float("inf")
    else:
        slope = dy / dx

    def y_on_line(x: np.ndarray) -> np.ndarray:
        return p1[1] + slope * (x - p1[0])

    ys_line = y_on_line(contour[:, 0])
    delta = contour[:, 1] - ys_line
    mask = delta <= tol
    if not np.any(mask):
        raise ValueError("no contour points near substrate line")

    cont = contour[mask]
    delta = delta[mask]
    near_mask = np.abs(delta) <= tol
    cont_near = cont[near_mask]
    if len(cont_near) == 0:
        cont_near = cont

    g1 = np.asarray(guess1, float)
    g2 = np.asarray(guess2, float)

    y_line = y_on_line(cont_near[:, 0])
    vert_dist = np.abs(cont_near[:, 1] - y_line)
    _, foots = project_onto_line(cont_near, line)

    def _select(pt: np.ndarray) -> np.ndarray:
        x_diff = np.abs(foots[:, 0] - pt[0])
        mask_close = vert_dist <= tol
        if np.any(mask_close):
            idx_pool = np.where(mask_close)[0]
        else:
            idx_pool = np.arange(len(foots))
        idx = idx_pool[np.argmin(x_diff[idx_pool])]
        return foots[idx]

    p_left = _select(g1)
    p_right = _select(g2)

    if p_left[0] > p_right[0]:
        p_left, p_right = p_right, p_left

    return p_left, p_right


def compute_apex(
    contour: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Return apex point between ``p1`` and ``p2``.

    The apex is chosen as the contour point between the contact points that has
    the maximum perpendicular distance to the substrate line. If several points
    share this maximum distance, the one whose ``x`` coordinate is closest to the
    average ``x`` of those candidates is returned.
    """
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
    idxs = np.where(np.isclose(dist, m))[0]
    seg_max = seg[idxs]
    # prefer points above the substrate line
    y_min = seg_max[:, 1].min()
    idxs = idxs[np.isclose(seg_max[:, 1], y_min)]
    if len(idxs) == 1:
        apex = seg[idxs[0]]
    else:
        xs = seg[idxs, 0]
        x_target = xs.mean()
        idx = idxs[np.argmin(np.abs(xs - x_target))]
        apex = seg[idx]
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
    drop_side: Literal["above", "below", "auto"] = "auto",
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> SessileMetrics:
    """Return sessile-drop metrics and geometry from ``frame``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)

    substrate_y = int(round((substrate[0][1] + substrate[1][1]) / 2))
    contours = clean_droplet_contour(mask, substrate_y)
    if contours:
        clean_contour = max(contours, key=cv2.contourArea)
    else:
        try:
            clean_contour = extract_outer_contour(mask)
        except ValueError:
            raise ValueError("no droplet region detected") from None

    keep_above: bool | None = None
    if drop_side != "auto":
        keep_above = drop_side == "above"
        line_pt = np.asarray(substrate[0], float)
        line_dir = np.subtract(substrate[1], substrate[0])
        clean_contour = split_contour_by_line(
            clean_contour.astype(float), line_pt, line_dir, keep_above=keep_above
        )

    p1_guess = contact_points[0] if contact_points else substrate[0]
    p2_guess = contact_points[1] if contact_points else substrate[1]

    cp1, cp2 = find_contact_points(clean_contour, substrate, p1_guess, p2_guess)
    apex_pt = compute_apex(clean_contour, substrate, cp1, cp2)

    metrics = compute_sessile_metrics_alt(
        clean_contour.astype(float),
        helpers.px_per_mm,
        substrate_line=substrate,
        keep_above=keep_above,
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
    "clean_droplet_contour",
    "project_onto_line",
    "find_contact_points",
    "compute_apex",
    "annotate_results",
    "compute_contact_angles",
]
