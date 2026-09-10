"""Separate the free liquid surface from a segmented solid-contact return path."""

import numpy as np

from menipy.common.cancellation import check_cancelled


def liquid_surface_arc(contour, contacts, *, liquid_below=False):
    """Return P1 → outer surface → P2, implicitly closed by the contact chord.

    Keep the existing cyclic point order and measured surface samples. Contact
    endpoints are geometry constraints, not extra measured edge samples. The
    solid side and the alternate segmentation path (including contact-band
    texture/highlight edges) are excluded. Missing/degenerate geometry is left
    unchanged for callers to handle through their existing validation.
    """
    check_cancelled()
    points = np.asarray(contour, dtype=float).reshape(-1, 2)
    if contacts is None or len(points) < 3:
        return points.copy()
    p1, p2 = np.asarray(contacts, dtype=float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length <= 1e-12 or not np.all(np.isfinite(points)):
        return points.copy()
    normal = np.array([-direction[1], direction[0]]) / length
    if normal[1] < 0:
        normal = -normal
    if not liquid_below:
        normal = -normal
    start = int(np.argmin(np.sum((points - p1) ** 2, axis=1)))
    end = int(np.argmin(np.sum((points - p2) ** 2, axis=1)))
    if start == end:
        return points.copy()
    forward = points[(start + np.arange((end - start) % len(points) + 1)) % len(points)]
    backward = points[
        (start - np.arange((start - end) % len(points) + 1)) % len(points)
    ]

    def score(arc):
        distance = (arc - p1) @ normal
        return float(np.max(distance)), float(np.mean(distance))

    arc = max((forward, backward), key=score)
    # The contact chord is the sole closure. Do not retain a solid-side tail.
    arc = arc[((arc - p1) @ normal) > 1e-9]
    check_cancelled()
    if len(arc) < 1:
        return points.copy()
    boundary = np.vstack([p1, arc, p2])
    boundary = boundary[np.r_[True, np.any(np.diff(boundary, axis=0) != 0, axis=1)]]
    # Exact simplification only: no pixel tolerance or curve approximation.
    previous = boundary[1:-1] - boundary[:-2]
    following = boundary[2:] - boundary[1:-1]
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    forward = np.sum(previous * following, axis=1) >= 0
    keep = np.r_[True, (cross != 0) | ~forward, True]
    return boundary[keep]


def update_calibration_boundary(result, pipeline):
    """Build the display region without changing scientific contour samples."""
    result.liquid_boundary = None
    if result.drop_contour is None or result.contact_points is None:
        return
    below = result.confidence_scores.get("detector_pipeline", pipeline) == "pendant"
    contacts = np.asarray(result.contact_points, dtype=float).copy()
    if not below and result.substrate_line is not None:
        a, b = np.asarray(result.substrate_line, dtype=float)
        direction = b - a
        squared = float(direction @ direction)
        if squared > 1e-12:
            contacts = a + ((contacts - a) @ direction)[:, None] * direction / squared
    result.liquid_boundary = liquid_surface_arc(
        result.drop_contour, contacts, liquid_below=below
    )
