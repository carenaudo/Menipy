"""Separate the free liquid surface from a segmented solid-contact return path."""

import numpy as np

from menipy.common.cancellation import check_cancelled
from menipy.common.geometry import cross2d
from menipy.models.geometry import LiquidGeometry


def _dedupe_consecutive(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    return points[np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]]


def _path(points: np.ndarray, start: int, end: int, step: int) -> np.ndarray:
    indexes = (start + step * np.arange((step * (end - start)) % len(points) + 1)) % len(points)
    return points[indexes]


def build_straight_liquid_geometry(
    contour, contact_line, *, apex=None, contact_points=None, provenance: str = "automatic"
) -> LiquidGeometry:
    """Validate the physical liquid region bounded by a straight contact line.

    The apex defines the allowed half-plane.  Exactly two crossings are required
    for a complete region; otherwise callers receive an unresolved geometry
    rather than a contour closed through the solid/reflection side.
    """
    check_cancelled()
    points = _dedupe_consecutive(np.asarray(contour, dtype=float).reshape(-1, 2))
    boundary = np.asarray(contact_line, dtype=float).reshape(-1, 2)
    base = LiquidGeometry(
        boundary_kind="line",
        boundary_provenance="manual" if provenance == "manual" else "automatic",
        contact_boundary=[tuple(map(float, point)) for point in boundary[:2]],
        apex=tuple(map(float, apex)) if apex is not None else None,
    )
    if len(points) < 3:
        base.rejection_reasons.append("liquid_contour_insufficient_points")
        return base
    if len(boundary) != 2:
        base.rejection_reasons.append("contact_line_invalid")
        return base
    p1, p2 = boundary
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        base.rejection_reasons.append("contact_line_degenerate")
        return base
    if apex is None:
        base.status = "partial"
        base.rejection_reasons.append("liquid_apex_unavailable")
        return base
    apex_point = np.asarray(apex, dtype=float)
    apex_side = float(cross2d(direction, apex_point - p1))
    tolerance = max(1e-7, length * 1e-8)
    if abs(apex_side) <= tolerance:
        base.rejection_reasons.append("apex_on_contact_boundary")
        return base
    sign = 1.0 if apex_side > 0 else -1.0

    # Insert line crossings into the cyclic contour while retaining source order.
    expanded: list[np.ndarray] = []
    crossings: list[int] = []
    for index, point in enumerate(points):
        expanded.append(point)
        next_point = points[(index + 1) % len(points)]
        d0 = float(cross2d(direction, point - p1)) * sign
        d1 = float(cross2d(direction, next_point - p1)) * sign
        if (d0 > tolerance and d1 < -tolerance) or (d0 < -tolerance and d1 > tolerance):
            fraction = d0 / (d0 - d1)
            crossing = point + fraction * (next_point - point)
            expanded.append(crossing)
            crossings.append(len(expanded) - 1)
    expanded_array = _dedupe_consecutive(np.asarray(expanded, dtype=float))
    # Deduplication only removes adjacent samples. Remap crossing positions by coordinate.
    crossing_indexes = [
        int(np.argmin(np.sum((expanded_array - expanded[index]) ** 2, axis=1)))
        for index in crossings
    ]
    crossing_indexes = list(dict.fromkeys(crossing_indexes))
    if len(crossing_indexes) != 2:
        # Edge detectors may report a noisy substrate return path with many
        # crossings. A pair of independently detected, line-projected contacts
        # may disambiguate it, but only if one cyclic arc is wholly apex-side.
        supplied = None
        if contact_points is not None:
            try:
                supplied = np.asarray(contact_points, dtype=float).reshape(2, 2)
            except (TypeError, ValueError):
                supplied = None
        if supplied is None:
            base.rejection_reasons.append("contact_boundary_ambiguous_crossings")
            return base
        unit = direction / length
        projected = np.asarray(
            [p1 + np.dot(point - p1, unit) * unit for point in supplied], dtype=float
        )
        start = int(np.argmin(np.sum((points - projected[0]) ** 2, axis=1)))
        end = int(np.argmin(np.sum((points - projected[1]) ** 2, axis=1)))
        if start == end:
            base.rejection_reasons.append("contact_boundary_ambiguous_crossings")
            return base
        candidate_paths = (_path(points, start, end, 1), _path(points, start, end, -1))
        valid_paths = []
        for candidate in candidate_paths:
            signed = np.asarray(cross2d(direction, candidate - p1), dtype=float) * sign
            if np.all(signed >= -tolerance):
                valid_paths.append(candidate)
        if len(valid_paths) != 1:
            base.rejection_reasons.append("contact_boundary_ambiguous_crossings")
            return base
        selected = valid_paths[0]
        if float(np.min(np.sum((selected - apex_point) ** 2, axis=1))) > max(9.0, length * 0.1) ** 2:
            base.rejection_reasons.append("apex_not_on_observed_liquid_surface")
            return base
        if float(np.dot(projected[0] - p1, direction)) > float(np.dot(projected[1] - p1, direction)):
            projected = projected[::-1]
            selected = selected[::-1]
        base.contact_points = (tuple(map(float, projected[0])), tuple(map(float, projected[1])))
        base.observed_surface = np.asarray(selected[1:-1], dtype=float)
        base.closed_region = np.vstack([projected[0], base.observed_surface, projected[1], projected[0]])
        base.status = "complete"
        return base

    first, second = crossing_indexes
    forward = _path(expanded_array, first, second, 1)
    backward = _path(expanded_array, first, second, -1)
    candidates = (forward, backward)
    valid: list[np.ndarray] = []
    for candidate in candidates:
        signed = np.asarray(cross2d(direction, candidate - p1), dtype=float) * sign
        if np.all(signed >= -tolerance):
            valid.append(candidate)
    if len(valid) != 1:
        base.rejection_reasons.append("liquid_surface_disconnected_or_ambiguous")
        return base
    surface_with_contacts = valid[0]
    if len(surface_with_contacts) < 3:
        base.rejection_reasons.append("liquid_surface_insufficient_points")
        return base
    # The apex must lie on the chosen, observed side rather than on its closure.
    if float(np.min(np.sum((surface_with_contacts - apex_point) ** 2, axis=1))) > max(9.0, length * 0.1) ** 2:
        base.rejection_reasons.append("apex_not_on_observed_liquid_surface")
        return base
    contacts = (tuple(map(float, surface_with_contacts[0])), tuple(map(float, surface_with_contacts[-1])))
    ordered = sorted(contacts, key=lambda point: float(np.dot(np.asarray(point) - p1, direction)))
    if ordered[0] != contacts[0]:
        surface_with_contacts = surface_with_contacts[::-1]
        contacts = (tuple(map(float, surface_with_contacts[0])), tuple(map(float, surface_with_contacts[-1])))
    base.contact_points = contacts
    base.observed_surface = np.asarray(surface_with_contacts[1:-1], dtype=float)
    # This is the only closed polygon. Its closing edge is the contact boundary;
    # it is intentionally separate from observed_surface.
    base.closed_region = np.vstack([surface_with_contacts, surface_with_contacts[0]])
    base.status = "complete"
    return base


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
    """Build validated liquid geometry and a display-only closed boundary.

    Legacy callers still receive ``liquid_boundary``.  New callers must use
    ``liquid_geometry.observed_surface`` for measurement and never fit the
    closing contact segment.
    """
    result.liquid_boundary = None
    result.liquid_geometry = None
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
    apex = getattr(result, "apex_point", None)
    geometry = build_straight_liquid_geometry(
        result.drop_contour,
        contacts,
        apex=apex,
        contact_points=getattr(result, "contact_points", None),
        provenance="manual" if "substrate" in getattr(result, "manual_regions", []) else "automatic",
    )
    result.liquid_geometry = geometry
    if geometry.status == "complete" and geometry.closed_region is not None:
        result.liquid_boundary = geometry.closed_region[:-1]
        result.contact_points = geometry.contact_points
        return
    # Retain legacy preview for unresolved images but make it explicitly
    # non-authoritative through liquid_geometry.status/reasons.
    result.liquid_boundary = liquid_surface_arc(result.drop_contour, contacts, liquid_below=below)
