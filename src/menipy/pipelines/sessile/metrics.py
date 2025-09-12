import numpy as np

from menipy.analysis.commons import compute_drop_metrics, find_apex_index
from .geometry_alt import line_params


def compute_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    *,
    keep_above: bool | None = None,
) -> dict:
    """Return sessile-drop metrics for ``contour``.

    Parameters
    ----------
    contour:
        Droplet contour points ``(x, y)``.
    px_per_mm:
        Calibration factor in pixels per millimetre.
    substrate_line:
        Two points defining the substrate line.
    keep_above:
        If ``True`` keep contour points above the line, if ``False`` keep the
        points below it. ``None`` selects the side with the larger area.
    """
    if substrate_line is None:
        return compute_drop_metrics(contour, px_per_mm, "contact-angle")

    apex_idx = find_apex_index(contour, "contact-angle")
    poly = np.array([substrate_line[0], substrate_line[1]], float)

    geo = geom_metrics_alt(poly, contour, px_per_mm, keep_above=keep_above)
    droplet_poly = geo.pop("droplet_poly")
    metrics = compute_drop_metrics(
        droplet_poly.astype(float),
        px_per_mm,
        "contact-angle",
        substrate_line=substrate_line,
    )
    metrics.update(geo)

    try:
        cp1, cp2 = contact_points_from_spline(contour, substrate_line, delta=0.5)
        metrics["contact_line"] = (
            (int(round(cp1[0])), int(round(cp1[1]))),
            (int(round(cp2[0])), int(round(cp2[1]))),
        )
    except Exception:
        pass

    return metrics

__all__ = ["compute_metrics"]



def geom_metrics_alt(
    substrate_poly: np.ndarray,
    contour_px: np.ndarray,
    px_per_mm: float,
    *,
    keep_above: bool | None = None,
) -> dict:
    """Return geometric metrics relative to a substrate polyline."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")

    line_pt = substrate_poly[0]
    line_dir = substrate_poly[-1] - substrate_poly[0]

    p1, p2 = find_contact_points(contour_px, line_pt, line_dir)
    contact_seg = trim_poly_between(substrate_poly, p1, p2)

    if keep_above is None:
        cont_above = split_contour_by_line(contour_px, line_pt, line_dir, keep_above=True)
        poly_above = np.vstack([cont_above, contact_seg[::-1]])
        area_above = _polygon_area(poly_above) if len(poly_above) >= 3 else 0.0

        cont_below = split_contour_by_line(contour_px, line_pt, line_dir, keep_above=False)
        poly_below = np.vstack([cont_below, contact_seg[::-1]])
        area_below = _polygon_area(poly_below) if len(poly_below) >= 3 else 0.0

        keep_above = area_above >= area_below
        droplet_contour = cont_above if keep_above else cont_below
        droplet_poly = poly_above if keep_above else poly_below
    else:
        droplet_contour = split_contour_by_line(
            contour_px, line_pt, line_dir, keep_above=keep_above
        )
        droplet_poly = np.vstack([droplet_contour, contact_seg[::-1]])

    mode = "sessile" if keep_above else "pendant"
    apex_px, _ = apex_point(droplet_contour, line_pt, line_dir, mode)

    a, b, c = line_params(tuple(p1), tuple(p2))
    h_px = abs(a * apex_px[0] + b * apex_px[1] + c)
    w_px = np.linalg.norm(p2 - p1)

    w_mm = w_px / px_per_mm
    rb_mm = w_mm / 2.0
    h_mm = h_px / px_per_mm

    _, foot = project_pts_onto_poly(np.array([apex_px]), substrate_poly)
    ratio = symmetry_area_ratio(droplet_poly, apex_px, foot[0])

    return {
        "xL_px": float(p1[0]),
        "xR_px": float(p2[0]),
        "w_mm": float(w_mm),
        "rb_mm": float(rb_mm),
        "h_mm": float(h_mm),
        "droplet_poly": droplet_poly,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "contact_segment": contact_seg,
        "symmetry_ratio": float(ratio),
        "apex": (int(round(apex_px[0])), int(round(apex_px[1]))),
    }