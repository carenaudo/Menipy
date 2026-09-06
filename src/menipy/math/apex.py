"""Droplet apex detection and sub-pixel summit refinement numerical methods.

Academic References:
    1. Rotenberg, Y., Boruvka, L., & Neumann, A. W. (1983).
       "Determination of surface tension and contact angle from the shapes of
       axisymmetric fluid interfaces."
       Journal of Colloid and Interface Science, 93(1), 169-183.
       DOI: 10.1016/0021-9797(83)90396-X

    2. Song, B., & Springer, J. (1996).
       "Determination of interfacial tension from the profile of a pendant drop
       using computer aided image processing: 1. Theoretical."
       Colloids and Surfaces A: Physicochemical and Engineering Aspects, 112(1), 41-52.
       DOI: 10.1016/0927-7757(95)03478-2

    3. Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y., & Tabor, R. F. (2015).
       "Measurement of surface and interfacial tension using pendant drop tensiometry."
       Journal of Colloid and Interface Science, 454, 226-237.
       DOI: 10.1016/j.jcis.2015.05.012

    4. Extrand, C. W., & Moon, M. W. (2008).
       "Indirect Measurement of Contact Angles on Curved Surfaces."
       Langmuir, 24(17), 9470-9473.
       DOI: 10.1021/la801091m

    5. Carroll, B. J. (1976).
       "The accurate measurement of contact angle, phase volume, and surface area
       of drops on cylindrical fibers."
       Journal of Colloid and Interface Science, 57(3), 488-495.
       DOI: 10.1016/0021-9797(76)90227-7

Attribution & Clean-Room Implementation:
    Independent clean-room Python/NumPy/SciPy implementation authored for Menipy under MIT license.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApexResult:
    """Result of droplet apex detection and sub-pixel refinement."""

    point: tuple[float, float]
    confidence: float
    method: str
    r0_px: float | None = None
    asymmetry_px: float = 0.0
    curvature_kappa: float | None = None
    band_points: int = 1
    tangent: tuple[float, float] = (1.0, 0.0)
    normal: tuple[float, float] = (0.0, -1.0)

    @property
    def x(self) -> float:
        return self.point[0]

    @property
    def y(self) -> float:
        return self.point[1]

    def to_int_tuple(self) -> tuple[int, int]:
        """Return integer pixel coordinates for legacy API compatibility."""
        return (int(round(self.point[0])), int(round(self.point[1])))

    def __iter__(self):
        """Allow unpacking as a 2-tuple: x, y = apex_res."""
        return iter(self.point)

    def __getitem__(self, index: int) -> float:
        """Allow indexing as a 2-tuple: apex_res[0], apex_res[1]."""
        return self.point[index]


def _ensure_2d_contour(contour: np.ndarray | Any) -> np.ndarray:
    """Normalize contour into a clean (N, 2) float array."""
    pts = np.asarray(contour, dtype=float)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    elif pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Contour must have shape (N, 2), got {pts.shape}")
    return pts


def detect_apex_flat(
    contour: np.ndarray,
    mode: str = "sessile",
    band_px: float = 1.0,
) -> ApexResult:
    """Detect apex on an un-tilted substrate using multi-point crest averaging.

    When multiple discrete points share the same minimum or maximum vertical
    coordinate (pixel grid discretization or flat crest), this algorithm collects
    all points within an epsilon band and returns the horizontal median/mean.
    This resolves the leftmost-index bias inherent to raw argmin/argmax.

    Args:
        contour: (N, 2) array of contour points.
        mode: Drop mode ('sessile', 'pendant', 'captive_bubble', 'capillary_rise').
        band_px: Vertical tolerance in pixels for collecting crest points.

    Returns:
        ApexResult containing apex coordinates, confidence, and method.
    """
    pts = _ensure_2d_contour(contour)
    if pts.shape[0] == 0:
        raise ValueError("Empty contour provided to apex detection")

    is_apex_at_bottom = mode.lower() in ("pendant", "captive_bubble")

    if is_apex_at_bottom:
        y_ext = float(np.max(pts[:, 1]))
        pts_exact = pts[np.abs(pts[:, 1] - y_ext) <= 1e-4]
        band = pts[pts[:, 1] >= (y_ext - max(0.25, float(band_px)))]
    else:
        y_ext = float(np.min(pts[:, 1]))
        pts_exact = pts[np.abs(pts[:, 1] - y_ext) <= 1e-4]
        band = pts[pts[:, 1] <= (y_ext + max(0.25, float(band_px)))]

    if pts_exact.shape[0] >= 2:
        # Multiple points share the exact same vertical value (flat crest):
        apex_x = float(np.median(pts_exact[:, 0]))
        apex_y = y_ext
    elif pts_exact.shape[0] == 1:
        # Unique discrete extreme point:
        apex_x = float(pts_exact[0, 0])
        apex_y = float(pts_exact[0, 1])
    else:
        apex_x = float(np.median(band[:, 0]))
        apex_y = y_ext

    normal_dir = (0.0, 1.0) if is_apex_at_bottom else (0.0, -1.0)
    return ApexResult(
        point=(apex_x, apex_y),
        confidence=0.92,
        method="flat_crest_median",
        band_points=int(max(band.shape[0], pts_exact.shape[0])),
        tangent=(1.0, 0.0),
        normal=normal_dir,
    )


def detect_apex_normal(
    contour: np.ndarray,
    baseline: tuple[tuple[float, float], tuple[float, float]],
    mode: str = "sessile",
    band_px: float = 1.0,
) -> ApexResult:
    """Detect apex along the inward substrate normal for planar tilted substrates.

    On a tilted or inclined substrate, the highest point in image Y does not
    coincide with the droplet summit. The physical apex is the point with the
    maximum perpendicular distance from the substrate baseline line.

    Args:
        contour: (N, 2) array of contour points.
        baseline: Substrate baseline chord ((x1, y1), (x2, y2)).
        mode: Drop mode ('sessile', 'pendant', 'captive_bubble', etc.).
        band_px: Tolerance in pixels for multi-point crest averaging.

    Returns:
        ApexResult with detected apex, normal/tangent vectors, and band points.
    """
    pts = _ensure_2d_contour(contour)
    p1 = np.asarray(baseline[0], dtype=float)
    p2 = np.asarray(baseline[1], dtype=float)

    base_vec = p2 - p1
    length = float(np.linalg.norm(base_vec))
    if length <= 1e-9:
        return detect_apex_flat(pts, mode=mode, band_px=band_px)

    # Unit vector along substrate (left to right)
    if base_vec[0] < 0:
        p1, p2 = p2, p1
        base_vec = -base_vec

    u = base_vec / length

    # Normal vector perpendicular to substrate
    n_candidate1 = np.array([-u[1], u[0]], dtype=float)
    n_candidate2 = -n_candidate1

    is_bottom_apex = mode.lower() in ("pendant", "captive_bubble")
    if is_bottom_apex:
        n = n_candidate1 if n_candidate1[1] > 0 else n_candidate2
    else:
        n = n_candidate1 if n_candidate1[1] < 0 else n_candidate2

    # Project contour points onto normal direction: h = (p - p1) . n
    diffs = pts - p1
    h_perps = np.dot(diffs, n)

    h_max = float(np.max(h_perps))
    exact_mask = np.abs(h_perps - h_max) <= 1e-4
    band_mask = h_perps >= (h_max - max(0.25, float(band_px)))

    if np.sum(exact_mask) >= 2:
        band_pts = pts[exact_mask]
        mean_h = h_max
    elif np.sum(exact_mask) == 1:
        band_pts = pts[exact_mask]
        mean_h = h_max
    else:
        band_pts = pts[band_mask]
        mean_h = float(np.mean(h_perps[band_mask]))

    u_coords = np.dot(band_pts - p1, u)
    median_u = float(np.median(u_coords))

    apex_coords = p1 + median_u * u + mean_h * n
    return ApexResult(
        point=(float(apex_coords[0]), float(apex_coords[1])),
        confidence=0.95,
        method="normal_projection",
        band_points=int(band_pts.shape[0]),
        tangent=(float(u[0]), float(u[1])),
        normal=(float(n[0]), float(n[1])),
    )


def detect_apex_curved_substrate(
    contour: np.ndarray,
    substrate: Any,
    mode: str = "sessile",
) -> ApexResult:
    """Detect apex on curved substrates (cylinders, spheres, fibers, circular arcs).

    For convex curved substrates (e.g. droplet on top of a cylinder or sphere),
    the apex is the point maximizing radial clearance from the substrate center:
        d_i = ||p_i - c_sub|| - R_sub

    For concave substrates (droplet inside a well):
        d_i = R_sub - ||p_i - c_sub||

    Args:
        contour: (N, 2) array of contour points.
        substrate: SubstrateProfile or dictionary containing substrate geometry.
        mode: Drop mode ('sessile', 'pendant', etc.).

    Returns:
        ApexResult containing apex coordinates and method.
    """
    pts = _ensure_2d_contour(contour)

    # Check if substrate is a circle arc
    sub_type = getattr(substrate, "type", None) or (substrate.get("type") if isinstance(substrate, dict) else None)
    params = getattr(substrate, "parameters", None) or (substrate.get("parameters") if isinstance(substrate, dict) else {})
    coeffs = getattr(substrate, "coefficients", None) or (substrate.get("coefficients") if isinstance(substrate, dict) else {})

    cx = params.get("center_x")
    cy = params.get("center_y")
    r_sub = params.get("radius")
    if cx is None and "center" in coeffs:
        cx, cy = coeffs["center"][0], coeffs["center"][1]
        r_sub = coeffs.get("radius")

    if sub_type in ("circle_arc", "circle") and cx is not None and cy is not None and r_sub is not None:
        c_sub = np.array([float(cx), float(cy)], dtype=float)
        r_sub = float(r_sub)
        is_convex = (params.get("convex", 1.0) > 0) if "convex" in params else params.get("is_convex", True)

        dists = np.linalg.norm(pts - c_sub, axis=1)
        if is_convex:
            clearance = dists - r_sub
        else:
            clearance = r_sub - dists

        max_c = float(np.max(clearance))
        exact_mask = np.abs(clearance - max_c) <= 1e-4
        band = pts[exact_mask] if np.sum(exact_mask) >= 2 else pts[clearance >= (max_c - 1.0)]
        apex = (float(np.median(band[:, 0])), float(np.median(band[:, 1])))
        return ApexResult(point=apex, confidence=0.92, method="curved_circle_arc", band_points=len(band))

    # General curved substrate with eval_y method
    if hasattr(substrate, "eval_y") and callable(substrate.eval_y):
        try:
            clearances = []
            valid_indices = []
            for i, p in enumerate(pts):
                y_sub = substrate.eval_y(float(p[0]))
                if y_sub is not None:
                    clearances.append(abs(float(y_sub) - float(p[1])))
                    valid_indices.append(i)

            if len(clearances) >= 3:
                clearances_arr = np.asarray(clearances, dtype=float)
                max_c = float(np.max(clearances_arr))
                band_idx = [valid_indices[k] for k in np.where(clearances_arr >= (max_c - 1.0))[0]]
                band = pts[band_idx]
                apex = (float(np.median(band[:, 0])), float(np.median(band[:, 1])))
                return ApexResult(point=apex, confidence=0.90, method="curved_eval_y", band_points=len(band))
        except Exception as e:
            logger.debug(f"Curved substrate eval_y fallback: {e}")

    # Fallback to substrate chord baseline normal
    if hasattr(substrate, "to_chord") and callable(substrate.to_chord):
        chord = substrate.to_chord()
        return detect_apex_normal(pts, chord, mode=mode)

    # Default fallback
    return detect_apex_flat(pts, mode=mode)


def refine_apex_polynomial(
    contour: np.ndarray,
    initial_apex: tuple[float, float] | ApexResult,
    baseline: tuple[tuple[float, float], tuple[float, float]] | None = None,
    mode: str = "sessile",
    window_px: float = 15.0,
    order: int = 2,
) -> ApexResult:
    """Refine droplet apex to sub-pixel coordinates using local polynomial modeling.

    Transforms contour points in a local window around `initial_apex` into the
    substrate-aligned coordinate frame (tangential xi, normal eta). Fits an
    order-2 (or order-3) polynomial:
        eta(xi) = a * xi^2 + b * xi + c (+ d * xi^3)

    Analytically locates the continuous peak:
        xi* = -b / (2 * a)  [order 2] or cubic stationary root [order 3]
        eta* = a * (xi*)^2 + b * xi* + c

    This provides:
    1. Sub-pixel continuous coordinates: p_apex = p_init + xi* * u + eta* * n.
    2. Apex radius of curvature: R0 = 1 / |2 * a|.
    3. Asymmetry offset: xi* directly quantifies crest shift for asymmetric drops.

    Args:
        contour: (N, 2) array of contour points.
        initial_apex: Initial candidate apex (x, y) or ApexResult.
        baseline: Optional baseline chord to align tangent/normal axes.
        mode: Drop mode ('sessile', 'pendant', 'captive_bubble', etc.).
        window_px: Search radius in pixels around initial_apex.
        order: Polynomial fit order (2 for parabolic, 3 for asymmetric cubic).

    Returns:
        ApexResult containing refined sub-pixel coordinates, R0, asymmetry, and confidence.
    """
    pts = _ensure_2d_contour(contour)
    if isinstance(initial_apex, ApexResult):
        p_init = np.asarray(initial_apex.point, dtype=float)
    else:
        p_init = np.asarray(initial_apex, dtype=float)

    is_bottom_apex = mode.lower() in ("pendant", "captive_bubble")

    # Frame orientation vectors
    if baseline is not None:
        p1 = np.asarray(baseline[0], dtype=float)
        p2 = np.asarray(baseline[1], dtype=float)
        base_vec = p2 - p1
        length = float(np.linalg.norm(base_vec))
        if length > 1e-9:
            u = base_vec / length
            if u[0] < 0:
                u = -u
            n = np.array([-u[1], u[0]], dtype=float)
            if np.dot(p_init - p1, n) < 0:
                n = -n
        else:
            u = np.array([1.0, 0.0], dtype=float)
            n = np.array([0.0, 1.0], dtype=float) if is_bottom_apex else np.array([0.0, -1.0], dtype=float)
    else:
        u = np.array([1.0, 0.0], dtype=float)
        n = np.array([0.0, 1.0], dtype=float) if is_bottom_apex else np.array([0.0, -1.0], dtype=float)

    # Extract points in local Euclidean window
    dists = np.linalg.norm(pts - p_init, axis=1)
    w_px = max(5.0, float(window_px))
    mask = dists <= w_px
    local_pts = pts[mask]

    if local_pts.shape[0] < 5:
        mask = dists <= (w_px * 2.0)
        local_pts = pts[mask]

    if local_pts.shape[0] < 5:
        return ApexResult(
            point=(float(p_init[0]), float(p_init[1])),
            confidence=0.5,
            method="initial_fallback",
            asymmetry_px=0.0,
            band_points=int(local_pts.shape[0]),
            tangent=(float(u[0]), float(u[1])),
            normal=(float(n[0]), float(n[1])),
        )

    # Transform into local (xi, eta) coordinates
    diffs = local_pts - p_init
    xi = np.dot(diffs, u)
    eta = np.dot(diffs, n)

    fit_order = min(order, 3) if len(local_pts) >= 7 else 2
    try:
        coeffs = np.polyfit(xi, eta, fit_order)
    except (np.linalg.LinAlgError, ValueError):
        return ApexResult(
            point=(float(p_init[0]), float(p_init[1])),
            confidence=0.5,
            method="fit_error_fallback",
            band_points=int(local_pts.shape[0]),
            tangent=(float(u[0]), float(u[1])),
            normal=(float(n[0]), float(n[1])),
        )

    if fit_order == 2:
        d_cub = 0.0
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    elif fit_order == 3:
        d_cub, a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
    else:
        d_cub, a, b, c = 0.0, 0.0, 0.0, 0.0

    if a >= -1e-8:
        apex_cand = local_pts[np.argmax(eta)]
        return ApexResult(
            point=(float(apex_cand[0]), float(apex_cand[1])),
            confidence=0.6,
            method="local_max_fallback",
            asymmetry_px=0.0,
            band_points=int(local_pts.shape[0]),
            tangent=(float(u[0]), float(u[1])),
            normal=(float(n[0]), float(n[1])),
        )

    if fit_order == 3 and abs(d_cub) > 1e-7:
        disc = 4.0 * (a**2) - 12.0 * d_cub * b
        if disc >= 0:
            root1 = (-2.0 * a + np.sqrt(disc)) / (6.0 * d_cub)
            root2 = (-2.0 * a - np.sqrt(disc)) / (6.0 * d_cub)
            xi_candidates = [r for r in (root1, root2) if (6.0 * d_cub * r + 2.0 * a) < 0]
            if xi_candidates:
                xi_star = min(xi_candidates, key=abs)
            else:
                xi_star = -b / (2.0 * a)
        else:
            xi_star = -b / (2.0 * a)
    else:
        xi_star = -b / (2.0 * a)

    # Gate: xi* must remain within the local window
    if abs(xi_star) > w_px:
        xi_star = float(np.clip(xi_star, -w_px * 0.5, w_px * 0.5))

    eta_star = a * (xi_star**2) + b * xi_star + c
    if fit_order == 3:
        eta_star += d_cub * (xi_star**3)

    p_refined = p_init + xi_star * u + eta_star * n

    # Apex curvature radius R0 = 1 / |2a|
    kappa_0 = abs(2.0 * a)
    r0_px = (1.0 / kappa_0) if kappa_0 > 1e-9 else None

    eta_pred = np.polyval(coeffs, xi)
    ss_tot = float(np.sum((eta - np.mean(eta)) ** 2))
    ss_res = float(np.sum((eta - eta_pred) ** 2))
    r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.8
    confidence = float(np.clip(0.5 + 0.49 * r2, 0.5, 0.99))

    return ApexResult(
        point=(float(p_refined[0]), float(p_refined[1])),
        confidence=confidence,
        method=f"subpixel_poly{fit_order}",
        r0_px=float(r0_px) if r0_px is not None else None,
        asymmetry_px=float(xi_star),
        curvature_kappa=float(kappa_0),
        band_points=int(local_pts.shape[0]),
        tangent=(float(u[0]), float(u[1])),
        normal=(float(n[0]), float(n[1])),
    )


def detect_apex_symmetry_axis(
    contour: np.ndarray,
    axis_direction: tuple[float, float] = (0.0, -1.0),
    axis_origin: tuple[float, float] | None = None,
    mode: str = "sessile",
) -> ApexResult:
    """Detect apex from symmetry axis intersection with contour envelope.

    Args:
        contour: (N, 2) array of contour points.
        axis_direction: Unit direction of the symmetry axis.
        axis_origin: Origin point on the symmetry axis.
        mode: Drop mode ('sessile', 'pendant', etc.).

    Returns:
        ApexResult containing apex coordinates and confidence.
    """
    pts = _ensure_2d_contour(contour)
    is_bottom_apex = mode.lower() in ("pendant", "captive_bubble")

    if is_bottom_apex and axis_direction == (0.0, -1.0):
        axis_direction = (0.0, 1.0)

    dir_vec = np.asarray(axis_direction, dtype=float)
    dir_vec /= np.linalg.norm(dir_vec)

    if axis_origin is not None:
        orig = np.asarray(axis_origin, dtype=float)
    else:
        orig = np.array([float(np.median(pts[:, 0])), float(np.mean(pts[:, 1]))], dtype=float)

    s_proj = np.dot(pts - orig, dir_vec)
    max_s_idx = np.argmax(s_proj)

    n_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=float)
    lateral = np.abs(np.dot(pts - orig, n_vec))

    s_max = s_proj[max_s_idx]
    crest_pts = pts[(s_proj >= s_max - 2.0) & (lateral <= 15.0)]

    if len(crest_pts) >= 3:
        apex = (float(np.mean(crest_pts[:, 0])), float(np.mean(crest_pts[:, 1])))
        conf = 0.95
    else:
        apex = (float(pts[max_s_idx, 0]), float(pts[max_s_idx, 1]))
        conf = 0.85

    return ApexResult(
        point=apex,
        confidence=conf,
        method="symmetry_axis",
        normal=(float(dir_vec[0]), float(dir_vec[1])),
        tangent=(float(n_vec[0]), float(n_vec[1])),
    )


def detect_apex(
    contour: np.ndarray,
    mode: str = "sessile",
    *,
    baseline: tuple[tuple[float, float], tuple[float, float]] | None = None,
    substrate: Any = None,
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    refine: bool = True,
    window_px: float = 15.0,
    band_px: float = 1.0,
    order: int = 3,
) -> ApexResult:
    """Unified master entry point for robust droplet apex detection.

    Automatically dispatches the optimal numerical algorithm based on available
    substrate geometry, tilt, and drop mode:
    1. Curved substrate geometry -> `detect_apex_curved_substrate`.
    2. Tilted or explicit planar baseline -> `detect_apex_normal` (normal projection).
    3. Contact points without explicit baseline -> baseline from contact chord.
    4. Un-tilted / default -> `detect_apex_flat` with multi-point horizontal centroid averaging.
    5. Sub-pixel continuous refinement -> `refine_apex_polynomial` for sub-pixel
       coordinates, apex curvature R0, and physical asymmetry estimation.

    Args:
        contour: (N, 2) array of drop contour points.
        mode: Analysis mode ('sessile', 'pendant', 'captive_bubble', 'capillary_rise').
        baseline: Optional ((x1, y1), (x2, y2)) substrate line.
        substrate: Optional SubstrateProfile or curved substrate object.
        contact_points: Optional ((xL, yL), (xR, yR)) three-phase contact points.
        refine: If True, executes sub-pixel polynomial peak refinement.
        window_px: Neighborhood radius in pixels for sub-pixel refinement.
        band_px: Vertical/normal band width in pixels for discrete crest averaging.
        order: Polynomial refinement order (2 for symmetric parabolic, 3 for asymmetric cubic).

    Returns:
        ApexResult with sub-pixel apex coordinates, confidence, R0, asymmetry, and method.
    """
    pts = _ensure_2d_contour(contour)
    if pts.shape[0] < 3:
        raise ValueError("Contour must contain at least 3 points for apex detection")

    # 1. Curved Substrate
    if substrate is not None and getattr(substrate, "type", "line") != "line":
        sub_res = detect_apex_curved_substrate(pts, substrate, mode=mode)
        if refine:
            baseline_chord = substrate.to_chord() if hasattr(substrate, "to_chord") else baseline
            res = refine_apex_polynomial(pts, sub_res, baseline=baseline_chord, mode=mode, window_px=window_px, order=order)
            return ApexResult(
                point=res.point,
                confidence=res.confidence,
                method=f"{sub_res.method}+{res.method}",
                r0_px=res.r0_px,
                asymmetry_px=res.asymmetry_px,
                curvature_kappa=res.curvature_kappa,
                band_points=res.band_points,
                tangent=sub_res.tangent,
                normal=sub_res.normal,
            )
        return sub_res

    # 2. Baseline or Contact Chord Normal Projection
    effective_baseline = baseline
    if effective_baseline is None and contact_points is not None and len(contact_points) == 2:
        effective_baseline = (
            (float(contact_points[0][0]), float(contact_points[0][1])),
            (float(contact_points[1][0]), float(contact_points[1][1])),
        )

    if effective_baseline is not None:
        norm_res = detect_apex_normal(pts, effective_baseline, mode=mode, band_px=band_px)
        if refine:
            res = refine_apex_polynomial(pts, norm_res, baseline=effective_baseline, mode=mode, window_px=window_px, order=order)
            return ApexResult(
                point=res.point,
                confidence=res.confidence,
                method=f"normal_projection+{res.method}",
                r0_px=res.r0_px,
                asymmetry_px=res.asymmetry_px,
                curvature_kappa=res.curvature_kappa,
                band_points=norm_res.band_points,
                tangent=norm_res.tangent,
                normal=norm_res.normal,
            )
        return norm_res

    # 3. Flat Crest Multi-Point Median (Un-tilted horizontal default)
    flat_res = detect_apex_flat(pts, mode=mode, band_px=band_px)
    if refine:
        res = refine_apex_polynomial(pts, flat_res, mode=mode, window_px=window_px, order=order)
        return ApexResult(
            point=res.point,
            confidence=res.confidence,
            method=f"flat_crest+{res.method}",
            r0_px=res.r0_px,
            asymmetry_px=res.asymmetry_px,
            curvature_kappa=res.curvature_kappa,
            band_points=flat_res.band_points,
            tangent=flat_res.tangent,
            normal=flat_res.normal,
        )

    return flat_res
