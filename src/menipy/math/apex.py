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

# --- Sub-pixel apex refinement tuning ----------------------------------------
# Sag the fit window aims to span. Well clear of the +/-0.5 px quantization of a
# binary-mask contour, yet short enough for a parabola to model a circular
# crest. Swept from 2 to 10 px against caps, needle-cut tops and rotated-ellipse
# crests with known summits: 5 px minimizes the ellipse error and sits within
# 0.03 px of the best cap and cut-top scores.
_SAGITTA_TARGET_PX = 5.0
# Widest and narrowest fit half-widths, in pixels.
_MAX_HALFWIDTH_PX = 60.0
_MIN_HALFWIDTH_PX = 3.0
# A circular crest gives the same chord radius at every depth. Measured on
# pixelized caps (R 20-150 px, tilt -8..10 deg) the estimates spread by at most
# 1.27x; a 4 px needle cut already spreads them by 1.73x and a 12 px cut by 2.3x.
# Past this ratio the summit is cut or kinked and there is no crest to refine.
_CHORD_SPREAD_MAX = 1.7
# Modelled sag must exceed the fit residuals by this factor to be believed.
_SIGNIFICANCE_K = 3.0
# Sideways march of the chord midpoint, per pixel of depth, above which the
# crest is asymmetric enough to fit a cubic. Measured on pixelized caps and on
# rotated-ellipse crests with analytically known summits: symmetric shapes stay
# under 0.31 px/px, ellipses rotated 20-35 deg reach 0.34-1.08.
_ASYMMETRY_DRIFT_MIN = 0.33


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
    maximum perpendicular distance from the substrate baseline line. Its
    position along the substrate is taken as the median midpoint of crest
    chords 2, 3 and 4 bands below that maximum, which is robust to pixel
    quantization and to flat (needle-occluded) tops.

    Args:
        contour: (N, 2) array of ordered contour points.
        baseline: Substrate baseline chord ((x1, y1), (x2, y2)).
        mode: Drop mode ('sessile', 'pendant', 'captive_bubble', etc.).
        band_px: Crest band width in pixels; sets the depth of the crest chords.

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
    u_coords = np.dot(diffs, u)

    # A pixel contour's crest is a horizontal run. Once the baseline is tilted,
    # even slightly, the single vertex of maximum h is an end of that run, not
    # its centre; on a needle-occluded (flat) top it is a corner of the chord.
    # Bisect the crest chord a few bands below the maximum instead.
    crest = _crest_chord_midpoint(pts, h_perps, u_coords, band_px)
    if crest is not None:
        median_u, crest_h, band_points = crest
    else:
        h_max = float(np.max(h_perps))
        exact_mask = np.abs(h_perps - h_max) <= 1e-4
        median_u = float(np.median(u_coords[exact_mask]))
        crest_h = h_max
        band_points = int(np.sum(exact_mask))

    apex_coords = p1 + median_u * u + crest_h * n
    return ApexResult(
        point=(float(apex_coords[0]), float(apex_coords[1])),
        confidence=0.95,
        method="normal_projection",
        band_points=band_points,
        tangent=(float(u[0]), float(u[1])),
        normal=(float(n[0]), float(n[1])),
    )


def _crest_chord_midpoint(
    pts: np.ndarray,
    heights: np.ndarray,
    along: np.ndarray,
    band_px: float,
    depths: tuple[float, ...] = (2.0, 3.0, 4.0),
) -> tuple[float, float, int] | None:
    """Locate the crest along the substrate from chords a few bands below it.

    Walks the ordered contour both ways from the vertex of maximum height until
    the height drops below ``h_max - depth * band_px``, interpolates the two
    crossings and takes their midpoint. For an axisymmetric crest every chord
    parallel to the substrate is bisected by the axis, so the midpoint does not
    depend on where pixel quantization put the maximum vertex.

    Parameters
    ----------
    pts : np.ndarray
        Ordered contour vertices, shape (N, 2).
    heights : np.ndarray
        Height of each vertex along the apex-side substrate normal.
    along : np.ndarray
        Coordinate of each vertex along the substrate.
    band_px : float
        Crest band width in pixels; chord depths are multiples of it.
    depths : tuple of float, optional
        Chord depths below the maximum, in units of ``band_px``.

    Returns
    -------
    tuple of (float, float, int) or None
        Crest coordinate along the substrate, contour height there and number
        of vertices in the crest run, or ``None`` when a chord does not close
        on both sides (open contour ending at its crest).
    """
    n_pts = pts.shape[0]
    if n_pts < 3:
        return None
    closed = float(np.linalg.norm(pts[0] - pts[-1])) <= 2.0
    top = int(np.argmax(heights))
    h_max = float(heights[top])
    band = max(0.25, float(band_px))

    midpoints = []
    run: set[int] = {top}
    for depth in depths:
        level = h_max - depth * band
        ends = []
        for step in (1, -1):
            i = top
            crossing = None
            for _ in range(n_pts - 1):
                j = i + step
                if not closed and not 0 <= j < n_pts:
                    break
                j %= n_pts
                if heights[j] < level:
                    t = (heights[i] - level) / (heights[i] - heights[j])
                    crossing = along[i] + t * (along[j] - along[i])
                    break
                run.add(j)
                i = j
            if crossing is None:
                return None
            ends.append(crossing)
        midpoints.append(0.5 * (ends[0] + ends[1]))

    crest_u = float(np.median(midpoints))
    run_idx = np.fromiter(run, dtype=int)
    order = np.argsort(along[run_idx])
    crest_h = float(
        np.interp(crest_u, along[run_idx][order], heights[run_idx][order])
    )
    return crest_u, crest_h, int(run_idx.size)


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


def _walk_branch(
    xi: np.ndarray,
    eta: np.ndarray,
    start: int,
    closed: bool,
    halfwidth: float,
    depth_limit: float,
) -> np.ndarray:
    """Collect the contiguous contour run around ``start`` inside a local window.

    Walking the contour, instead of masking every vertex by distance, keeps the
    fit on the crest branch: an unrelated part of the outline that happens to
    pass close to the apex -- the substrate edge of a small drop, the opposite
    wall of a narrow neck -- is never picked up.

    Parameters
    ----------
    xi : np.ndarray
        Coordinate of each vertex along the substrate tangent, apex-relative.
    eta : np.ndarray
        Coordinate of each vertex along the apex-side normal, apex-relative.
    start : int
        Index of the vertex closest to the initial apex estimate.
    closed : bool
        Whether the contour wraps around from its last vertex to its first.
    halfwidth : float
        Maximum ``|xi|`` of a vertex in the window.
    depth_limit : float
        Minimum ``eta`` of a vertex in the window (a negative depth below the
        crest); stops the walk from running down a steep flank.

    Returns
    -------
    np.ndarray
        Indices of the contiguous run, unordered.
    """
    n_pts = xi.size
    idx = [start]
    for step in (1, -1):
        i = start
        for _ in range(n_pts - 1):
            j = i + step
            if not closed and not 0 <= j < n_pts:
                break
            j %= n_pts
            if j == start:
                break
            if abs(xi[j]) > halfwidth or eta[j] < depth_limit:
                break
            idx.append(j)
            i = j
    return np.asarray(idx, dtype=int)


@dataclass(frozen=True)
class CrestChords:
    """Crest geometry read off contour chords, independently of any fit window.

    Attributes
    ----------
    radius : float
        Median per-depth estimate of the apex radius of curvature, in pixels.
    radius_spread : float
        Ratio of the largest to the smallest per-depth radius estimate. One for
        a circular crest; well above one for a cut or kinked top.
    tight_radius : float
        Same radius measured from the shorter flank alone, which is the one a
        fit window must not outgrow on an asymmetric crest.
    axis_drift : float
        Sideways march of the chord midpoint per pixel of depth. Near zero for
        a symmetric crest, where every chord is bisected by the same axis.
    """

    radius: float
    radius_spread: float
    tight_radius: float
    axis_drift: float


def _chord_crossings(
    xi: np.ndarray,
    eta: np.ndarray,
    start: int,
    closed: bool,
    depth: float,
) -> tuple[float, float] | None:
    """Locate where the contour crosses a chord ``depth`` pixels below the crest.

    Parameters
    ----------
    xi : np.ndarray
        Apex-relative tangential coordinate of every vertex.
    eta : np.ndarray
        Apex-relative normal coordinate of every vertex.
    start : int
        Index of the vertex closest to the initial apex estimate.
    closed : bool
        Whether the contour wraps around.
    depth : float
        Depth of the chord below the crest, in pixels.

    Returns
    -------
    tuple of float or None
        The two crossing coordinates along the tangent, or ``None`` when the
        chord does not close on both sides.
    """
    n_pts = xi.size
    ends: list[float] = []
    for step in (1, -1):
        i = start
        crossing = None
        for _ in range(n_pts - 1):
            j = i + step
            if not closed and not 0 <= j < n_pts:
                break
            j %= n_pts
            if j == start:
                break
            if eta[j] < -depth:
                denom = eta[i] - eta[j]
                t = (eta[i] + depth) / denom if abs(denom) > 1e-12 else 0.0
                crossing = xi[i] + t * (xi[j] - xi[i])
                break
            i = j
        if crossing is None:
            return None
        ends.append(float(crossing))
    return ends[0], ends[1]


def _measure_crest_chords(
    xi: np.ndarray,
    eta: np.ndarray,
    start: int,
    closed: bool,
    depths: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0),
) -> CrestChords | None:
    """Measure the crest from chords at several depths below the summit.

    For a circular crest a chord ``d`` below the summit has half-width ``c``
    obeying ``c**2 = 2 R d - d**2``, hence ``R = (c**2 + d**2) / (2 d)``, and
    every chord is bisected by the axis. Chords are read off the contour by
    interpolation, so none of this needs a fit window -- which is what makes it
    usable to *choose* one, and to judge what the window will be able to fit.

    Two departures from that ideal carry information. A cut (needle-occluded)
    top reads far wider just under the chord than below it, so its per-depth
    radii fan out instead of agreeing. And on an asymmetric crest the chord
    midpoints march sideways with depth rather than staying on one axis.

    Parameters
    ----------
    xi : np.ndarray
        Apex-relative tangential coordinate of every vertex.
    eta : np.ndarray
        Apex-relative normal coordinate of every vertex.
    start : int
        Index of the vertex closest to the initial apex estimate.
    closed : bool
        Whether the contour wraps around.
    depths : tuple of float, optional
        Chord depths below the crest, in pixels.

    Returns
    -------
    CrestChords or None
        Crest measurements, or ``None`` when no chord closes on both sides.
    """
    radii: list[float] = []
    tight: list[float] = []
    used: list[float] = []
    midpoints: list[float] = []
    for depth in depths:
        ends = _chord_crossings(xi, eta, start, closed, depth)
        if ends is None:
            continue
        half_width = 0.5 * abs(ends[0] - ends[1])
        if half_width <= 1e-6:
            continue
        radii.append((half_width**2 + depth**2) / (2.0 * depth))
        # The same radius from the shorter flank alone. On an asymmetric crest
        # the flat side runs far out and would size the fit window for a
        # curvature the steep side does not have.
        flank = max(min(abs(ends[0]), abs(ends[1])), 1e-6)
        tight.append((flank**2 + depth**2) / (2.0 * depth))
        midpoints.append(0.5 * (ends[0] + ends[1]))
        used.append(depth)

    if not radii:
        return None

    drift = float(np.polyfit(used, midpoints, 1)[0]) if len(used) >= 3 else 0.0

    return CrestChords(
        radius=float(np.median(radii)),
        radius_spread=float(max(radii) / min(radii)),
        tight_radius=float(np.median(tight)),
        axis_drift=drift,
    )


def _stationary_point(coeffs: np.ndarray, order: int) -> float | None:
    """Locate the crest of a fitted polynomial along the tangent.

    Parameters
    ----------
    coeffs : np.ndarray
        Polynomial coefficients in :func:`numpy.polyfit` order.
    order : int
        Order of the fit, 2 or 3.

    Returns
    -------
    float or None
        Tangential coordinate of the maximum, or ``None`` when the polynomial
        has no concave stationary point.
    """
    if order == 3:
        d_cub, a, b = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    else:
        d_cub, a, b = 0.0, float(coeffs[0]), float(coeffs[1])

    if abs(d_cub) > 1e-7:
        disc = 4.0 * (a**2) - 12.0 * d_cub * b
        if disc >= 0:
            roots = (
                (-2.0 * a + np.sqrt(disc)) / (6.0 * d_cub),
                (-2.0 * a - np.sqrt(disc)) / (6.0 * d_cub),
            )
            maxima = [r for r in roots if (6.0 * d_cub * r + 2.0 * a) < 0]
            if maxima:
                return float(min(maxima, key=abs))

    if a >= -1e-8:
        return None
    return float(-b / (2.0 * a))


def _window_for_radius(r0: float, sagitta_px: float, cap: float) -> float:
    """Choose a fit half-width that resolves ``sagitta_px`` of sag at radius ``r0``.

    A circle of radius ``r0`` sags by ``w**2 / (2 r0)`` over a half-width ``w``.
    Solving for the half-width whose sag clears the +/-0.5 px quantization of a
    binary-mask contour by a few pixels gives ``w = sqrt(2 r0 s)``. The window is
    additionally kept below ``r0 / 2``, where the quartic term a circle carries
    beyond its osculating parabola still amounts to only ~6% of the sag.

    Parameters
    ----------
    r0 : float
        Apex radius of curvature estimate, in pixels.
    sagitta_px : float
        Target sag over the window, in pixels.
    cap : float
        Hard upper bound on the half-width.

    Returns
    -------
    float
        Fit half-width in pixels.
    """
    w = float(np.sqrt(2.0 * max(r0, 1e-6) * sagitta_px))
    w = min(w, 0.5 * r0, cap)
    return float(max(w, _MIN_HALFWIDTH_PX))


def refine_apex_polynomial(
    contour: np.ndarray,
    initial_apex: tuple[float, float] | ApexResult,
    baseline: tuple[tuple[float, float], tuple[float, float]] | None = None,
    mode: str = "sessile",
    window_px: float | None = None,
    order: int = 3,
    *,
    sagitta_px: float = _SAGITTA_TARGET_PX,
) -> ApexResult:
    """Refine a droplet apex to sub-pixel coordinates with a curvature-adapted fit.

    Contour points around ``initial_apex`` are expressed in the substrate-aligned
    frame (tangential ``xi``, normal ``eta``) and modelled by
    ``eta(xi) = a xi**2 + b xi + c (+ d xi**3)``; the continuous crest is the
    stationary point ``xi*`` of that polynomial, and the apex curvature radius
    follows from the quadratic term.

    The fit window is *not* fixed. Crest chords first give the apex radius ``R0``
    (:func:`_measure_crest_chords`), and the half-width is set to ``w = sqrt(2 R0 s)``
    with ``s`` a few pixels of sag, capped at ``R0 / 2``
    (:func:`_window_for_radius`); the fit then iterates so that ``w`` tracks the
    radius it implies. A narrower window leaves the sag under pixel quantization
    and makes ``xi* = -b / 2a`` ill-conditioned on large drops; a wider one lets
    the parabola overshoot the crown of a small one.

    Refinement is abandoned -- the crest estimate is returned unchanged, with a
    ``*_fallback`` method name -- whenever the window provably cannot resolve a
    crest: a flat (needle-cut) top, a fit whose modelled sag does not stand out
    of its own residuals, or a stationary point outside the fitted region.

    Parameters
    ----------
    contour : np.ndarray
        Contour points, shape (N, 2).
    initial_apex : tuple of float or ApexResult
        Crest estimate to refine.
    baseline : tuple of tuple of float, optional
        Substrate chord ``((x1, y1), (x2, y2))`` aligning the tangent and normal.
    mode : str, optional
        Drop mode ('sessile', 'pendant', 'captive_bubble', 'capillary_rise').
    window_px : float, optional
        Upper bound on the fit half-width, in pixels. ``None`` (the default)
        lets the curvature alone set it.
    order : int, optional
        Highest polynomial order allowed. A cubic is fitted only on a crest the
        chords measured as asymmetric, so a lopsided summit is resolved without
        letting pixel noise bend a symmetric one.
    sagitta_px : float, optional
        Sag the window aims to span, in pixels.

    Returns
    -------
    ApexResult
        Refined sub-pixel apex with ``r0_px``, ``asymmetry_px`` and confidence,
        or the unrefined crest when the crest cannot be resolved.
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

    tangent = (float(u[0]), float(u[1]))
    normal = (float(n[0]), float(n[1]))

    def _fallback(method: str, points: int, r0: float | None = None) -> ApexResult:
        """Return the unrefined crest estimate under a diagnostic method name."""
        return ApexResult(
            point=(float(p_init[0]), float(p_init[1])),
            confidence=0.6 if points >= 5 else 0.5,
            method=method,
            r0_px=r0,
            asymmetry_px=0.0,
            curvature_kappa=(1.0 / r0) if r0 else None,
            band_points=int(points),
            tangent=tangent,
            normal=normal,
        )

    if pts.shape[0] < 5:
        return _fallback("initial_fallback", pts.shape[0])

    # Apex-relative local frame for the whole contour.
    diffs_all = pts - p_init
    xi_all = np.dot(diffs_all, u)
    eta_all = np.dot(diffs_all, n)
    closed = bool(pts.shape[0] > 3 and float(np.linalg.norm(pts[0] - pts[-1])) <= 2.0)
    start = int(np.argmin(np.linalg.norm(diffs_all, axis=1)))

    w_cap = _MAX_HALFWIDTH_PX
    if window_px is not None and float(window_px) > 0.0:
        w_cap = min(w_cap, max(float(window_px), _MIN_HALFWIDTH_PX))

    chords = _measure_crest_chords(xi_all, eta_all, start, closed)
    r0_chord = chords.radius if chords is not None else None

    # A needle-occluded (or otherwise cut) top is not a crest: its chords fan
    # out with depth instead of tracing one radius. Nothing there can be
    # refined, so keep the measured chord midpoint -- and report no radius,
    # since the chords of a cut top measure the cut, not the hidden summit.
    if chords is not None and chords.radius_spread > _CHORD_SPREAD_MAX:
        return _fallback("cut_crest_fallback", pts.shape[0])

    # Only a crest whose chord midpoints visibly march sideways is asymmetric
    # enough for a cubic term to describe rather than to chase pixel noise.
    allow_cubic = (
        order >= 3
        and chords is not None
        and abs(chords.axis_drift) >= _ASYMMETRY_DRIFT_MIN
    )

    w = (
        _window_for_radius(chords.tight_radius, sagitta_px, w_cap)
        if chords is not None
        else min(15.0, w_cap)
    )

    coeffs: np.ndarray | None = None
    fit_order = 2
    local_idx = np.empty(0, dtype=int)
    w_fit = w
    rms_resid = 0.0
    r2 = 0.0

    for _ in range(3):
        # Widen until the window holds enough vertices to fit and test a parabola.
        while True:
            local_idx = _walk_branch(
                xi_all, eta_all, start, closed, w, -max(2.0 * sagitta_px, 2.0)
            )
            if local_idx.size >= 6 or w >= w_cap - 1e-9:
                break
            w = min(w * 1.6, w_cap)
        w_fit = w
        if local_idx.size < 5:
            return _fallback("initial_fallback", local_idx.size, r0_chord)

        xi = xi_all[local_idx]
        eta = eta_all[local_idx]
        try:
            quad = np.polyfit(xi, eta, 2)
        except (np.linalg.LinAlgError, ValueError):
            return _fallback("fit_error_fallback", local_idx.size, r0_chord)

        coeffs, fit_order = quad, 2
        # Prefer the parabola. A cubic is taken only on a crest whose chords
        # already showed it to be asymmetric. Judging the cubic by its own
        # residuals instead does not work: the staircase of a pixel contour is
        # structured, not independent, noise, and a cubic fits that staircase
        # well enough to pass an F-test on a perfectly symmetric drop -- which
        # then moves the apex by several pixels.
        if allow_cubic and local_idx.size >= 9:
            try:
                cubic = np.polyfit(xi, eta, 3)
            except (np.linalg.LinAlgError, ValueError):
                cubic = None
            if cubic is not None and _stationary_point(cubic, 3) is not None:
                coeffs, fit_order = cubic, 3

        a = float(coeffs[-3])
        if a >= -1e-8:
            # No concave crest in the window (flat or needle-cut top): there is
            # nothing to refine, and ``argmax(eta)`` would only return whichever
            # tied vertex comes first in contour order.
            return _fallback("no_peak_fallback", local_idx.size, r0_chord)

        resid = eta - np.polyval(coeffs, xi)
        rms_resid = float(np.sqrt(np.mean(resid**2)))
        ss_tot = float(np.sum((eta - np.mean(eta)) ** 2))
        r2 = float(max(0.0, 1.0 - float(np.sum(resid**2)) / ss_tot)) if ss_tot > 1e-12 else 0.8

        # Re-centre the window on the radius the fit itself implies.
        w_next = _window_for_radius(1.0 / (2.0 * abs(a)), sagitta_px, w_cap)
        if abs(w_next - w) <= 0.1 * w:
            break
        w = w_next

    if coeffs is None:
        return _fallback("initial_fallback", local_idx.size, r0_chord)

    if fit_order == 3:
        d_cub, a = float(coeffs[0]), float(coeffs[1])
    else:
        d_cub, a = 0.0, float(coeffs[0])

    # A crest the window cannot resolve: the modelled sag does not stand out of
    # the residuals, so -b / 2a is driven by pixel noise rather than by shape.
    if abs(a) * w_fit * w_fit < _SIGNIFICANCE_K * rms_resid:
        return _fallback("unresolved_curvature_fallback", local_idx.size, r0_chord)

    xi_star = _stationary_point(coeffs, fit_order)
    if xi_star is None:
        return _fallback("no_peak_fallback", local_idx.size, r0_chord)

    # The stationary point must sit well inside the fitted region: outside it the
    # polynomial only extrapolates, and clamping it back would invent a crest
    # several pixels away from the one that was actually measured.
    if not np.isfinite(xi_star) or abs(xi_star) > 0.5 * w_fit:
        return _fallback("vertex_outside_window_fallback", local_idx.size, r0_chord)

    eta_star = float(np.polyval(coeffs, xi_star))
    if abs(eta_star) > max(1.0, sagitta_px):
        return _fallback("vertex_off_crest_fallback", local_idx.size, r0_chord)

    p_refined = p_init + xi_star * u + eta_star * n

    # Curvature at the stationary point: 2a for a parabola, 6 d xi* + 2a cubic.
    kappa_0 = abs(6.0 * d_cub * xi_star + 2.0 * a)
    r0_px = (1.0 / kappa_0) if kappa_0 > 1e-9 else None

    confidence = float(np.clip(0.5 + 0.49 * r2, 0.5, 0.99))

    return ApexResult(
        point=(float(p_refined[0]), float(p_refined[1])),
        confidence=confidence,
        method=f"subpixel_poly{fit_order}",
        r0_px=float(r0_px) if r0_px is not None else None,
        asymmetry_px=float(xi_star),
        curvature_kappa=float(kappa_0),
        band_points=int(local_idx.size),
        tangent=tangent,
        normal=normal,
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
    window_px: float | None = None,
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
       coordinates, apex curvature R0, and physical asymmetry estimation. Its
       fit window follows the measured curvature, and it returns the crest
       estimate untouched when the crest cannot be resolved.

    Args:
        contour: (N, 2) array of drop contour points.
        mode: Analysis mode ('sessile', 'pendant', 'captive_bubble', 'capillary_rise').
        baseline: Optional ((x1, y1), (x2, y2)) substrate line.
        substrate: Optional SubstrateProfile or curved substrate object.
        contact_points: Optional ((xL, yL), (xR, yR)) three-phase contact points.
        refine: If True, executes sub-pixel polynomial peak refinement.
        window_px: Upper bound on the sub-pixel fit half-width in pixels; None
            (default) lets the apex curvature alone set it.
        band_px: Vertical/normal band width in pixels for discrete crest averaging.
        order: Highest polynomial refinement order (2 parabolic; 3 allows an
            asymmetric cubic on crests measured to be asymmetric).

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
