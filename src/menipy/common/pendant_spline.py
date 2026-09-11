"""Two-zone clothoid spline model of a pendant drop interface.

A pendant profile has two zones with different geometry and different roles:

* zone 1, apex -> equator (maximum diameter): convex, curvature falling from
  the apex; it fixes the drop's scale;
* zone 2, equator -> needle contact (``P1``, ``P2``): the neck, where the
  meridional curvature can change sign; it carries most of the information on
  the Bond number, hence on the surface tension.

Each side is a chain of G1 Hermite clothoids (see
:mod:`menipy.common.clothoid_spline`, whose residuals and analytic Jacobian are
reused) from the apex -- horizontal tangent -- over the equator -- a node whose
tangent is fixed vertical, height free -- to the contact on the needle, whose
heading is fitted: it is the angle at the needle. Fitting happens in the frame
of the drop's symmetry axis, estimated from the midpoints of the two sides, so a
tilted camera or needle does not bias the zones.

The physics prior is the anchored Young-Laplace profile of
:mod:`menipy.math.pendant_box` (apex height and equatorial radius anchor it, a
golden-section search picks the Bond number). It gives

* the surface tension ``Δρ g b² / Bo``;
* the clothoid count of each zone, from ``∫ (|κ''| / (384 tol))^(1/4) ds``, and
  the joins at equal quantiles of ``∫ |κ''|^(1/4) ds``;
* the discretization bias of the needle angle, measured by fitting the same
  layout to the noise-free matched profile.

The fitted spline is a denoised contour with few parameters. Its curvature
also gives an independent surface tension: by the Young-Laplace equation the
mean curvature ``κ_meridional + sin φ / r`` falls linearly with height at the
rate ``Δρ g / γ``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from menipy.common.arc_spline import ArcSplineLevel, _Edge, _Frame, estimate_edge_noise
from menipy.common.clothoid_spline import ClothoidSegment, _bounds, _Problem
from menipy.math.pendant_box import PendantShape, fit_pendant_box

BiasMode = Literal["model", "off"]
_HALF_PI = 0.5 * np.pi


# -----------------------------------------------------------------------------
# Layout: per side, zone-1 joins, the equator (vertical tangent), zone-2 joins
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZoneLayout:
    """Parameter layout of the two-zone chains.

    ``[t_apex, h_apex, side-1 nodes, side-2 nodes, theta_P1, theta_P2]`` and,
    with ``slide``, the sideways moves of the two contacts; the nodes of a side
    run from the apex outwards: ``z1 - 1`` zone-1 joins ``(t, h, theta)``, the
    equator ``(t, h)`` and ``z2 - 1`` zone-2 joins.
    """

    zones1: tuple[int, int]
    zones2: tuple[int, int]
    slide: bool = False

    def zones(self, side: int) -> tuple[int, int]:
        return self.zones1 if side == 1 else self.zones2

    def n_nodes(self, side: int) -> int:
        z1, z2 = self.zones(side)
        return z1 + z2 - 1

    @property
    def n1(self) -> int:
        return self.n_nodes(1)

    @property
    def n2(self) -> int:
        return self.n_nodes(2)

    def _count(self, side: int) -> int:
        return 3 * (self.n_nodes(side) - 1) + 2

    def block(self, side: int) -> int:
        return 2 if side == 1 else 2 + self._count(1)

    @property
    def size(self) -> int:
        return 2 + self._count(1) + self._count(2) + (4 if self.slide else 2)

    def contact_index(self, side: int) -> int:
        return 2 + self._count(1) + self._count(2) + (side - 1)

    def slide_index(self, side: int) -> int:
        return self.contact_index(2) + side

    def equator(self, side: int) -> int:
        return self.zones(side)[0] - 1

    def t_index(self, side: int) -> np.ndarray:
        k = np.arange(self.n_nodes(side))
        return self.block(side) + 3 * k - (k > self.equator(side))

    def h_index(self, side: int) -> np.ndarray:
        return self.t_index(side) + 1

    def theta_index(self, side: int) -> np.ndarray:
        k = np.arange(self.n_nodes(side))
        return np.where(k == self.equator(side), -1, self.t_index(side) + 2)

    def fixed_theta(self, side: int) -> np.ndarray:
        k = np.arange(self.n_nodes(side))
        return np.where(k == self.equator(side), _HALF_PI, np.nan)


def _pack(lay: _ZoneLayout, ta: float, ha: float, sides, c1: float, c2: float,
          slides: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    p = np.zeros(lay.size)
    p[0], p[1] = ta, ha
    for side, (t, h, th) in zip((1, 2), sides):
        p[lay.t_index(side)] = t
        p[lay.h_index(side)] = h
        ith = lay.theta_index(side)
        free = ith >= 0
        p[ith[free]] = np.asarray(th)[free]
    p[lay.contact_index(1)] = c1
    p[lay.contact_index(2)] = c2
    if lay.slide:
        p[lay.slide_index(1)], p[lay.slide_index(2)] = slides
    return p


def _slides(lay: _ZoneLayout, p: np.ndarray) -> tuple[float, float]:
    return (float(p[lay.slide_index(1)]), float(p[lay.slide_index(2)])) if lay.slide else (0.0, 0.0)


class _ZoneProblem(_Problem):
    """Clothoid-chain problem whose contacts may move sideways (perpendicular to the axis).

    A detected contact point is only as good as the needle-wall location; a
    fraction of a pixel sideways, forced through, bends the last clothoid and
    moves the needle angle by degrees. The contact height stays fixed.
    """

    def _side_nodes(self, p: np.ndarray, side: int):
        pos, d_pos, heading, d_head = super()._side_nodes(p, side)
        lay = self.layout
        if lay.slide:
            i = lay.slide_index(side)
            pos[-1, 0] += p[i]
            d_pos[-1, 0, i] = 1.0
        return pos, d_pos, heading, d_head


def _unpack(lay: _ZoneLayout, p: np.ndarray, side: int):
    ith = lay.theta_index(side)
    th = np.where(ith >= 0, p[np.maximum(ith, 0)], lay.fixed_theta(side))
    return p[lay.t_index(side)].copy(), p[lay.h_index(side)].copy(), th


# -----------------------------------------------------------------------------
# Interface, frame and anchors
# -----------------------------------------------------------------------------


def extract_pendant_interface(contour: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return the ordered drop edge from ``P1`` around the apex to ``P2``.

    Points on the needle side of the contact chord (the needle shaft) are
    dropped; the largest gap left in the cyclic contour order is where the
    needle was, so the remaining run is rotated to start and end there. This
    also accepts contours that were already clipped at the contacts, whatever
    their starting vertex.

    Parameters
    ----------
    contour : np.ndarray
        Ordered drop silhouette, shape ``(N, 2)``.
    p1, p2 : np.ndarray
        Contact points on the needle.

    Returns
    -------
    np.ndarray
        Edge points ordered from ``P1`` to ``P2``, the contacts included.

    Raises
    ------
    ValueError
        If fewer than 10 drop points remain.
    """
    pts = np.asarray(contour, float).reshape(-1, 2)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    frame = _Frame.from_contacts(p1, p2, pts)
    local = frame.to_local(pts)
    width = float(np.hypot(*(p2 - p1)))
    keep = local[:, 1] < 0.5
    keep &= (np.hypot(*(pts - p1).T) > 0.5) & (np.hypot(*(pts - p2).T) > 0.5)
    run = pts[keep]
    if len(run) < 10:
        raise ValueError("too few drop points below the needle")
    gaps = np.hypot(*(np.roll(run, -1, axis=0) - run).T)
    start = (int(np.argmax(gaps)) + 1) % len(run)
    run = np.roll(run, -start, axis=0)
    if np.hypot(*(run[0] - p1)) > np.hypot(*(run[0] - p2)):
        run = run[::-1]
    # trim stray vertices beyond the contacts (e.g. along the needle tip)
    lr = frame.to_local(run)
    inside = (lr[:, 0] > -0.5 * width) & (lr[:, 0] < 1.5 * width) | (lr[:, 1] < -1.0)
    run = run[inside]
    return np.vstack([p1, run, p2])


def _axis_frame(p1: np.ndarray, p2: np.ndarray, up: np.ndarray) -> _Frame:
    """Frame with ``P1`` at the origin and ``+y`` along ``up`` (apex -> needle)."""
    ey = np.asarray(up, float) / (np.hypot(*up) or 1.0)
    ex = np.array([ey[1], -ey[0]])
    if (np.asarray(p2, float) - p1) @ ex < 0:
        ex = -ex
    return _Frame(np.asarray(p1, float), ex, ey)


def _symmetry_slope(q: np.ndarray, max_points: int = 240) -> float | None:
    """``dx/dy`` of the mirror-symmetry axis of the edge, in its current frame.

    One side is reflected across a candidate axis (angle and offset) and its
    clipped distance to the other side minimized. Midpoints of horizontal
    chords would not do: near the apex, where the sides are flat, the chord
    midpoints of a tilted drop stay on a vertical line whatever the tilt.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    i_low = int(np.argmin(q[:, 1]))
    left, right = q[: i_low + 1], q[i_low:]
    if len(left) < 10 or len(right) < 10:
        return None
    # dense copy of the left side so nearest-vertex distances are sub-pixel
    steps = np.hypot(*np.diff(left, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    grid = np.arange(0.0, s[-1], 0.25)
    tree = cKDTree(np.column_stack([np.interp(grid, s, left[:, 0]), np.interp(grid, s, left[:, 1])]))
    if len(right) > max_points:
        right = right[np.linspace(0, len(right) - 1, max_points).round().astype(int)]
    y0 = float(q[i_low, 1])
    c0 = 0.5 * (float(np.min(q[:, 0])) + float(np.max(q[:, 0])))

    def cost(v: np.ndarray) -> float:
        a, c = v
        normal = np.array([np.cos(a), -np.sin(a)])  # axis direction (sin a, cos a)
        d = (right - np.array([c, y0])) @ normal
        mirrored = right - 2.0 * d[:, None] * normal
        return float(np.mean(np.minimum(tree.query(mirrored)[0], 3.0) ** 2))

    res = minimize(cost, np.array([0.0, c0]), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-9, "initial_simplex": [[0.0, c0], [0.02, c0], [0.0, c0 + 2.0]]})
    a = float(res.x[0])
    return float(np.tan(a)) if abs(a) < 0.5 else None


def _vertex(u: np.ndarray, v: np.ndarray, half: float) -> tuple[float, float] | None:
    """Stationary point of a cubic ``v(u)`` nearest ``u = 0`` within ``half``."""
    if len(u) < 6:
        return None
    c3, c2, c1, c0 = np.polyfit(u, v, 3)
    roots = np.roots([3.0 * c3, 2.0 * c2, c1]) if abs(c3) > 1e-15 else np.array([-c1 / (2.0 * c2)])
    roots = roots[np.isreal(roots)].real
    roots = roots[np.abs(roots) <= half]
    if roots.size == 0:
        return None
    r = float(roots[np.argmin(np.abs(roots))])
    return r, float(np.polyval([c3, c2, c1, c0], r))


def _anchors(q: np.ndarray) -> dict[str, float] | None:
    """Well-conditioned extremum values: equatorial radius, axis and apex height.

    The widest point of each side is refined by a cubic ``x(y)`` over a window
    of about a third of the radius (the value is sharp, its height is not); the
    axis is midway between the two. The apex height is the vertex value of a
    cubic ``y(x)`` over the bottom third, centred on the axis.
    """
    i_low = int(np.argmin(q[:, 1]))
    left, right = q[: i_low + 1], q[i_low:]
    if len(left) < 10 or len(right) < 10:
        return None
    r_guess = 0.5 * float(np.max(q[:, 0]) - np.min(q[:, 0]))
    half = 0.35 * r_guess
    extrema = []
    for branch, sign in ((left, -1.0), (right, 1.0)):
        i = int(np.argmax(sign * branch[:, 0]))
        y0, val = float(branch[i, 1]), float(sign * branch[i, 0])
        for _ in range(3):
            near = np.abs(branch[:, 1] - y0) <= half
            hit = _vertex(branch[near, 1] - y0, sign * branch[near, 0], half)
            if hit is None:
                break
            y0, val = y0 + hit[0], hit[1]
        extrema.append((sign * val, y0))
    (x_left, y_left), (x_right, y_right) = extrema
    axis = 0.5 * (x_left + x_right)
    r_eq = 0.5 * (x_right - x_left)
    if r_eq <= 2.0:
        return None
    xa, ya = axis, float(q[i_low, 1])
    half_a = 0.3 * r_eq
    for _ in range(3):
        near = (np.abs(q[:, 0] - xa) <= half_a) & (q[:, 1] <= ya + half_a)
        hit = _vertex(q[near, 0] - xa, q[near, 1], half_a)
        if hit is None:
            break
        xa, ya = xa + hit[0], hit[1]
    bottom = np.abs(q[:, 0] - axis) <= half_a
    i_apex = int(np.argmin(np.where(bottom, np.hypot(q[:, 0] - axis, q[:, 1] - ya), np.inf)))
    return {"axis": float(axis), "r_eq": float(r_eq), "apex_y": float(ya), "i_apex": i_apex,
            "equator_y": (float(y_left), float(y_right))}


# -----------------------------------------------------------------------------
# Fit result
# -----------------------------------------------------------------------------


@dataclass
class PendantSplineFit:
    """Fitted two-zone clothoid spline of a pendant drop.

    Attributes
    ----------
    p1_segments, p2_segments : list of ClothoidSegment
        Chains from the apex to ``P1`` and to ``P2``, in the fitting frame.
    zones_p1, zones_p2 : tuple of int
        Clothoids in zone 1 and zone 2 of each side.
    needle_angle_p1_deg, needle_angle_p2_deg : float
        Tangent angle at the needle contact, from the horizontal (the
        Young-Laplace ``φ``: 90° is vertical, above 90° the neck turns back
        towards the axis); raw plus bias correction.
    needle_angle_p1_raw_deg, needle_angle_p2_raw_deg : float
        Fitted end headings as needle angles.
    correction_p1_deg, correction_p2_deg : float
        Model-based discretization correction.
    sigma_p1_deg, sigma_p2_deg : float
        Standard deviation of the raw needle angles from the fit covariance.
    shape : PendantShape
        Anchored Young-Laplace profile (Bond number, apex radius).
    surface_tension_mN_m : float
        ``Δρ g b² / Bo`` from ``shape``; NaN without scale or densities.
    laplace : dict
        Linear fit of the spline's mean curvature against height, and the
        surface tension it implies.
    axis_tilt_deg : float
        Symmetry-axis tilt from the contact-chord normal.
    contact_slide_px : tuple of float
        Sideways move of each contact (perpendicular to the axis) found by the fit.
    rmse_px : float
        Residual RMS of the edge points.
    edge_noise_px : float
        Model-free edge-noise estimate.
    n_points : int
        Edge points fitted.
    levels : list of ArcSplineLevel
        Fitted levels (segments per side, needle angles, residual, BIC).
    init : str
        ``"physics"`` or ``"physics+refined"``.
    rejection_reasons : list of str
        Quality-gate failures; empty when usable.
    """

    p1_segments: list[ClothoidSegment]
    p2_segments: list[ClothoidSegment]
    zones_p1: tuple[int, int]
    zones_p2: tuple[int, int]
    needle_angle_p1_deg: float
    needle_angle_p2_deg: float
    needle_angle_p1_raw_deg: float
    needle_angle_p2_raw_deg: float
    correction_p1_deg: float
    correction_p2_deg: float
    sigma_p1_deg: float
    sigma_p2_deg: float
    shape: PendantShape
    surface_tension_mN_m: float
    laplace: dict[str, float]
    axis_tilt_deg: float
    contact_slide_px: tuple[float, float]
    apex_xy: tuple[float, float]
    rmse_px: float
    edge_noise_px: float
    n_points: int
    levels: list[ArcSplineLevel]
    init: str
    rejection_reasons: list[str]
    _frame: _Frame = field(repr=False)
    _anchors: dict[str, Any] = field(repr=False)

    @property
    def n_segments(self) -> int:
        """Total number of clothoids."""
        return len(self.p1_segments) + len(self.p2_segments)

    @property
    def accepted(self) -> bool:
        """Whether the fit passed every quality gate."""
        return not self.rejection_reasons

    @property
    def needle_angle_deg(self) -> float:
        """Mean needle angle of the two sides."""
        return 0.5 * (self.needle_angle_p1_deg + self.needle_angle_p2_deg)

    def _local_samples(self, step_px: float) -> np.ndarray:
        parts = []
        for seg in reversed(self.p1_segments):
            s = np.linspace(seg.length, 0.0, max(2, int(np.ceil(seg.length / step_px)) + 1))
            parts.append(seg.points(s)[:-1])
        for i, seg in enumerate(self.p2_segments):
            s = np.linspace(0.0, seg.length, max(2, int(np.ceil(seg.length / step_px)) + 1))
            pts = seg.points(s)
            parts.append(pts if i == len(self.p2_segments) - 1 else pts[:-1])
        return np.vstack(parts)

    def sample(self, step_px: float = 1.0) -> np.ndarray:
        """Sample the spline from ``P1`` over the apex to ``P2`` (image coordinates).

        Parameters
        ----------
        step_px : float, optional
            Arc-length spacing of the samples.

        Returns
        -------
        np.ndarray
            Points, shape ``(M, 2)``.
        """
        return self._frame.to_image(self._local_samples(step_px))

    @property
    def contact_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the spline meets the needle (after any contact slide), image coordinates."""
        ends = np.array([self.p1_segments[-1].points(np.array([self.p1_segments[-1].length]))[0],
                         self.p2_segments[-1].points(np.array([self.p2_segments[-1].length]))[0]])
        img = self._frame.to_image(ends)
        return img[0], img[1]

    @property
    def axis_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Symmetry axis as (point at the apex height, unit direction apex -> needle), image coordinates."""
        a = self._anchors
        point = self._frame.to_image(np.array([a["axis"], a["apex_y"]]))[0]
        return point, self._frame.ey.copy()

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for results and exports."""
        sh = self.shape
        return {
            "segment": "clothoid",
            "n_segments": self.n_segments,
            "zones_p1": list(self.zones_p1),
            "zones_p2": list(self.zones_p2),
            "needle_angle_deg": self.needle_angle_deg,
            "needle_angle_p1_deg": self.needle_angle_p1_deg,
            "needle_angle_p2_deg": self.needle_angle_p2_deg,
            "needle_angle_p1_raw_deg": self.needle_angle_p1_raw_deg,
            "needle_angle_p2_raw_deg": self.needle_angle_p2_raw_deg,
            "correction_p1_deg": self.correction_p1_deg,
            "correction_p2_deg": self.correction_p2_deg,
            "sigma_p1_deg": self.sigma_p1_deg,
            "sigma_p2_deg": self.sigma_p2_deg,
            "bond": sh.bond,
            "apex_radius_px": sh.apex_radius_px,
            "equator_radius_px": sh.equator_radius_px,
            "model_needle_angle_deg": sh.needle_angle_deg,
            "box_rms_px": sh.rms_px,
            "box_evaluations": sh.evaluations,
            "surface_tension_mN_m": self.surface_tension_mN_m,
            "laplace": dict(self.laplace),
            "axis_tilt_deg": self.axis_tilt_deg,
            "contact_slide_px": [float(v) for v in self.contact_slide_px],
            "contact_points_xy": [[float(v) for v in c] for c in self.contact_points],
            "apex_xy": [float(self.apex_xy[0]), float(self.apex_xy[1])],
            "rmse_px": self.rmse_px,
            "edge_noise_px": self.edge_noise_px,
            "n_points": self.n_points,
            "init": self.init,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "levels": [
                {"n_p1": lv.n_p1, "n_p2": lv.n_p2, "needle_p1_deg": lv.theta_p1_deg,
                 "needle_p2_deg": lv.theta_p2_deg, "rmse_px": lv.rmse_px, "bic": lv.bic}
                for lv in self.levels
            ],
        }


# -----------------------------------------------------------------------------
# Physics layout, refinement, bias, Laplace slope
# -----------------------------------------------------------------------------


def _side_targets(shape: PendantShape, lay: _ZoneLayout, side: int):
    """Arc-length positions (px from the apex) and headings of one side's nodes."""
    z1, z2 = lay.zones(side)
    s_nodes = np.concatenate([shape.join_positions(z1, 1), [shape.equator_s_px], shape.join_positions(z2, 2)])
    phi = np.interp(s_nodes, shape.s_px, shape.phi)
    heading = np.pi - phi if side == 1 else phi
    return s_nodes, heading


def _map_to_edge(edge: _Edge, i_apex: int, axis: float, apex_y: float, shape: PendantShape,
                 s_nodes: np.ndarray, side: int) -> np.ndarray:
    """Edge positions of the profile points at ``s_nodes`` on one side, strictly outward."""
    sign = -1.0 if side == 1 else 1.0
    target = np.column_stack([axis + sign * np.interp(s_nodes, shape.s_px, shape.x_px),
                              apex_y + np.interp(s_nodes, shape.s_px, shape.z_px)])
    if side == 1:
        q, seg = edge.q[: i_apex + 1][::-1], edge.seg[: i_apex + 1][::-1]
    else:
        q, seg = edge.q[i_apex:], edge.seg[i_apex:]
    if len(q) < 3:
        raise ValueError("side too short")
    idx = np.argmin(((q[None, 1:-1, :] - target[:, None, :]) ** 2).sum(-1), axis=1) + 1
    t = seg[idx].astype(float)
    step = 1e-3 * edge.total
    for k in range(len(t)):
        prev = seg[0] if k == 0 else t[k - 1]
        if side == 1 and t[k] >= prev - step:
            t[k] = prev - step
        if side == 2 and t[k] <= prev + step:
            t[k] = prev + step
    return t


def _initial_params(edge: _Edge, anchors: dict[str, Any], shape: PendantShape, lay: _ZoneLayout):
    i_apex = anchors["i_apex"]
    sides = []
    for side in (1, 2):
        s_nodes, heading = _side_targets(shape, lay, side)
        t = _map_to_edge(edge, i_apex, anchors["axis"], anchors["apex_y"], shape, s_nodes, side)
        sides.append((t, np.zeros(len(t)), heading))
    phi_n = np.radians(shape.needle_angle_deg)
    return _pack(lay, float(edge.seg[i_apex]), 0.0, sides, np.pi - phi_n, phi_n)


def _needle_angles(p: np.ndarray, lay: _ZoneLayout) -> tuple[float, float]:
    """Needle angles (Young-Laplace φ at the contacts), degrees."""
    return (float(np.degrees(np.pi - p[lay.contact_index(1)])),
            float(np.degrees(p[lay.contact_index(2)])))


def _split(problem: _Problem, params: np.ndarray, cap: int) -> tuple[_ZoneLayout, np.ndarray] | None:
    """Insert a node at the middle of the worst segment of each side (``None`` if capped)."""
    e = problem._evaluate(params)
    lay: _ZoneLayout = problem.layout
    n_seg = lay.n1 + lay.n2 + 2
    score = np.bincount(e["own"], weights=e["r"] ** 2, minlength=n_seg)
    chains = problem.chains(params)
    new_sides, new_zones = [], []
    for side, segs, offset in ((1, chains[0], 0), (2, chains[1], lay.n1 + 1)):
        t, h, th = (list(a) for a in _unpack(lay, params, side))
        z1, z2 = lay.zones(side)
        order = np.argsort(score[offset : offset + len(segs)])[::-1]
        j = next((int(k) for k in order if (z1 if k < z1 else z2) < cap), None)
        if j is None:
            return None
        bounds = [params[0], *t, 0.0 if side == 1 else problem.edge.total]
        seg = segs[j]
        half = 0.5 * seg.length
        t.insert(j, 0.5 * (bounds[j] + bounds[j + 1]))
        h.insert(j, 0.0)
        th.insert(j, seg.heading + seg.curvature * half + 0.5 * seg.curvature_rate * half**2)
        new_sides.append((np.array(t), np.array(h), np.array(th)))
        new_zones.append((z1 + 1, z2) if j < z1 else (z1, z2 + 1))
    new_lay = _ZoneLayout(*new_zones, slide=lay.slide)
    p = _pack(new_lay, params[0], params[1], new_sides,
              params[lay.contact_index(1)], params[lay.contact_index(2)], _slides(lay, params))
    return new_lay, p


def _problem(lay: _ZoneLayout, edge: _Edge, e1: np.ndarray, e2: np.ndarray, g2_weight: float) -> _ZoneProblem:
    n_seg = lay.n1 + lay.n2 + 2
    span = edge.total / n_seg
    return _ZoneProblem(lay, edge, e1, e2, g2_weight, span**2 / 8.0 * np.sqrt(len(edge.q) / n_seg))


def _refit_zones(p: np.ndarray, problem: _ZoneProblem, max_offset: float, loss: str, max_nfev: int,
                 max_slide: float) -> np.ndarray:
    """Bounded least squares of one layout (clothoid bounds plus the contact slides)."""
    lay = problem.layout
    lo, hi = _bounds(p, lay, problem.edge.total, max_offset, 1.0)
    if lay.slide:
        for side in (1, 2):
            i = lay.slide_index(side)
            lo[i], hi[i] = -max_slide, max_slide
    p0 = np.clip(p, lo + 1e-9, hi - 1e-9)
    res = least_squares(
        problem.residuals, p0, jac=problem.jacobian, bounds=(lo, hi), x_scale="jac",
        max_nfev=max_nfev, loss=loss, f_scale=1.0, ftol=1e-8, xtol=1e-7, gtol=1e-8,
    )
    return res.x


def _model_bias(shape: PendantShape, anchors: dict[str, Any], lay: _ZoneLayout, max_offset: float,
                max_nfev: int, g2_weight: float, max_slide: float, step_px: float = 3.0):
    """Errors of this layout on the noise-free matched profile.

    Returns
    -------
    needle : tuple of float
        Needle-angle correction per side, degrees (exact minus fitted).
    slope_ratio : float
        Exact over fitted Laplace slope ``c1``: a spline sized for position
        flattens the curvature-height relation slightly.
    """
    axis, ya = anchors["axis"], anchors["apex_y"]
    s = np.append(np.arange(0.0, shape.s_px[-1], step_px), shape.s_px[-1])
    r = np.interp(s, shape.s_px, shape.x_px)
    z = np.interp(s, shape.s_px, shape.z_px)
    left = np.column_stack([axis - r, ya + z])
    right = np.column_stack([axis + r, ya + z])
    medge = _Edge(np.vstack([left[::-1], right[1:]]))
    manchors = dict(anchors, i_apex=len(left) - 1)
    p0 = _initial_params(medge, manchors, shape, lay)
    problem = _problem(lay, medge, left[-1], right[-1], g2_weight)
    p = _refit_zones(p0, problem, max_offset, "linear", max_nfev, max_slide)
    fitted = _needle_angles(p, lay)
    c1_fit = _laplace_fit(problem.chains(p), anchors)["c1_per_px2"]
    c1_exact = -shape.bond / shape.apex_radius_px**2
    ratio = c1_exact / c1_fit if np.isfinite(c1_fit) and c1_fit < 0 else float("nan")
    return (shape.needle_angle_deg - fitted[0], shape.needle_angle_deg - fitted[1]), float(ratio)


def _angle_sigmas(problem: _Problem, p: np.ndarray, rss: float, m: int) -> tuple[float, float]:
    jac = problem.jacobian(p)[:m]
    dof = max(m - problem.layout.size, 1)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * (rss / dof)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    lay = problem.layout
    return tuple(float(np.degrees(np.sqrt(max(cov[i, i], 0.0)))) for i in (lay.contact_index(1), lay.contact_index(2)))


def _laplace_fit(chains, anchors: dict[str, Any], step_px: float = 2.0) -> dict[str, float]:
    """Linear fit of the spline's mean curvature against height above the apex.

    ``κ_m + sin φ / r = c0 + c1 z``: ``c0 = 2 / b`` and ``c1 = -Δρ g / γ``
    (pixel units). Samples closer to the axis than 10 % of the equatorial
    radius are left out, where ``sin φ / r`` amplifies the axis error.
    """
    axis, ya, r_eq = anchors["axis"], anchors["apex_y"], anchors["r_eq"]
    zs, ks = [], []
    for side, segs in ((1, chains[0]), (2, chains[1])):
        for seg in segs:
            s = np.arange(0.5 * step_px, seg.length, step_px)
            if s.size == 0:
                continue
            pts = seg.points(s)
            psi = seg.heading + seg.curvature * s + 0.5 * seg.curvature_rate * s**2
            kap = seg.curvature + seg.curvature_rate * s
            phi, k_m = (np.pi - psi, -kap) if side == 1 else (psi, kap)
            r = np.abs(pts[:, 0] - axis)
            ok = r > 0.1 * r_eq
            zs.append(pts[ok, 1] - ya)
            ks.append(k_m[ok] + np.sin(phi[ok]) / r[ok])
    z = np.concatenate(zs) if zs else np.empty(0)
    k = np.concatenate(ks) if ks else np.empty(0)
    if len(z) < 10:
        return {"c0_per_px": float("nan"), "c1_per_px2": float("nan"), "bond": float("nan"),
                "apex_radius_px": float("nan"), "rms_per_px": float("nan")}
    c1, c0 = np.polyfit(z, k, 1)
    b = 2.0 / c0 if c0 > 0 else float("nan")
    return {"c0_per_px": float(c0), "c1_per_px2": float(c1), "bond": float(-c1 * b * b),
            "apex_radius_px": float(b), "rms_per_px": float(np.std(k - (c0 + c1 * z)))}


def _gate(fit: PendantSplineFit, max_rmse_px: float) -> list[str]:
    reasons = []
    noise = fit.edge_noise_px
    for side, phi in (("p1", fit.needle_angle_p1_deg), ("p2", fit.needle_angle_p2_deg)):
        if not (np.isfinite(phi) and 0.0 < phi < 200.0):
            reasons.append(f"{side}_needle_angle_out_of_range")
    if fit.rmse_px > max(max_rmse_px, 4.0 * noise):
        reasons.append("residual_above_edge_noise")
    if fit.shape.rms_px > max(2.0, 4.0 * noise):
        reasons.append("young_laplace_mismatch")
    if not 0.005 < fit.shape.bond < 0.98:
        reasons.append("bond_out_of_range")
    return reasons


# -----------------------------------------------------------------------------
# Public fit
# -----------------------------------------------------------------------------


def fit_pendant_spline(
    points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    axis_direction: np.ndarray | None = None,
    px_per_mm: float | None = None,
    delta_rho: float | None = None,
    g: float = 9.80665,
    max_segments_per_zone: int = 4,
    max_node_offset_px: float = 8.0,
    noise_floor_px: float = 0.02,
    tol_px: float = 0.25,
    tol_noise_fraction: float = 2.0,
    g2_weight: float = 1.0,
    bias_correction: BiasMode = "model",
    loss: str = "soft_l1",
    max_nfev: int = 100,
    max_rmse_px: float = 1.5,
    adequacy: float = 1.5,
    max_contact_slide_px: float = 2.0,
) -> PendantSplineFit:
    """Fit the two-zone clothoid spline to an ordered pendant edge.

    Parameters
    ----------
    points : np.ndarray
        Edge points ordered from ``P1`` over the apex to ``P2``, image
        coordinates (see :func:`extract_pendant_interface`).
    p1, p2 : np.ndarray
        Contact points on the needle; the spline passes through them.
    axis_direction : np.ndarray, optional
        Image direction from the apex to the needle. By default it is
        estimated from the midpoints of the two sides in zone 1.
    px_per_mm : float, optional
        Scale, for the surface tension.
    delta_rho : float, optional
        Density difference in kg/m³, for the surface tension.
    g : float, optional
        Gravitational acceleration, m/s².
    max_segments_per_zone : int, optional
        Cap on clothoids per zone and side.
    max_node_offset_px : float, optional
        How far a node may leave the observed edge along its normal.
    noise_floor_px : float, optional
        Residual RMS floor of the BIC.
    tol_px, tol_noise_fraction : float, optional
        Zone counts are sized for a deviation of
        ``max(tol_px, tol_noise_fraction * edge_noise)``.
    g2_weight : float, optional
        Weight of the curvature-continuity penalty at joins and the apex.
    bias_correction : {"model", "off"}, optional
        Subtract the layout's needle-angle error on the matched profile.
    loss : str, optional
        Robust loss for :func:`scipy.optimize.least_squares`.
    max_nfev : int, optional
        Evaluation budget per fit.
    max_rmse_px : float, optional
        Quality gate on the residual RMS (with four times the edge noise).
    adequacy : float, optional
        Nodes are added (and kept only if the BIC improves) while the residual
        exceeds this multiple of the expected one.
    max_contact_slide_px : float, optional
        How far each contact may move perpendicular to the axis, at fixed
        height, to meet the edge; 0 forces the spline through the given points.

    Returns
    -------
    PendantSplineFit
        Fitted model, surface tension, needle angles and diagnostics.

    Raises
    ------
    ValueError
        If the edge is too short, has no equator, or no Young-Laplace profile
        reaches the needle.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 20:
        raise ValueError("pendant spline needs at least 20 edge points")
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    frame = _Frame.from_contacts(p1, p2, pts)
    if axis_direction is not None:
        frame = _axis_frame(p1, p2, np.asarray(axis_direction, float))
    else:
        slope = _symmetry_slope(frame.to_local(pts))
        if slope is not None:
            up = frame.to_image(np.array([slope, 1.0])) - frame.to_image(np.zeros(2))
            frame = _axis_frame(p1, p2, up[0])
    chord_normal = _Frame.from_contacts(p1, p2, pts).ey
    tilt = float(np.degrees(np.arctan2(chord_normal[0] * frame.ey[1] - chord_normal[1] * frame.ey[0],
                                       chord_normal @ frame.ey)))

    edge = _Edge(frame.to_local(pts))
    lp1 = frame.to_local(p1)[0]
    lp2 = frame.to_local(p2)[0]
    anchors = _anchors(edge.q)
    if anchors is None:
        raise ValueError("no pendant equator or apex found")
    m = len(edge.q)
    noise = estimate_edge_noise(edge.q)
    tol = max(tol_px, tol_noise_fraction * noise)

    rz = np.column_stack([np.abs(edge.q[:, 0] - anchors["axis"]), edge.q[:, 1] - anchors["apex_y"]])
    top = 0.5 * (lp1[1] + lp2[1]) - anchors["apex_y"]
    shape = fit_pendant_box(rz, anchors["r_eq"], top)
    cap = max(1, int(max_segments_per_zone))
    zones = (min(cap, shape.segments_needed(tol, 1)), min(cap, shape.segments_needed(tol, 2)))
    lay = _ZoneLayout(zones, zones, slide=max_contact_slide_px > 0)
    params = _initial_params(edge, anchors, shape, lay)
    levels: list[ArcSplineLevel] = []

    def solve(lay, params):
        problem = _problem(lay, edge, lp1, lp2, g2_weight)
        params = _refit_zones(params, problem, max_node_offset_px, loss, max_nfev, max_contact_slide_px)
        e = problem._evaluate(params)
        rss = float(e["r"] @ e["r"])
        bic = m * np.log(rss / m + noise_floor_px**2) + lay.size * np.log(m)
        a1, a2 = _needle_angles(params, lay)
        levels.append(ArcSplineLevel(lay.n1 + 1, lay.n2 + 1, a1, a2, float(np.sqrt(rss / m)), float(bic)))
        return problem, params, rss

    problem, params, rss = solve(lay, params)
    init = "physics"
    expected = max(noise, 0.5 * tol, noise_floor_px)
    best = (problem, params, rss, len(levels) - 1)
    while levels[-1].rmse_px > adequacy * expected + 0.05:
        refined = _split(problem, params, cap)
        if refined is None:
            break
        lay, params = refined
        problem, params, rss = solve(lay, params)
        if levels[-1].bic >= levels[best[3]].bic - 1e-9:
            break
        best = (problem, params, rss, len(levels) - 1)
        init = "physics+refined"
    problem, params, rss, sel = best
    lay = problem.layout

    sig1, sig2 = _angle_sigmas(problem, params, rss, m)
    lv = levels[sel]
    corr1 = corr2 = 0.0
    slope_ratio = 1.0
    if bias_correction == "model":
        (corr1, corr2), slope_ratio = _model_bias(shape, anchors, lay, max_node_offset_px, max_nfev, g2_weight,
                                                   max_contact_slide_px)
    chains = problem.chains(params)
    laplace = _laplace_fit(chains, anchors)
    laplace["slope_correction"] = slope_ratio
    gamma = float("nan")
    if px_per_mm and delta_rho:
        gamma = shape.surface_tension_mN_m(px_per_mm, delta_rho, g)
        px_per_m = px_per_mm * 1000.0
        for key, c1 in (("surface_tension_raw_mN_m", laplace["c1_per_px2"]),
                        ("surface_tension_mN_m", laplace["c1_per_px2"] * slope_ratio)):
            ok = np.isfinite(c1) and c1 < 0
            laplace[key] = float(-delta_rho * g / (c1 * px_per_m**2) * 1000.0) if ok else float("nan")
    apex_local = problem._side_nodes(params, 1)[0][0]
    fit = PendantSplineFit(
        p1_segments=chains[0], p2_segments=chains[1],
        zones_p1=lay.zones1, zones_p2=lay.zones2,
        needle_angle_p1_deg=lv.theta_p1_deg + corr1, needle_angle_p2_deg=lv.theta_p2_deg + corr2,
        needle_angle_p1_raw_deg=lv.theta_p1_deg, needle_angle_p2_raw_deg=lv.theta_p2_deg,
        correction_p1_deg=corr1, correction_p2_deg=corr2,
        sigma_p1_deg=sig1, sigma_p2_deg=sig2,
        shape=shape, surface_tension_mN_m=gamma, laplace=laplace, axis_tilt_deg=tilt,
        contact_slide_px=_slides(lay, params),
        apex_xy=tuple(float(v) for v in frame.to_image(apex_local)[0]),
        rmse_px=lv.rmse_px, edge_noise_px=noise, n_points=m, levels=levels, init=init,
        rejection_reasons=[], _frame=frame, _anchors=anchors,
    )
    fit.rejection_reasons = _gate(fit, max_rmse_px)
    return fit


def refine_pendant_on_image(
    image: np.ndarray,
    fit: PendantSplineFit,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    iterations: int = 1,
    search_px: float = 4.0,
    step_px: float = 1.0,
    needle_margin_px: float = 3.0,
    min_relative_contrast: float = 0.3,
    **fit_kwargs: Any,
) -> tuple[PendantSplineFit, np.ndarray]:
    """Move the edge points to the steepest drop-edge contrast along the spline normals.

    A binary-mask contour traces the boundary pixels, a fraction of a pixel
    inside the true edge, which shrinks the drop and biases the surface
    tension. Each iteration samples the spline, finds the sub-pixel gradient
    peak of the dark-drop polarity on every normal, and refits. Within
    ``needle_margin_px`` of the contact line the normals graze the needle, so
    only the contact points anchor the ends there.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image the contour came from.
    fit : PendantSplineFit
        Initial fit.
    p1, p2 : np.ndarray
        Contact points.
    iterations : int, optional
        Sample-and-refit rounds.
    search_px : float, optional
        Half-length of the normal profiles.
    step_px : float, optional
        Spacing of the samples along the spline.
    needle_margin_px : float, optional
        Band below the contact line where image edges are not used.
    min_relative_contrast : float, optional
        Samples weaker than this fraction of the median peak are discarded.
    **fit_kwargs
        Forwarded to :func:`fit_pendant_spline`.

    Returns
    -------
    fit : PendantSplineFit
        The refined fit.
    points : np.ndarray
        The image-derived edge points it was fitted to.
    """
    from menipy.common.arc_spline import sample_edge_along_normals

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    points = fit.sample(step_px)
    for _ in range(max(1, iterations)):
        frame, a = fit._frame, fit._anchors
        z_eq = float(fit.shape.z_px[fit.shape.i_equator])
        interior = frame.to_image(np.array([a["axis"], a["apex_y"] + z_eq]))[0]
        curve = fit.sample(step_px)
        found, strength = sample_edge_along_normals(image, curve, search_px=search_px, interior_point=interior)
        contact_y = min(frame.to_local(p1)[0, 1], frame.to_local(p2)[0, 1])
        good = (frame.to_local(found)[:, 1] < contact_y - needle_margin_px) & (strength > 0)
        good &= frame.to_local(curve)[:, 1] < contact_y - needle_margin_px
        if good.any():
            good &= strength >= min_relative_contrast * float(np.median(strength[good]))
        if good.sum() < 20:
            break
        points = np.vstack([p1, found[good], p2])
        fit = fit_pendant_spline(points, p1, p2, **fit_kwargs)
    return fit, points


def fit_pendant_contour(
    contour: np.ndarray, p1: np.ndarray, p2: np.ndarray, **kwargs: Any
) -> tuple[PendantSplineFit, np.ndarray]:
    """Extract the drop edge from a silhouette and fit the two-zone spline.

    Parameters
    ----------
    contour : np.ndarray
        Ordered drop silhouette (needle shaft optional), shape ``(N, 2)``.
    p1, p2 : np.ndarray
        Contact points on the needle.
    **kwargs
        Forwarded to :func:`fit_pendant_spline`.

    Returns
    -------
    fit : PendantSplineFit
        The fitted model.
    interface : np.ndarray
        The edge points that were fitted.
    """
    interface = extract_pendant_interface(contour, p1, p2)
    return fit_pendant_spline(interface, p1, p2, **kwargs), interface
