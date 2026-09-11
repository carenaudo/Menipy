"""G1 clothoid spline model of a sessile drop interface.

Same geometry as :mod:`menipy.common.arc_spline` -- two chains from the apex
(tangent parallel to the contact chord) to the contact points, fitted in a frame
where the chord is horizontal -- but every segment is a clothoid (curvature
linear in arc length) instead of a circular arc. A drop's curvature varies
smoothly along its profile, so a linear-curvature segment follows it with far
fewer pieces: a clothoid deviates by ``|κ''| L⁴ / 384`` against an arc's
``|κ'| L³ / 40.5``.

Each segment is the unique G1 Hermite clothoid between two nodes with given
tangent directions (Bertolazzi & Frego, 2015), so a segment depends only on its
own two nodes -- no heading propagates down the chain. The node headings are
fitted parameters; at the contact points they *are* the contact angles, so the
angle's uncertainty comes straight from the fit covariance. A soft penalty
equalizes the curvature across each join (G2), the apex included.

Residuals are signed orthogonal distances to foot points found by Newton
iteration; by the envelope theorem their derivative is ``-n · ∂C/∂p`` at the
foot point, and ``∂C/∂p`` follows from implicit differentiation of the Hermite
solution. Generalized Fresnel integrals are evaluated by Gauss-Legendre
quadrature, so the whole Jacobian is analytic.

The physics prior of :mod:`menipy.math.sessile_box` sizes the spline: segment
counts from ``∫ (|κ''| / (384 tol))^(1/4) ds`` and joins at equal quantiles of
``∫ |κ''|^(1/4) ds`` on the box-matched Young-Laplace profile; the remaining
discretization bias is measured on that profile and subtracted.

References
----------
Bertolazzi, E. and Frego, M. (2015). G1 fitting with clothoids. Mathematical
Methods in the Applied Sciences, 38(5), 881-897.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from menipy.common.arc_spline import (
    ArcSplineLevel,
    _apex_vertex,
    _Edge,
    _end_angle,
    _flank_frames,
    _Frame,
    estimate_edge_noise,
    extract_interface,
)

_Q = 12
_GL_X, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_Q)
GL_T = 0.5 * (_GL_X + 1.0)  # nodes on [0, 1]
GL_W = 0.5 * _GL_WEIGHTS

BiasMode = Literal["model", "off"]


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ClothoidSegment:
    """One clothoid of the spline, in the fitting frame.

    Attributes
    ----------
    start : np.ndarray
        Start point ``(x, y)``.
    heading : float
        Tangent direction at the start, radians.
    curvature : float
        Curvature at the start, ``1/px``.
    curvature_rate : float
        ``d(curvature)/ds``, ``1/px^2``.
    length : float
        Arc length, pixels.
    """

    start: np.ndarray
    heading: float
    curvature: float
    curvature_rate: float
    length: float

    def points(self, s: np.ndarray) -> np.ndarray:
        """Return points at arc-length positions ``s`` along the segment.

        Parameters
        ----------
        s : np.ndarray
            Arc-length positions, shape ``(M,)``.

        Returns
        -------
        np.ndarray
            Points, shape ``(M, 2)``.
        """
        s = np.asarray(s, float)
        u = s[:, None] * GL_T[None, :]
        psi = self.heading + self.curvature * u + 0.5 * self.curvature_rate * u**2
        return self.start + s[:, None] * np.column_stack(
            [(GL_W * np.cos(psi)).sum(1), (GL_W * np.sin(psi)).sum(1)]
        )

    @property
    def end_heading(self) -> float:
        """Tangent direction at the end of the segment, radians."""
        return self.heading + self.curvature * self.length + 0.5 * self.curvature_rate * self.length**2


# -----------------------------------------------------------------------------
# G1 Hermite clothoids and their derivatives
# -----------------------------------------------------------------------------


@dataclass
class _Hermite:
    """Vectorized G1 Hermite clothoids and differentials of their parameters.

    Differentials are 6-vectors over ``(p0x, p0y, p1x, p1y, theta0, theta1)``.
    """

    p0: np.ndarray
    th0: np.ndarray
    delta: np.ndarray
    A: np.ndarray
    L: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    dA: np.ndarray  # (S, 6)
    dL: np.ndarray  # (S, 6)
    dk0: np.ndarray  # (S, 6)
    dk1: np.ndarray  # (S, 6)
    valid: np.ndarray  # (S,) bool


def solve_hermite(p0: np.ndarray, p1: np.ndarray, th0: np.ndarray, th1: np.ndarray) -> _Hermite:
    """Solve G1 Hermite clothoids from ``(p0, th0)`` to ``(p1, th1)``.

    In the normalized frame (chord from 0 to 1) the heading is
    ``Θ(τ) = φ0 (1-τ) + φ1 τ + A (τ² - τ)``, and ``A`` solves
    ``∫₀¹ sin Θ dτ = 0`` (the end lies on the chord); then
    ``L = d / ∫₀¹ cos Θ dτ``. Newton iteration from ``A = 3 (φ0 + φ1)``.

    Parameters
    ----------
    p0, p1 : np.ndarray
        Segment end points, shape ``(S, 2)``.
    th0, th1 : np.ndarray
        Tangent directions at the ends, shape ``(S,)``.

    Returns
    -------
    _Hermite
        Clothoid parameters and their first-order differentials.
    """
    p0 = np.asarray(p0, float)
    c = np.asarray(p1, float) - p0
    d = np.maximum(np.hypot(c[:, 0], c[:, 1]), 1e-9)
    phi = np.arctan2(c[:, 1], c[:, 0])
    f0 = _wrap(th0 - phi)
    f1 = _wrap(th1 - phi)
    T, W = GL_T[None, :], GL_W[None, :]
    u = T * T - T
    A = 3.0 * (f0 + f1)
    for _ in range(30):
        th = f0[:, None] * (1.0 - T) + f1[:, None] * T + A[:, None] * u
        g = (W * np.sin(th)).sum(1)
        ga = (W * np.cos(th) * u).sum(1)
        step = np.clip(g / np.where(np.abs(ga) > 1e-14, ga, -1e-14), -1.0, 1.0)
        A = A - step
        if np.max(np.abs(step)) < 1e-12:
            break
    th = f0[:, None] * (1.0 - T) + f1[:, None] * T + A[:, None] * u
    cos_t, sin_t = np.cos(th), np.sin(th)
    X = (W * cos_t).sum(1)
    valid = X > 1e-3
    Xs = np.where(valid, X, 1e-3)
    L = d / Xs
    g0 = (W * cos_t * (1.0 - T)).sum(1)
    g1 = (W * cos_t * T).sum(1)
    ga = (W * cos_t * u).sum(1)
    ga = np.where(np.abs(ga) > 1e-14, ga, -1e-14)
    s0 = (W * sin_t * (1.0 - T)).sum(1)
    s1 = (W * sin_t * T).sum(1)
    sa = (W * sin_t * u).sum(1)

    n = len(d)
    zero = np.zeros(n)
    dphi = np.column_stack([c[:, 1], -c[:, 0], -c[:, 1], c[:, 0], zero, zero]) / (d**2)[:, None]
    dd = np.column_stack([-c[:, 0], -c[:, 1], c[:, 0], c[:, 1], zero, zero]) / d[:, None]
    e5 = np.zeros((n, 6))
    e5[:, 4] = 1.0
    e6 = np.zeros((n, 6))
    e6[:, 5] = 1.0
    dphi0 = e5 - dphi
    dphi1 = e6 - dphi
    dA = -(g0[:, None] * dphi0 + g1[:, None] * dphi1) / ga[:, None]
    dX = -(s0[:, None] * dphi0 + s1[:, None] * dphi1 + sa[:, None] * dA)
    dL = dd / Xs[:, None] - (d / Xs**2)[:, None] * dX
    delta = f1 - f0
    k0 = (delta - A) / L
    k1 = (delta + A) / L
    ddelta = e6 - e5
    dk0 = (ddelta - dA) / L[:, None] - (k0 / L)[:, None] * dL
    dk1 = (ddelta + dA) / L[:, None] - (k1 / L)[:, None] * dL
    return _Hermite(p0, np.asarray(th0, float), delta, A, L, k0, k1, dA, dL, dk0, dk1, valid)


def _curve(h: _Hermite, seg: np.ndarray, tau: np.ndarray):
    """Points, headings and quadrature integrals at normalized positions ``tau``."""
    t = tau[:, None] * GL_T[None, :]
    th0, delta, A = h.th0[seg][:, None], h.delta[seg][:, None], h.A[seg][:, None]
    psi_q = th0 + (delta - A) * t + A * t * t
    cw = GL_W * np.cos(psi_q)
    sw = GL_W * np.sin(psi_q)
    integral = tau[:, None] * np.column_stack([cw.sum(1), sw.sum(1)])
    point = h.p0[seg] + h.L[seg][:, None] * integral
    psi = h.th0[seg] + (h.delta[seg] - h.A[seg]) * tau + h.A[seg] * tau**2
    return point, psi, integral, t, cw, sw


# -----------------------------------------------------------------------------
# Model layout and problem
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """Parameter layout.

    ``[t_apex, h_apex, (t, h, theta) per P1-side node (apex outwards),
    (t, h, theta) per P2-side node, theta_P1, theta_P2]``.
    """

    n1: int
    n2: int

    @property
    def size(self) -> int:
        return 2 + 3 * (self.n1 + self.n2) + 2

    def block(self, side: int) -> int:
        return 2 if side == 1 else 2 + 3 * self.n1

    def contact_index(self, side: int) -> int:
        return 2 + 3 * (self.n1 + self.n2) + (side - 1)

    # interface shared with other layouts (see menipy.common.pendant_spline):
    # parameter indices of each interior node, apex outwards; -1 marks a fixed heading
    def n_nodes(self, side: int) -> int:
        return self.n1 if side == 1 else self.n2

    def t_index(self, side: int) -> np.ndarray:
        return self.block(side) + 3 * np.arange(self.n_nodes(side))

    def h_index(self, side: int) -> np.ndarray:
        return self.t_index(side) + 1

    def theta_index(self, side: int) -> np.ndarray:
        return self.t_index(side) + 2

    def fixed_theta(self, side: int) -> np.ndarray:
        return np.full(self.n_nodes(side), np.nan)


class _Problem:
    """Residuals and analytic Jacobian of one layout, sharing a cache."""

    def __init__(self, layout: _Layout, edge: _Edge, p1: np.ndarray, p2: np.ndarray,
                 g2_weight: float, g2_scale: float):
        self.layout = layout
        self.edge = edge
        self.ends = (np.asarray(p1, float), np.asarray(p2, float))
        self.g2_weight = g2_weight
        self.g2_scale = g2_scale
        self._key: bytes | None = None
        self._cache: dict[str, Any] = {}

    # nodes of one side, apex first: positions, their derivatives and headings
    def _side_nodes(self, p: np.ndarray, side: int):
        lay = self.layout
        n = lay.n_nodes(side)
        idx_t = np.concatenate([[0], lay.t_index(side)])
        idx_h = np.concatenate([[1], lay.h_index(side)])
        pos, d_t, d_h = self.edge.point(p[idx_t], p[idx_h])
        n_par = lay.size
        rows = np.arange(n + 1)
        d_pos = np.zeros((n + 2, 2, n_par))
        d_pos[rows, :, idx_t] = d_t
        d_pos[rows, :, idx_h] = d_h
        pos = np.vstack([pos, self.ends[side - 1]])
        heading = np.empty(n + 2)
        d_head = np.zeros((n + 2, n_par))
        heading[0] = np.pi if side == 1 else 0.0
        mid = lay.theta_index(side)
        free = mid >= 0
        heading[1 : n + 1] = np.where(free, p[np.maximum(mid, 0)], lay.fixed_theta(side))
        d_head[np.arange(1, n + 1)[free], mid[free]] = 1.0
        ci = lay.contact_index(side)
        heading[-1] = p[ci]
        d_head[-1, ci] = 1.0
        return pos, d_pos, heading, d_head

    def owner(self, p: np.ndarray):
        """Segment (P1 side first, apex outwards) and normalized position of each point."""
        lay = self.layout
        ta = p[0]
        seg = self.edge.seg
        t1 = p[lay.t_index(1)]
        t2 = p[lay.t_index(2)]
        b1 = np.concatenate([[ta], t1, [0.0]])  # decreasing
        b2 = np.concatenate([[ta], t2, [self.edge.total]])  # increasing
        own = np.empty(len(seg), dtype=int)
        tau = np.empty(len(seg))
        on1 = seg < ta
        j1 = np.clip(np.searchsorted(-b1[1:-1], -seg[on1], side="left"), 0, lay.n1)
        own[on1] = j1
        tau[on1] = (b1[j1] - seg[on1]) / np.maximum(b1[j1] - b1[j1 + 1], 1e-9)
        j2 = np.clip(np.searchsorted(b2[1:-1], seg[~on1], side="right"), 0, lay.n2)
        own[~on1] = lay.n1 + 1 + j2
        tau[~on1] = (seg[~on1] - b2[j2]) / np.maximum(b2[j2 + 1] - b2[j2], 1e-9)
        return own, np.clip(tau, 0.0, 1.0)

    def _evaluate(self, p: np.ndarray) -> dict[str, Any]:
        key = p.tobytes()
        if key == self._key:
            return self._cache
        s1 = self._side_nodes(p, 1)
        s2 = self._side_nodes(p, 2)
        seg_nodes = []  # (start node array index, end node array index) into stacked arrays
        pos = np.vstack([s1[0], s2[0]])
        d_pos = np.concatenate([s1[1], s2[1]])
        head = np.concatenate([s1[2], s2[2]])
        d_head = np.concatenate([s1[3], s2[3]])
        n1n = len(s1[0])
        for j in range(n1n - 1):
            seg_nodes.append((j, j + 1))
        for j in range(len(s2[0]) - 1):
            seg_nodes.append((n1n + j, n1n + j + 1))
        a_idx = np.array([a for a, _ in seg_nodes])
        b_idx = np.array([b for _, b in seg_nodes])
        h = solve_hermite(pos[a_idx], pos[b_idx], head[a_idx], head[b_idx])

        # foot points by Newton on (C - q) . C' = 0, started from each point's
        # position along the edge (a warm start from the previous evaluation
        # inherits foot points of rejected trial steps and can diverge)
        own, tau = self.owner(p)
        q = self.edge.q
        point, psi, integral, t, cw, sw = _curve(h, own, tau)
        for _ in range(8):
            L = h.L[own]
            dtheta = (h.delta[own] - h.A[own]) + 2.0 * h.A[own] * tau
            cos_p, sin_p = np.cos(psi), np.sin(psi)
            diff = point - q
            # C' = L e(psi), C'' = L dpsi/dtau e_perp(psi)
            f = L * (diff[:, 0] * cos_p + diff[:, 1] * sin_p)
            fp = L * L + L * dtheta * (-diff[:, 0] * sin_p + diff[:, 1] * cos_p)
            step = f / np.where(fp > 1e-9, fp, 1e-9)
            tau = np.clip(tau - step, -0.25, 1.25)
            point, psi, integral, t, cw, sw = _curve(h, own, tau)
            if float(np.max(np.abs(step * L))) < 1e-4:
                break
        normal = np.column_stack([-np.sin(psi), np.cos(psi)])
        r = ((q - point) * normal).sum(1)
        self._key = key
        self._cache = {
            "h": h, "own": own, "tau": tau, "normal": normal, "integral": integral,
            "t": t, "cw": cw, "sw": sw, "r": r, "a_idx": a_idx, "b_idx": b_idx,
            "d_pos": d_pos, "d_head": d_head, "n_side1": n1n - 1,
        }
        return self._cache

    def _g2_rows(self, e: dict[str, Any]):
        """Curvature jumps at interior joins and across the apex, and their jacobian."""
        h = e["h"]
        n1 = e["n_side1"]
        n_seg = len(h.L)
        pairs = [(j, j + 1, 1.0) for j in range(n1 - 1)]
        pairs += [(j, j + 1, 1.0) for j in range(n1, n_seg - 1)]
        rows = [(h.k1[a] - h.k0[b], h.dk1[a], -h.dk0[b], a, b) for a, b, _ in pairs]
        # the apex: the P1 chain turns the other way, so kappa_P1 = -kappa_P2 there
        rows.append((h.k0[0] + h.k0[n1], h.dk0[0], h.dk0[n1], 0, n1))
        return rows

    def residuals(self, p: np.ndarray) -> np.ndarray:
        e = self._evaluate(p)
        extra = []
        if self.g2_weight > 0:
            w = self.g2_weight * self.g2_scale
            extra = [w * val for val, *_ in self._g2_rows(e)]
        valid_penalty = 1e3 * np.count_nonzero(~e["h"].valid)
        return np.concatenate([e["r"], extra, [valid_penalty]])

    def _segment_to_global(self, e, seg: np.ndarray, local: np.ndarray) -> np.ndarray:
        """Map (K, 6) differentials over segment primitives to (K, P) parameters."""
        a = e["a_idx"][seg]
        b = e["b_idx"][seg]
        d_pos, d_head = e["d_pos"], e["d_head"]
        return (
            local[:, 0:1] * d_pos[a, 0] + local[:, 1:2] * d_pos[a, 1]
            + local[:, 2:3] * d_pos[b, 0] + local[:, 3:4] * d_pos[b, 1]
            + local[:, 4:5] * d_head[a] + local[:, 5:6] * d_head[b]
        )

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        e = self._evaluate(p)
        h, own, tau = e["h"], e["own"], e["tau"]
        n, t, cw, sw = e["normal"], e["t"], e["cw"], e["sw"]
        tau_c = tau[:, None]
        # ∫₀^τ e⊥(ψ) w dτ' for weights w = 1 - τ', τ', τ'² - τ'
        perp_x, perp_y = -sw, cw  # (M, Q) weighted
        def perp_integral(weight):
            return tau_c * np.column_stack([(perp_x * weight).sum(1), (perp_y * weight).sum(1)])
        j1 = perp_integral(1.0 - t)
        j2 = perp_integral(t)
        j3 = perp_integral(t * t - t)
        L = h.L[own]
        n_i = (n * e["integral"]).sum(1)
        n_j1, n_j2, n_j3 = (n * j1).sum(1), (n * j2).sum(1), (n * j3).sum(1)
        local = np.zeros((len(own), 6))
        local[:, 0] = -n[:, 0]
        local[:, 1] = -n[:, 1]
        local -= n_i[:, None] * h.dL[own]
        local[:, 4] -= L * n_j1
        local[:, 5] -= L * n_j2
        local -= (L * n_j3)[:, None] * h.dA[own]
        jac = [self._segment_to_global(e, own, local)]
        if self.g2_weight > 0:
            w = self.g2_weight * self.g2_scale
            for _, da, db, a, b in self._g2_rows(e):
                row = self._segment_to_global(e, np.array([a]), da[None, :])
                row += self._segment_to_global(e, np.array([b]), db[None, :])
                jac.append(w * row)
        jac.append(np.zeros((1, self.layout.size)))
        return np.vstack(jac)

    def chains(self, p: np.ndarray) -> tuple[list[ClothoidSegment], list[ClothoidSegment]]:
        e = self._evaluate(p)
        h = e["h"]
        segs = [
            ClothoidSegment(h.p0[j].copy(), float(h.th0[j]), float(h.k0[j]),
                            float((h.k1[j] - h.k0[j]) / h.L[j]), float(h.L[j]))
            for j in range(len(h.L))
        ]
        return segs[: e["n_side1"]], segs[e["n_side1"] :]


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


@dataclass
class ClothoidSplineFit:
    """Fitted clothoid spline and derived contact angles.

    Attributes
    ----------
    p1_segments, p2_segments : list of ClothoidSegment
        Chains from the apex to ``P1`` and to ``P2``, in the fitting frame.
    theta_p1_deg, theta_p2_deg : float
        Reported contact angles (raw plus bias correction).
    theta_p1_raw_deg, theta_p2_raw_deg : float
        Fitted end headings as contact angles.
    correction_p1_deg, correction_p2_deg : float
        Model-based discretization correction.
    sigma_p1_deg, sigma_p2_deg : float
        Standard deviation of each raw angle from the fit covariance.
    apex_xy : tuple of float
        Apex node, image coordinates.
    rmse_px : float
        Residual RMS of the edge points.
    n_points : int
        Edge points fitted.
    levels : list of ArcSplineLevel
        Fitted levels (segment counts, angles, residual, BIC).
    init : str
        ``"physics"`` or ``"blind"``.
    physics : dict or None
        Box-matched Young-Laplace prior per side.
    rejection_reasons : list of str
        Quality-gate failures; empty when usable.
    """

    p1_segments: list[ClothoidSegment]
    p2_segments: list[ClothoidSegment]
    theta_p1_deg: float
    theta_p2_deg: float
    theta_p1_raw_deg: float
    theta_p2_raw_deg: float
    correction_p1_deg: float
    correction_p2_deg: float
    sigma_p1_deg: float
    sigma_p2_deg: float
    apex_xy: tuple[float, float]
    rmse_px: float
    n_points: int
    levels: list[ArcSplineLevel]
    init: str
    physics: dict[str, Any] | None
    rejection_reasons: list[str]
    _frame: _Frame = field(repr=False)

    @property
    def n_segments(self) -> int:
        """Total number of clothoids."""
        return len(self.p1_segments) + len(self.p2_segments)

    @property
    def accepted(self) -> bool:
        """Whether the fit passed every quality gate."""
        return not self.rejection_reasons

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
        """Sample the spline from ``P1`` over the apex to ``P2`` (image coordinates)."""
        return self._frame.to_image(self._local_samples(step_px))

    @property
    def contact_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the spline ends, image coordinates."""
        pts = self.sample(max(1.0, 1e3))
        return pts[0], pts[-1]

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for results and exports."""
        return {
            "segment": "clothoid",
            "n_segments": self.n_segments,
            "n_segments_p1": len(self.p1_segments),
            "n_segments_p2": len(self.p2_segments),
            "n_arcs": self.n_segments,
            "n_arcs_p1": len(self.p1_segments),
            "n_arcs_p2": len(self.p2_segments),
            "theta_p1_raw_deg": self.theta_p1_raw_deg,
            "theta_p2_raw_deg": self.theta_p2_raw_deg,
            "correction_p1_deg": self.correction_p1_deg,
            "correction_p2_deg": self.correction_p2_deg,
            "sigma_p1_deg": self.sigma_p1_deg,
            "sigma_p2_deg": self.sigma_p2_deg,
            "rmse_px": self.rmse_px,
            "n_points": self.n_points,
            "apex_xy": [float(self.apex_xy[0]), float(self.apex_xy[1])],
            "init": self.init,
            "physics": self.physics,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "levels": [
                {"n_p1": lv.n_p1, "n_p2": lv.n_p2, "theta_p1_deg": lv.theta_p1_deg,
                 "theta_p2_deg": lv.theta_p2_deg, "rmse_px": lv.rmse_px, "bic": lv.bic}
                for lv in self.levels
            ],
        }


def _bounds(p: np.ndarray, lay: _Layout, total: float, max_offset: float, turn: float):
    lo = p - turn  # headings: +-turn around the start value
    hi = p + turn
    ta = p[0]
    it1, it2 = lay.t_index(1), lay.t_index(2)
    n1, n2 = len(it1), len(it2)
    chain1 = np.concatenate([[0.0], p[it1][::-1], [ta]])  # increasing edge positions
    chain = np.concatenate([chain1, p[it2], [total]])
    inner = chain[1:-1]
    lo_all = inner - 0.4 * (inner - chain[:-2])
    hi_all = inner + 0.4 * (chain[2:] - inner)
    # map back: chain order is P1 nodes (outermost first), apex, P2 nodes
    lo[0], hi[0] = lo_all[n1], hi_all[n1]
    lo[1], hi[1] = -max_offset, max_offset
    for side, it, ks in ((1, it1, n1 - 1 - np.arange(n1)), (2, it2, n1 + 1 + np.arange(n2))):
        lo[it], hi[it] = lo_all[ks], hi_all[ks]
        ih = lay.h_index(side)
        lo[ih], hi[ih] = -max_offset, max_offset
    return lo, hi


def _refit(p, problem: _Problem, max_offset: float, loss: str, max_nfev: int, turn: float = 1.0):
    lo, hi = _bounds(p, problem.layout, problem.edge.total, max_offset, turn)
    p0 = np.clip(p, lo + 1e-9, hi - 1e-9)
    res = least_squares(
        problem.residuals, p0, jac=problem.jacobian, bounds=(lo, hi), x_scale="jac",
        max_nfev=max_nfev, loss=loss, f_scale=1.0, ftol=1e-8, xtol=1e-7, gtol=1e-8,
    )
    return res.x


def _join_params(edge: _Edge, i_apex: int, apex: tuple[float, float], shape, side: int, n: int):
    """Edge positions and headings of one side's joins from the physics profile."""
    if n <= 1:
        return np.empty(0), np.empty(0)
    xa, ya = apex
    sign = -1.0 if side == 1 else 1.0
    s_join = shape.join_positions(n, order=2)
    target = np.column_stack(
        [xa + sign * np.interp(s_join, shape.s_px, shape.x_px), ya + np.interp(s_join, shape.s_px, shape.z_px)]
    )
    phi = np.interp(s_join, shape.s_px, shape.heading())
    heading = np.pi - phi if side == 1 else phi
    if side == 1:
        q, seg = edge.q[: i_apex + 1][::-1], edge.seg[: i_apex + 1][::-1]
    else:
        q, seg = edge.q[i_apex:], edge.seg[i_apex:]
    idx = np.argmin(((q[None, 1:-1, :] - target[:, None, :]) ** 2).sum(-1), axis=1) + 1
    t = seg[idx].astype(float)
    step = 1e-3 * edge.total
    for k in range(1, len(t)):
        if side == 1 and t[k] >= t[k - 1] - step:
            t[k] = t[k - 1] - step
        if side == 2 and t[k] <= t[k - 1] + step:
            t[k] = t[k - 1] + step
    return t, heading


def _pack(lay: _Layout, ta, ha, t1, h1, th1, t2, h2, th2, c1, c2) -> np.ndarray:
    p = np.empty(lay.size)
    p[0], p[1] = ta, ha
    for side, (t, h, th) in ((1, (t1, h1, th1)), (2, (t2, h2, th2))):
        b = lay.block(side)
        p[b : b + 3 * len(t) : 3] = t
        p[b + 1 : b + 3 * len(t) : 3] = h
        p[b + 2 : b + 3 * len(t) : 3] = th
    p[lay.contact_index(1)] = c1
    p[lay.contact_index(2)] = c2
    return p


def _physics_setup(edge: _Edge, lp2: np.ndarray, max_segments: int, tol: float):
    """Box-matched profiles, segment counts and initial parameters (or ``None``)."""
    from menipy.math.sessile_box import fit_side_box

    i_apex, xa, ya = _apex_vertex(edge, float(lp2[0]))
    height = -float(ya)
    widths = (float(xa), float(lp2[0] - xa))
    if height <= 1.0 or min(widths) <= 1.0 or i_apex in (0, len(edge.q) - 1):
        return None
    sides = (edge.q[: i_apex + 1][::-1], edge.q[i_apex:])
    try:
        shapes = [
            fit_side_box(np.column_stack([sg * (q[:, 0] - xa), q[:, 1] - ya]), w, height)
            for q, sg, w in zip(sides, (-1.0, 1.0), widths)
        ]
    except ValueError:
        return None
    cap = max(1, max_segments // 2)
    counts = tuple(min(cap, sh.segments_needed(tol, order=2)) for sh in shapes)
    return {"i_apex": i_apex, "apex": (float(xa), float(ya)), "shapes": shapes, "counts": counts}


def _physics_params(edge: _Edge, setup: dict[str, Any], counts: tuple[int, int]):
    i_apex = setup["i_apex"]
    sh1, sh2 = setup["shapes"]
    t1, th1 = _join_params(edge, i_apex, setup["apex"], sh1, 1, counts[0])
    t2, th2 = _join_params(edge, i_apex, setup["apex"], sh2, 2, counts[1])
    lay = _Layout(len(t1), len(t2))
    c1 = np.pi - np.radians(sh1.theta_deg)
    c2 = np.radians(sh2.theta_deg)
    p = _pack(lay, float(edge.seg[i_apex]), 0.0, t1, np.zeros(len(t1)), th1,
              t2, np.zeros(len(t2)), th2, c1, c2)
    return lay, p


def _angle_sigmas(problem: _Problem, p: np.ndarray, rss: float, m: int) -> tuple[float, float]:
    """Standard deviation of the two contact headings from the fit covariance."""
    jac = problem.jacobian(p)[:m]
    dof = max(m - problem.layout.size, 1)
    sigma2 = rss / dof
    try:
        cov = np.linalg.pinv(jac.T @ jac) * sigma2
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    lay = problem.layout
    return tuple(float(np.degrees(np.sqrt(max(cov[i, i], 0.0)))) for i in (lay.contact_index(1), lay.contact_index(2)))


def fit_clothoid_spline(
    points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    max_segments: int = 10,
    substrate_tangents: tuple[np.ndarray, np.ndarray] | None = None,
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
) -> ClothoidSplineFit:
    """Fit the apex-anchored G1 clothoid spline to ordered interface points.

    Parameters
    ----------
    points : np.ndarray
        Interface points ordered from ``P1`` to ``P2``, image coordinates.
    p1, p2 : np.ndarray
        Contact points; the spline passes through them exactly.
    max_segments : int, optional
        Maximum total number of clothoids (both sides).
    substrate_tangents : tuple of np.ndarray, optional
        Substrate direction at ``P1`` and ``P2`` (curved substrates).
    max_node_offset_px : float, optional
        How far a node may leave the observed edge along its normal.
    noise_floor_px : float, optional
        Residual RMS floor of the BIC.
    tol_px, tol_noise_fraction : float, optional
        Segment counts are sized for a deviation of
        ``max(tol_px, tol_noise_fraction * edge_noise)``.
    g2_weight : float, optional
        Weight of the curvature-continuity penalty at joins and the apex;
        0 gives a plain G1 spline.
    bias_correction : {"model", "off"}, optional
        Subtract the layout's angle error on the matched Young-Laplace profile.
    loss : str, optional
        Robust loss for :func:`scipy.optimize.least_squares`.
    max_nfev : int, optional
        Evaluation budget per fit.
    max_rmse_px : float, optional
        Quality gate on the residual RMS (with four times the edge noise).
    adequacy : float, optional
        The physics-sized fit must reach this multiple of the expected residual,
        otherwise segments are added one per side (blind refinement).

    Returns
    -------
    ClothoidSplineFit
        Fitted model, contact angles and diagnostics.

    Raises
    ------
    ValueError
        If there are too few points or the contact points coincide.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 10:
        raise ValueError("clothoid spline needs at least 10 interface points")
    frame = _Frame.from_contacts(p1, p2, pts)
    edge = _Edge(frame.to_local(pts))
    lp1 = frame.to_local(np.asarray(p1, float))[0]
    lp2 = frame.to_local(np.asarray(p2, float))[0]
    flank = _flank_frames(frame, substrate_tangents)
    m = len(edge.q)
    noise = estimate_edge_noise(edge.q)
    tol = max(tol_px, tol_noise_fraction * noise)
    levels: list[ArcSplineLevel] = []

    setup = _physics_setup(edge, lp2, max_segments, tol)
    if setup is not None:
        counts = setup["counts"]
        lay, params = _physics_params(edge, setup, counts)
        init = "physics"
    else:
        # blind start: one clothoid per side from the observed top and flanks
        i_apex, _, _ = _apex_vertex(edge, float(lp2[0]))
        lay = _Layout(0, 0)
        d1 = edge.q[0] - edge.q[min(5, i_apex)]
        d2 = edge.q[-1] - edge.q[max(len(edge.q) - 6, i_apex)]
        params = _pack(lay, float(edge.seg[i_apex]), 0.0, [], [], [], [], [], [],
                       float(np.arctan2(d1[1], d1[0])), float(np.arctan2(d2[1], d2[0])))
        counts = (1, 1)
        init = "blind"

    def solve(lay, params):
        span = edge.total / max(lay.n1 + lay.n2 + 2, 1)
        problem = _Problem(lay, edge, lp1, lp2, g2_weight, span**2 / 8.0 * np.sqrt(m / (lay.n1 + lay.n2 + 2)))
        params = _refit(params, problem, max_node_offset_px, loss, max_nfev)
        e = problem._evaluate(params)
        rss = float(e["r"] @ e["r"])
        bic = m * np.log(rss / m + noise_floor_px**2) + lay.size * np.log(m)
        th1 = _end_angle(params[lay.contact_index(1)], flank[0])
        th2 = _end_angle(params[lay.contact_index(2)], flank[1])
        levels.append(ArcSplineLevel(lay.n1 + 1, lay.n2 + 1, th1, th2, float(np.sqrt(rss / m)), float(bic)))
        return problem, params, rss

    problem, params, rss = solve(lay, params)
    expected = max(noise, 0.5 * tol, noise_floor_px)
    cap = max(1, max_segments // 2)
    best = (problem, params, rss, len(levels) - 1)
    while levels[-1].rmse_px > adequacy * expected + 0.05 and max(lay.n1, lay.n2) + 1 < cap:
        # not explained down to the noise: split the worst segment of each side,
        # but keep the split only if the BIC improves -- a binary-mask staircase
        # leaves correlated residuals that more segments would merely follow
        lay, params = _split(problem, params)
        problem, params, rss = solve(lay, params)
        if levels[-1].bic >= levels[best[3]].bic - 1e-9:
            break
        best = (problem, params, rss, len(levels) - 1)
        init = init if init == "blind" else "physics+refined"
    problem, params, rss, sel = best
    lay = problem.layout

    sig1, sig2 = _angle_sigmas(problem, params, rss, m)
    lv = levels[sel]
    corr1 = corr2 = 0.0
    if bias_correction == "model" and setup is not None:
        corr1, corr2 = _model_bias(setup, lay, lp1, lp2, flank, max_node_offset_px, max_nfev, g2_weight)
    theta1, theta2 = lv.theta_p1_deg + corr1, lv.theta_p2_deg + corr2
    seg1, seg2 = problem.chains(params)
    apex_local = problem._side_nodes(params, 1)[0][0]
    physics = None
    if setup is not None:
        physics = {
            side: {"bond": sh.bond, "theta_deg": sh.theta_deg, "apex_radius_px": sh.apex_radius_px,
                   "rms_px": sh.rms_px, "evaluations": sh.evaluations}
            for side, sh in zip(("p1", "p2"), setup["shapes"])
        }
        physics["edge_noise_px"] = noise
    fit = ClothoidSplineFit(
        p1_segments=seg1, p2_segments=seg2,
        theta_p1_deg=theta1, theta_p2_deg=theta2,
        theta_p1_raw_deg=lv.theta_p1_deg, theta_p2_raw_deg=lv.theta_p2_deg,
        correction_p1_deg=corr1, correction_p2_deg=corr2,
        sigma_p1_deg=sig1, sigma_p2_deg=sig2,
        apex_xy=tuple(float(v) for v in frame.to_image(apex_local)[0]),
        rmse_px=lv.rmse_px, n_points=m, levels=levels, init=init, physics=physics,
        rejection_reasons=[], _frame=frame,
    )
    fit.rejection_reasons = _gate(fit, noise, max_rmse_px)
    return fit


def _split(problem: _Problem, params: np.ndarray) -> tuple[_Layout, np.ndarray]:
    """Insert a node in the worst segment of each side, at its middle."""
    e = problem._evaluate(params)
    lay = problem.layout
    n_seg = lay.n1 + lay.n2 + 2
    score = np.bincount(e["own"], weights=e["r"] ** 2, minlength=n_seg)
    seg1, seg2 = problem.chains(params)
    new = {}
    for side, segs, offset in ((1, seg1, 0), (2, seg2, lay.n1 + 1)):
        n = len(segs) - 1
        b = lay.block(side)
        t = list(params[b : b + 3 * n : 3])
        hh = list(params[b + 1 : b + 3 * n : 3])
        th = list(params[b + 2 : b + 3 * n : 3])
        j = int(np.argmax(score[offset : offset + len(segs)]))
        bounds = [params[0], *t, 0.0 if side == 1 else problem.edge.total]
        seg = segs[j]
        t.insert(j, 0.5 * (bounds[j] + bounds[j + 1]))
        hh.insert(j, 0.0)
        half = 0.5 * seg.length
        th.insert(j, seg.heading + seg.curvature * half + 0.5 * seg.curvature_rate * half**2)
        new[side] = (np.array(t), np.array(hh), np.array(th))
    new_lay = _Layout(lay.n1 + 1, lay.n2 + 1)
    p = _pack(new_lay, params[0], params[1], *new[1], *new[2],
              params[lay.contact_index(1)], params[lay.contact_index(2)])
    return new_lay, p


def _model_bias(setup, lay: _Layout, lp1, lp2, flank, max_offset, max_nfev, g2_weight, step_px=3.0):
    """Angle error of this layout on the noise-free matched Young-Laplace profile."""
    xa, ya = setup["apex"]
    sides = []
    for sh, sign in zip(setup["shapes"], (-1.0, 1.0)):
        s = np.append(np.arange(0.0, sh.s_px[-1], step_px), sh.s_px[-1])
        sides.append(np.column_stack([xa + sign * np.interp(s, sh.s_px, sh.x_px), ya + np.interp(s, sh.s_px, sh.z_px)]))
    medge = _Edge(np.vstack([sides[0][::-1], sides[1][1:]]))
    msetup = dict(setup, i_apex=len(sides[0]) - 1)
    mlay, mp = _physics_params(medge, msetup, (lay.n1 + 1, lay.n2 + 1))
    m = len(medge.q)
    span = medge.total / max(mlay.n1 + mlay.n2 + 2, 1)
    problem = _Problem(mlay, medge, lp1, lp2, g2_weight, span**2 / 8.0 * np.sqrt(m / (mlay.n1 + mlay.n2 + 2)))
    mp = _refit(mp, problem, max_offset, "linear", max_nfev)
    exact = (np.pi - np.radians(setup["shapes"][0].theta_deg), np.radians(setup["shapes"][1].theta_deg))
    return (
        _end_angle(exact[0], flank[0]) - _end_angle(mp[mlay.contact_index(1)], flank[0]),
        _end_angle(exact[1], flank[1]) - _end_angle(mp[mlay.contact_index(2)], flank[1]),
    )


def _gate(fit: ClothoidSplineFit, noise: float, max_rmse_px: float) -> list[str]:
    reasons = []
    for side, theta in (("p1", fit.theta_p1_deg), ("p2", fit.theta_p2_deg)):
        if not (np.isfinite(theta) and 0.0 < theta < 180.0):
            reasons.append(f"{side}_angle_out_of_range")
    local = fit._local_samples(2.0)
    if float(np.max(local[:, 1])) > max(2.0, 4.0 * noise):
        reasons.append("model_below_contact_line")
    if fit.rmse_px > max(max_rmse_px, 4.0 * noise):
        reasons.append("residual_above_edge_noise")
    return reasons


def fit_sessile_clothoid_spline(
    contour: np.ndarray, p1: np.ndarray, p2: np.ndarray, **kwargs: Any
) -> tuple[ClothoidSplineFit, np.ndarray]:
    """Extract the interface from a closed silhouette and fit the clothoid spline.

    Parameters
    ----------
    contour : np.ndarray
        Ordered drop silhouette, shape ``(N, 2)``.
    p1, p2 : np.ndarray
        Contact points.
    **kwargs
        Forwarded to :func:`fit_clothoid_spline`.

    Returns
    -------
    fit : ClothoidSplineFit
        The fitted model.
    interface : np.ndarray
        The interface points that were fitted.
    """
    interface = extract_interface(contour, p1, p2)
    return fit_clothoid_spline(interface, p1, p2, **kwargs), interface
