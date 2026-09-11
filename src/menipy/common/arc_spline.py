"""G1 circular-arc spline model of a sessile drop interface.

The liquid-fluid interface between the two contact points is represented by the
fewest tangent-continuous circular arcs that follow the observed edge. The model
is split at the apex: the edge point farthest from the chord ``P1-P2`` has its
tangent exactly parallel to that chord for any smooth profile, so each side is an
independent arc chain that starts at the apex with a fixed tangent and ends
exactly on its contact point. The substrate (or needle) between the contact
points is not part of the model. Tilted substrates need no special handling: the
fit runs in a frame where the chord is horizontal.

The shape is physics-first, not a blind search. Each side's box (half-width
from the apex axis to its contact, apex height) selects a Young-Laplace profile
for every Bond number; a golden-section search over the Bond number
(:mod:`menipy.math.sessile_box`) finds the one that follows the edge. That
profile's curvature fixes how many arcs each side needs for a tolerance tied to
the measured edge noise and where their joins go (arc lengths ``∝ |κ'|^(-1/3)``),
so one least-squares refit finishes the job. If the arcs cannot explain the edge
down to its noise -- the drop is not a single Young-Laplace profile (pinning,
asymmetry) -- a blind coarse-to-fine search takes over: one arc per side, then
the worst arc of each side is split until the Bayesian information criterion
stops improving.

Every node (apex and arc joins) slides along the observed edge and may leave it
by a bounded normal offset; each arc is the unique one that leaves its start
node with the incoming tangent and passes through its end node. The
least-squares Jacobian is analytic.

Contact angles are the end tangents of the two chains. Constant-curvature arcs
read the end tangent low where curvature keeps rising towards the contact line.
With the physics prior, the same arc layout is fitted to the noise-free matched
profile and its angle error there is subtracted; otherwise the bias, which falls
roughly as ``1/N**2`` with ``N`` arcs on a side, is removed by a Richardson step
between the selected level and the level with half as many arcs.

:func:`refine_on_image` then moves the edge points to the steepest contrast
along the spline normals and refits, which removes the half-pixel inward bias of
binary-mask contours.

References
----------
Rosin, P. L. and West, G. A. W. (1989). Segmentation of edges into lines and
arcs. Image and Vision Computing, 7(2), 109-114.
Maier, G., Janda, F. and Schindler, A. (2012). Minimum description length arc
spline approximation of digital curves. Proc. IEEE ICIP, 1869-1872.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

RichardsonMode = Literal["auto", "on", "off"]
InitMode = Literal["physics", "blind"]
BiasMode = Literal["model", "richardson", "off"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ArcSegment:
    """One circular arc of the spline, in the fitting frame.

    Attributes
    ----------
    start : np.ndarray
        Start point ``(x, y)``.
    heading : float
        Tangent direction at the start, radians.
    curvature : float
        Signed curvature ``d(heading)/ds`` in ``1/px``.
    length : float
        Arc length in pixels.
    """

    start: np.ndarray
    heading: float
    curvature: float
    length: float

    def points(self, s: np.ndarray) -> np.ndarray:
        """Return points at arc-length positions ``s`` along the arc.

        Parameters
        ----------
        s : np.ndarray
            Arc-length positions, shape ``(M,)``.

        Returns
        -------
        np.ndarray
            Points, shape ``(M, 2)``.
        """
        return _arc_points(self.start, self.heading, self.curvature, np.asarray(s, float))

    @property
    def end_heading(self) -> float:
        """Tangent direction at the end of the arc, radians."""
        return self.heading + self.curvature * self.length


@dataclass(frozen=True)
class ArcSplineLevel:
    """Summary of one refinement level of the coarse-to-fine fit.

    Attributes
    ----------
    n_p1, n_p2 : int
        Arcs on the ``P1`` and ``P2`` sides.
    theta_p1_deg, theta_p2_deg : float
        Contact angles at this level, before any Richardson step.
    rmse_px : float
        Root mean square distance from the edge points to the spline.
    bic : float
        Bayesian information criterion of the level.
    """

    n_p1: int
    n_p2: int
    theta_p1_deg: float
    theta_p2_deg: float
    rmse_px: float
    bic: float


@dataclass
class ArcSplineFit:
    """Fitted arc spline and derived contact angles.

    Attributes
    ----------
    p1_arcs, p2_arcs : list of ArcSegment
        Arc chains from the apex to ``P1`` and to ``P2``, in the fitting frame.
    theta_p1_deg, theta_p2_deg : float
        Reported contact angles (raw angle plus the bias correction).
    theta_p1_raw_deg, theta_p2_raw_deg : float
        End-tangent angles of the selected level.
    correction_p1_deg, correction_p2_deg : float
        Bias correction added to the raw angles (model-based or Richardson;
        0 when not applied).
    sigma_p1_deg, sigma_p2_deg : float
        Noise-only standard deviation estimate of each raw angle.
    apex_xy : tuple of float
        Apex node in image coordinates.
    rmse_px : float
        Residual RMS of the selected level.
    n_points : int
        Number of edge points fitted.
    levels : list of ArcSplineLevel
        Every refinement level, coarsest first.
    selected_level : int
        Index into ``levels`` of the reported model.
    init : str
        ``"physics"`` or ``"blind"``: how arc counts and joins were chosen.
    physics : dict or None
        Box-matched Young-Laplace prior per side (Bond number, angle, apex
        radius, residual), when the physics initialization was used.
    rejection_reasons : list of str
        Quality-gate failures; empty when the fit is usable.
    """

    p1_arcs: list[ArcSegment]
    p2_arcs: list[ArcSegment]
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
    selected_level: int
    init: str
    physics: dict[str, Any] | None
    rejection_reasons: list[str]
    _frame: _Frame = field(repr=False)

    @property
    def n_arcs(self) -> int:
        """Total number of arcs in the reported model."""
        return len(self.p1_arcs) + len(self.p2_arcs)

    @property
    def accepted(self) -> bool:
        """Whether the fit passed every quality gate."""
        return not self.rejection_reasons

    @property
    def contact_points(self) -> tuple[np.ndarray, np.ndarray]:
        """The contact points the spline ends on, image coordinates."""
        ends = self._frame.to_image(
            np.vstack([self.p1_arcs[-1].points(np.array([self.p1_arcs[-1].length])),
                       self.p2_arcs[-1].points(np.array([self.p2_arcs[-1].length]))])
        )
        return ends[0], ends[1]

    def sample(self, step_px: float = 1.0) -> np.ndarray:
        """Sample the spline from ``P1`` over the apex to ``P2``.

        Parameters
        ----------
        step_px : float, optional
            Approximate spacing between samples along the curve.

        Returns
        -------
        np.ndarray
            Image-coordinate points, shape ``(M, 2)``.
        """
        return self._frame.to_image(_sample_chains(self.p1_arcs, self.p2_arcs, step_px))

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for results and exports.

        Returns
        -------
        dict
            Arc counts, angles, corrections, residuals and the level history.
        """
        return {
            "segment": "arc",
            "n_arcs": self.n_arcs,
            "n_arcs_p1": len(self.p1_arcs),
            "n_arcs_p2": len(self.p2_arcs),
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
            "selected_level": self.selected_level,
            "levels": [
                {
                    "n_p1": lv.n_p1,
                    "n_p2": lv.n_p2,
                    "theta_p1_deg": lv.theta_p1_deg,
                    "theta_p2_deg": lv.theta_p2_deg,
                    "rmse_px": lv.rmse_px,
                    "bic": lv.bic,
                }
                for lv in self.levels
            ],
        }


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _arc_points(start: np.ndarray, heading: float, k: float, s: np.ndarray) -> np.ndarray:
    k = k if abs(k) > 1e-12 else 1e-12
    a = heading + k * s
    return np.column_stack(
        [
            start[0] + (np.sin(a) - np.sin(heading)) / k,
            start[1] + (np.cos(heading) - np.cos(a)) / k,
        ]
    )


def arc_through(start: np.ndarray, heading: float, end: np.ndarray) -> tuple[float, float, float]:
    """Return the arc leaving ``start`` with ``heading`` that passes through ``end``.

    Parameters
    ----------
    start, end : np.ndarray
        Arc endpoints.
    heading : float
        Tangent direction at ``start``, radians.

    Returns
    -------
    curvature : float
        Signed curvature in ``1/px``.
    length : float
        Arc length in pixels.
    end_heading : float
        Tangent direction at ``end``, radians.
    """
    c = np.asarray(end, float) - np.asarray(start, float)
    chord = float(np.hypot(c[0], c[1]))
    delta = 2.0 * (((np.arctan2(c[1], c[0]) - heading) + np.pi) % _TWO_PI - np.pi)
    half = 0.5 * delta
    length = chord if abs(half) < 1e-9 else chord * half / np.sin(half)
    return 2.0 * np.sin(half) / max(chord, 1e-12), max(length, 1e-12), heading + delta


def _sample_chains(p1_arcs: list[ArcSegment], p2_arcs: list[ArcSegment], step_px: float) -> np.ndarray:
    parts = []
    for arc in reversed(p1_arcs):  # P1 -> apex: walk the P1 chain backwards
        s = np.linspace(arc.length, 0.0, max(2, int(np.ceil(arc.length / step_px)) + 1))
        parts.append(arc.points(s)[:-1])
    for i, arc in enumerate(p2_arcs):
        s = np.linspace(0.0, arc.length, max(2, int(np.ceil(arc.length / step_px)) + 1))
        pts = arc.points(s)
        parts.append(pts if i == len(p2_arcs) - 1 else pts[:-1])
    return np.vstack(parts)


@dataclass(frozen=True)
class _Frame:
    """Fitting frame: ``P1`` at the origin, ``P2`` on +x, the drop at negative y."""

    origin: np.ndarray
    ex: np.ndarray
    ey: np.ndarray

    @classmethod
    def from_contacts(cls, p1: np.ndarray, p2: np.ndarray, points: np.ndarray) -> _Frame:
        p1 = np.asarray(p1, float)
        chord = np.asarray(p2, float) - p1
        length = float(np.hypot(*chord))
        if length <= 1e-9:
            raise ValueError("contact points coincide")
        ex = chord / length
        normal = np.array([-ex[1], ex[0]])
        heights = (np.asarray(points, float) - p1) @ normal
        # the apex is the vertex farthest from the chord; put the drop at negative y
        ey = normal if heights[int(np.argmax(np.abs(heights)))] < 0 else -normal
        return cls(p1, ex, ey)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        rel = np.asarray(pts, float).reshape(-1, 2) - self.origin
        return np.column_stack([rel @ self.ex, rel @ self.ey])

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        return self.origin + pts[:, :1] * self.ex + pts[:, 1:] * self.ey

    def vec_to_local(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        return np.array([v @ self.ex, v @ self.ey])


class _Edge:
    """Observed edge polyline, its arc length, and a smoothed node guide path.

    Nodes slide along a moving-average copy of the edge, indexed by the raw arc
    length: the raw polyline turns at every noisy vertex, which would make node
    derivatives jump; the normal offset absorbs the smoothing.
    """

    def __init__(self, q: np.ndarray):
        keep = np.concatenate([[True], np.any(np.diff(q, axis=0) != 0.0, axis=1)])
        q = q[keep]
        self.q = q
        steps = np.hypot(*np.diff(q, axis=0).T)
        self.seg = np.concatenate([[0.0], np.cumsum(steps)])
        self.total = float(self.seg[-1])
        w = max(3, len(q) // 60)
        kernel = np.full(2 * w + 1, 1.0 / (2 * w + 1))
        padded = np.pad(q, ((w, w), (0, 0)), mode="reflect", reflect_type="odd")
        guide = np.column_stack(
            [np.convolve(padded[:, 0], kernel, "valid"), np.convolve(padded[:, 1], kernel, "valid")]
        )
        d = np.gradient(guide, axis=0)
        self.guide = guide
        self.head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self._dq = np.diff(guide, axis=0) / steps[:, None]
        self._dhead = np.diff(self.head) / steps

    def point(self, t: np.ndarray, h: np.ndarray):
        """Return node positions and their derivatives along ``t`` and ``h``."""
        t = np.asarray(t, float)
        h = np.asarray(h, float)
        i = np.clip(np.searchsorted(self.seg, t, side="right") - 1, 0, len(self.seg) - 2)
        u = t - self.seg[i]
        base = self.guide[i] + u[:, None] * self._dq[i]
        a = self.head[i] + u * self._dhead[i]
        nu = np.column_stack([-np.sin(a), np.cos(a)])
        pos = base + h[:, None] * nu
        d_t = self._dq[i] + (h * self._dhead[i])[:, None] * np.column_stack([-np.cos(a), -np.sin(a)])
        return pos, d_t, nu


# -----------------------------------------------------------------------------
# Model: apex node + interior nodes per side, analytic residual Jacobian
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Model:
    """Parameter layout: ``[t_apex, h_apex, t_p1..., h_p1..., t_p2..., h_p2...]``.

    With ``slide_px > 0`` two more parameters follow: how far each contact
    point moves along the contact line (bounded by ``slide_px``).
    """

    n1: int  # interior nodes on the P1 side
    n2: int  # interior nodes on the P2 side
    slide_px: float = 0.0

    @property
    def size(self) -> int:
        return 2 + 2 * (self.n1 + self.n2) + (2 if self.slide_px > 0 else 0)

    def unpack(self, p: np.ndarray):
        o = 2
        t1, h1 = p[o : o + self.n1], p[o + self.n1 : o + 2 * self.n1]
        o += 2 * self.n1
        t2, h2 = p[o : o + self.n2], p[o + self.n2 : o + 2 * self.n2]
        return p[0], p[1], t1, h1, t2, h2

    def slides(self, p: np.ndarray) -> tuple[float, float]:
        return (float(p[-2]), float(p[-1])) if self.slide_px > 0 else (0.0, 0.0)

    def pack(self, ta, ha, t1, h1, t2, h2, slides=(0.0, 0.0)) -> np.ndarray:
        tail = list(slides) if self.slide_px > 0 else []
        return np.concatenate([[ta, ha], t1, h1, t2, h2, tail]).astype(float)


@dataclass
class _Chain:
    """Arrays of one side's arcs (apex outwards) and their parameter derivatives."""

    start: np.ndarray  # (A, 2)
    heading: np.ndarray  # (A,)
    curvature: np.ndarray  # (A,)
    length: np.ndarray  # (A,)
    d_start: np.ndarray  # (A, 2, P)
    d_heading: np.ndarray  # (A, P)
    d_curvature: np.ndarray  # (A, P)
    end_heading: float

    def segments(self) -> list[ArcSegment]:
        return [
            ArcSegment(self.start[j].copy(), float(self.heading[j]), float(self.curvature[j]), float(self.length[j]))
            for j in range(len(self.heading))
        ]


def _build_chain(nodes: np.ndarray, d_nodes: np.ndarray, heading0: float) -> _Chain:
    """Chain arcs through ``nodes`` starting with ``heading0`` (fixed at the apex)."""
    n_arcs = len(nodes) - 1
    n_par = d_nodes.shape[2]
    heading = np.empty(n_arcs)
    curvature = np.empty(n_arcs)
    length = np.empty(n_arcs)
    d_heading = np.empty((n_arcs, n_par))
    d_curvature = np.empty((n_arcs, n_par))
    psi = heading0
    d_psi = np.zeros(n_par)
    for j in range(n_arcs):
        c = nodes[j + 1] - nodes[j]
        dc = d_nodes[j + 1] - d_nodes[j]
        cc = max(float(c @ c), 1e-24)
        chord = np.sqrt(cc)
        d_alpha = (c[0] * dc[1] - c[1] * dc[0]) / cc
        delta = 2.0 * (((np.arctan2(c[1], c[0]) - psi) + np.pi) % _TWO_PI - np.pi)
        d_delta = 2.0 * (d_alpha - d_psi)
        half = 0.5 * delta
        sin_h, cos_h = np.sin(half), np.cos(half)
        heading[j] = psi
        d_heading[j] = d_psi
        curvature[j] = 2.0 * sin_h / chord
        d_curvature[j] = cos_h * d_delta / chord - 2.0 * sin_h * (c @ dc) / (chord * cc)
        length[j] = chord if abs(half) < 1e-9 else chord * half / sin_h
        psi = psi + delta
        d_psi = d_psi + d_delta
    return _Chain(nodes[:-1], heading, curvature, length, d_nodes[:-1], d_heading, d_curvature, psi)


class _Problem:
    """Residuals and analytic Jacobian of one model layout, sharing a cache."""

    def __init__(self, model: _Model, edge: _Edge, p1: np.ndarray, p2: np.ndarray):
        self.model = model
        self.edge = edge
        self.p1 = p1
        self.p2 = p2
        self._key: bytes | None = None
        self._cache: dict[str, Any] = {}

    def chains(self, p: np.ndarray) -> tuple[_Chain, _Chain]:
        m = self.model
        n_par = m.size
        ta, ha, t1, h1, t2, h2 = m.unpack(p)
        o1 = np.argsort(-t1)  # apex -> P1 walks towards decreasing edge position
        o2 = np.argsort(t2)
        t_all = np.concatenate([[ta], t1[o1], t2[o2]])
        h_all = np.concatenate([[ha], h1[o1], h2[o2]])
        pos, d_t, d_h = self.edge.point(t_all, h_all)
        idx_t = np.concatenate([[0], 2 + o1, 2 + 2 * m.n1 + o2])
        idx_h = np.concatenate([[1], 2 + m.n1 + o1, 2 + 2 * m.n1 + m.n2 + o2])
        d_pos = np.zeros((len(t_all), 2, n_par))
        rows = np.arange(len(t_all))
        d_pos[rows, :, idx_t] = d_t
        d_pos[rows, :, idx_h] = d_h
        # contact ends: fixed, or sliding along the contact line (local x axis)
        s1, s2 = m.slides(p)
        end1 = np.zeros((1, 2, n_par))
        end2 = np.zeros((1, 2, n_par))
        if m.slide_px > 0:
            end1[0, 0, n_par - 2] = 1.0
            end2[0, 0, n_par - 1] = 1.0
        side1 = np.concatenate([[0], 1 + np.arange(m.n1)])
        side2 = np.concatenate([[0], 1 + m.n1 + np.arange(m.n2)])
        c1 = _build_chain(
            np.vstack([pos[side1], self.p1 + [s1, 0.0]]), np.concatenate([d_pos[side1], end1]), np.pi
        )
        c2 = _build_chain(
            np.vstack([pos[side2], self.p2 + [s2, 0.0]]), np.concatenate([d_pos[side2], end2]), 0.0
        )
        return c1, c2

    def owner(self, p: np.ndarray) -> np.ndarray:
        """Arc index (P1 arcs first, apex outwards) owning each edge point."""
        ta, _, t1, _, t2, _ = self.model.unpack(p)
        seg = self.edge.seg
        on1 = seg < ta
        own = np.empty(len(seg), dtype=int)
        own[on1] = self.model.n1 - np.searchsorted(np.sort(t1), seg[on1], side="right")
        own[~on1] = self.model.n1 + 1 + np.searchsorted(np.sort(t2), seg[~on1], side="right")
        return own

    def _evaluate(self, p: np.ndarray) -> dict[str, Any]:
        key = p.tobytes()
        if key == self._key:
            return self._cache
        c1, c2 = self.chains(p)
        own = self.owner(p)
        start = np.vstack([c1.start, c2.start])[own]
        psi = np.concatenate([c1.heading, c2.heading])[own]
        k = np.concatenate([c1.curvature, c2.curvature])[own]
        w = self.edge.q - start
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        n = np.column_stack([-sin_p, cos_p])
        ww = np.einsum("ij,ij->i", w, w)
        a = k * ww - 2.0 * np.einsum("ij,ij->i", w, n)
        # 1 + k*a == |n - k w|^2: the signed circle distance a/(1+S) is stable at k -> 0
        s = np.hypot(n[:, 0] - k * w[:, 0], n[:, 1] - k * w[:, 1])
        r = a / (1.0 + s)
        self._key = key
        self._cache = {
            "c1": c1, "c2": c2, "own": own, "w": w, "n": n, "k": k, "a": a, "s": s,
            "ww": ww, "cos": cos_p, "sin": sin_p, "r": r,
        }
        return self._cache

    def residuals(self, p: np.ndarray) -> np.ndarray:
        return self._evaluate(p)["r"]

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        e = self._evaluate(p)
        c1, c2, own = e["c1"], e["c2"], e["own"]
        d_start = np.concatenate([c1.d_start, c2.d_start])[own]  # (M, 2, P)
        d_psi = np.concatenate([c1.d_heading, c2.d_heading])[own]  # (M, P)
        d_k = np.concatenate([c1.d_curvature, c2.d_curvature])[own]  # (M, P)
        w, n, k, a, s = e["w"], e["n"], e["k"], e["a"], e["s"]
        s = np.maximum(s, 1e-12)
        one_s = 1.0 + s
        f_a = 1.0 / one_s - a * k / (2.0 * s * one_s**2)
        f_k = -(a**2) / (2.0 * s * one_s**2)
        grad_w = 2.0 * k[:, None] * w - 2.0 * n  # dA/dw; dw = -d_start
        term_w = -np.einsum("md,mdp->mp", grad_w, d_start)
        # dA/dn = -2w, dn/dpsi = (-cos, -sin)
        term_n = 2.0 * (w[:, 0] * e["cos"] + w[:, 1] * e["sin"])[:, None] * d_psi
        term_k = e["ww"][:, None] * d_k
        return f_a[:, None] * (term_w + term_n + term_k) + f_k[:, None] * d_k


def _bounds(p, model: _Model, total: float, max_offset: float):
    ta, _, t1, _, t2, _ = model.unpack(p)
    ts = np.concatenate([[0.0], np.sort(t1), [ta], np.sort(t2), [total]])
    lo_all = ts[1:-1] - 0.4 * (ts[1:-1] - ts[:-2])
    hi_all = ts[1:-1] + 0.4 * (ts[2:] - ts[1:-1])
    lo = np.full_like(p, -max_offset)
    hi = np.full_like(p, max_offset)
    lo[0], hi[0] = lo_all[model.n1], hi_all[model.n1]
    rank1 = np.argsort(np.argsort(t1))
    lo[2 : 2 + model.n1] = lo_all[rank1]
    hi[2 : 2 + model.n1] = hi_all[rank1]
    o = 2 + 2 * model.n1
    rank2 = np.argsort(np.argsort(t2))
    lo[o : o + model.n2] = lo_all[model.n1 + 1 + rank2]
    hi[o : o + model.n2] = hi_all[model.n1 + 1 + rank2]
    if model.slide_px > 0:
        lo[-2:], hi[-2:] = -model.slide_px, model.slide_px
    return lo, hi


def _refit(p, problem: _Problem, max_offset: float, loss: str, max_nfev: int) -> np.ndarray:
    lo, hi = _bounds(p, problem.model, problem.edge.total, max_offset)
    p0 = np.clip(p, lo + 1e-9, hi - 1e-9)
    res = least_squares(
        problem.residuals,
        p0,
        jac=problem.jacobian,
        bounds=(lo, hi),
        x_scale="jac",
        max_nfev=max_nfev,
        loss=loss,
        f_scale=1.0,
        # node positions are in pixels: 1e-6 relative is far below edge accuracy
        ftol=1e-7,
        xtol=1e-6,
        gtol=1e-7,
    )
    return res.x


def _end_angle(end_heading: float, flank_frame: tuple[np.ndarray, np.ndarray]) -> float:
    """Contact angle between the flank and the substrate, degrees.

    ``flank_frame`` holds the substrate direction pointing into the drop and the
    normal pointing to the apex side, both in the fitting frame.
    """
    flank = -np.array([np.cos(end_heading), np.sin(end_heading)])  # up the flank
    inward, normal = flank_frame
    return float(np.degrees(np.arctan2(flank @ normal, flank @ inward)))


def _angle_sigma_deg(length: float, rmse: float, n_points: int) -> float:
    """Noise-only standard deviation of an end tangent supported by one arc."""
    if n_points < 3 or length <= 0:
        return float("nan")
    # slope of a least-squares line evaluated at an end of its support
    return float(np.degrees(2.0 * rmse * np.sqrt(12.0 / n_points) / length))


def _flank_frames(frame: _Frame, tangents) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Inward substrate direction and apex-side normal at each contact, local frame."""
    out = []
    for i, inward_default in enumerate((np.array([1.0, 0.0]), np.array([-1.0, 0.0]))):
        if tangents is None:
            inward = inward_default
        else:
            t = frame.vec_to_local(tangents[i])
            t = t / (np.hypot(*t) or 1.0)
            inward = t if t @ inward_default >= 0 else -t
        normal = np.array([-inward[1], inward[0]])
        if normal[1] > 0:  # the drop (apex side) is at negative y
            normal = -normal
        out.append((inward, normal))
    return tuple(out)


def _richardson(levels, sel: int, side: str, sigma: float, mode: str, z: float) -> float:
    """Richardson correction for one side, assuming a ``1/N**2`` bias."""
    if mode == "off":
        return 0.0

    def count(lv):
        return lv.n_p1 if side == "p1" else lv.n_p2

    def angle(lv):
        return lv.theta_p1_deg if side == "p1" else lv.theta_p2_deg

    n = count(levels[sel])
    half = int(round(n / 2))
    if n < 2 or half < 1 or half >= n:
        return 0.0
    coarse = [lv for lv in levels[: sel + 1] if count(lv) == half]
    if not coarse:
        return 0.0
    delta = angle(levels[sel]) - angle(coarse[-1])
    if mode == "auto" and not (np.isfinite(sigma) and abs(delta) > z * sigma):
        return 0.0
    return float(delta * half**2 / (n**2 - half**2))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def extract_interface(contour: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return the ordered edge points from ``P1`` over the apex to ``P2``.

    The closed silhouette is walked both ways between the vertices nearest to the
    contact points; the branch that rises farther from the contact chord is the
    liquid-fluid interface, the other runs along the substrate or reflection.

    Parameters
    ----------
    contour : np.ndarray
        Ordered closed (or open) contour, shape ``(N, 2)``.
    p1, p2 : np.ndarray
        Contact points.

    Returns
    -------
    np.ndarray
        Interface points ordered from ``P1`` to ``P2``, shape ``(M, 2)``.

    Raises
    ------
    ValueError
        If the contour is too short or both contacts map to one vertex.
    """
    pts = np.asarray(contour, float).reshape(-1, 2)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n = len(pts)
    if n < 5:
        raise ValueError("contour has too few points")
    i1 = int(np.argmin(np.hypot(*(pts - p1).T)))
    i2 = int(np.argmin(np.hypot(*(pts - p2).T)))
    if i1 == i2:
        raise ValueError("contact points map to the same contour vertex")
    chord = p2 - p1
    normal = np.array([-chord[1], chord[0]]) / (np.hypot(*chord) or 1.0)

    def branch(step: int) -> np.ndarray:
        count = (i2 - i1) % n if step == 1 else (i1 - i2) % n
        return pts[(i1 + step * np.arange(count + 1)) % n]

    candidates = [branch(1), branch(-1)]
    rise = [float(np.max(np.abs((c - p1) @ normal))) for c in candidates]
    arc = candidates[int(np.argmax(rise))]
    # anchor the ends on the contact points themselves
    return np.vstack([p1, arc[1:-1], p2])


def fit_arc_spline(
    points: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    max_arcs: int = 12,
    patience: int = 2,
    richardson: RichardsonMode = "auto",
    richardson_z: float = 2.0,
    substrate_tangents: tuple[np.ndarray, np.ndarray] | None = None,
    max_node_offset_px: float = 8.0,
    noise_floor_px: float = 0.02,
    loss: str = "soft_l1",
    max_nfev: int = 100,
    init: InitMode = "physics",
    physics_tol_px: float = 0.25,
    physics_noise_fraction: float = 2.0,
    physics_adequacy: float = 1.5,
    bias_correction: BiasMode = "model",
    max_rmse_px: float = 1.5,
    contact_slide_px: float = 0.0,
) -> ArcSplineFit:
    """Fit the apex-anchored G1 arc spline to ordered interface points.

    Parameters
    ----------
    points : np.ndarray
        Interface points ordered from ``P1`` to ``P2``, shape ``(M, 2)``, image
        coordinates (see :func:`extract_interface`).
    p1, p2 : np.ndarray
        Contact points; the spline passes through them exactly.
    max_arcs : int, optional
        Maximum total number of arcs (both sides).
    patience : int, optional
        Blind mode only: stop refining after this many levels without a BIC
        improvement.
    richardson : {"auto", "on", "off"}, optional
        Apply the ``1/N**2`` bias extrapolation always, never, or only when the
        angle change between levels exceeds ``richardson_z`` noise sigmas.
    richardson_z : float, optional
        Significance threshold of the ``"auto"`` mode.
    substrate_tangents : tuple of np.ndarray, optional
        Substrate direction vectors at ``P1`` and ``P2`` in image coordinates,
        for curved substrates. Defaults to the contact chord.
    max_node_offset_px : float, optional
        How far a node may leave the observed edge along its normal.
    noise_floor_px : float, optional
        Residual RMS below which further refinement is not rewarded by the BIC;
        keeps noise-free input from refining to ``max_arcs``.
    loss : str, optional
        Robust loss passed to :func:`scipy.optimize.least_squares`.
    max_nfev : int, optional
        Function-evaluation budget per refinement level.
    init : {"physics", "blind"}, optional
        ``"physics"`` matches a Young-Laplace profile to each side's box
        (golden-section Bond search, :mod:`menipy.math.sessile_box`) and takes
        the arc counts and join positions from its curvature; it falls back to
        ``"blind"`` -- coarse-to-fine splitting with BIC selection -- when the
        box is degenerate.
    physics_tol_px : float, optional
        Minimum target deviation used to size the physics arc counts. Image
        edges carry 0.1-0.2 px of low-frequency localization error that looks
        like shape, not noise; arcs finer than that only follow it.
    physics_noise_fraction : float, optional
        The target deviation is at least this multiple of the edge noise
        (:func:`estimate_edge_noise`).
    physics_adequacy : float, optional
        The physics-sized fit is kept only if its residual RMS stays below this
        multiple of the larger of the edge noise and half the arc tolerance
        (plus 0.05 px); otherwise the drop departs from Young-Laplace and the
        blind search is used instead.
    bias_correction : {"model", "richardson", "off"}, optional
        End-tangent bias removal. ``"model"`` fits the same arc layout to the
        noise-free matched Young-Laplace profile and subtracts the arcs' angle
        error there (physics initialization only; otherwise Richardson).
        ``"richardson"`` extrapolates between the half and full levels as set by
        ``richardson``.
    max_rmse_px : float, optional
        Quality gate: residual RMS above the larger of this and four times the
        edge noise marks the fit as rejected (see ``rejection_reasons``).
    contact_slide_px : float, optional
        When positive, each contact point becomes a least-squares parameter
        that slides along the contact line by at most this many pixels, so the
        edge data -- not the contact detector -- decide where the interface
        meets the substrate. ``0`` keeps the given contact points fixed.

    Returns
    -------
    ArcSplineFit
        Fitted model, contact angles and diagnostics.

    Raises
    ------
    ValueError
        If there are too few points or the contact points coincide.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) < 10:
        raise ValueError("arc spline needs at least 10 interface points")
    frame = _Frame.from_contacts(p1, p2, pts)
    edge = _Edge(frame.to_local(pts))
    lp1 = frame.to_local(np.asarray(p1, float))[0]
    lp2 = frame.to_local(np.asarray(p2, float))[0]
    flank = _flank_frames(frame, substrate_tangents)
    m = len(edge.q)

    levels: list[ArcSplineLevel] = []
    states: list[tuple[_Model, np.ndarray]] = []

    def record(model: _Model, params: np.ndarray) -> dict[str, Any]:
        e = _Problem(model, edge, lp1, lp2)._evaluate(params)
        rss = float(e["r"] @ e["r"])
        bic = m * np.log(rss / m + noise_floor_px**2) + model.size * np.log(m)
        levels.append(
            ArcSplineLevel(
                n_p1=model.n1 + 1,
                n_p2=model.n2 + 1,
                theta_p1_deg=_end_angle(e["c1"].end_heading, flank[0]),
                theta_p2_deg=_end_angle(e["c2"].end_heading, flank[1]),
                rmse_px=float(np.sqrt(rss / m)),
                bic=float(bic),
            )
        )
        states.append((model, params.copy()))
        return e

    noise = estimate_edge_noise(edge.q)
    layout = None
    prior = None  # physics prior, kept for diagnostics even when rejected
    if init == "physics":
        prior = layout = _physics_layout(
            edge, lp2, max_arcs, physics_tol_px, physics_noise_fraction, noise
        )
    if layout is not None:
        # physics prior: arc counts and join positions from the box-matched
        # Young-Laplace profile; the Richardson mode also fits the half level
        counts = [layout["full"]]
        if bias_correction == "richardson" and layout["half"] != layout["full"]:
            counts.insert(0, layout["half"])
        for c in counts:
            record(*_fit_physics_level(
                edge, lp1, lp2, layout["i_apex"], layout, c, max_node_offset_px, loss, max_nfev,
                contact_slide_px,
            ))
        sel = len(levels) - 1
        # adequacy: arcs sized from the physics profile must explain the edge
        # down to its noise; a drop that departs from Young-Laplace (pinning,
        # asymmetry) leaves a larger residual and is handed to the blind search
        expected = max(noise, 0.5 * layout["tol"], noise_floor_px)
        if levels[-1].rmse_px > physics_adequacy * expected + 0.05:
            layout = None
            levels.clear()
            states.clear()
    if layout is None:
        # blind coarse-to-fine: split the worst arc of each side per level
        model = _Model(0, 0, contact_slide_px)
        empty = np.empty(0)
        params = model.pack(float(edge.seg[int(np.argmin(edge.q[:, 1]))]), 0.0, empty, empty, empty, empty)
        params = _refit(params, _Problem(model, edge, lp1, lp2), max_node_offset_px, loss, max_nfev)
        best_bic = np.inf
        since_best = 0
        while True:
            e = record(model, params)
            if levels[-1].bic < best_bic - 1e-9:
                best_bic, since_best = levels[-1].bic, 0
            else:
                since_best += 1
            n_total = model.n1 + model.n2 + 2
            if n_total + 2 > max_arcs or since_best >= patience:
                break
            r, own = e["r"], e["own"]
            score = np.bincount(own, weights=r**2, minlength=n_total)
            ta, ha, t1, h1, t2, h2 = model.unpack(params)
            a1 = int(np.argmax(score[: model.n1 + 1]))
            b1 = np.concatenate([[ta], np.sort(t1)[::-1], [0.0]])
            t1 = np.append(t1, 0.5 * (b1[a1] + b1[a1 + 1]))
            h1 = np.append(h1, 0.0)
            a2 = int(np.argmax(score[model.n1 + 1 :]))
            b2 = np.concatenate([[ta], np.sort(t2), [edge.total]])
            t2 = np.append(t2, 0.5 * (b2[a2] + b2[a2 + 1]))
            h2 = np.append(h2, 0.0)
            slides = model.slides(params)
            model = _Model(model.n1 + 1, model.n2 + 1, contact_slide_px)
            params = _refit(
                model.pack(ta, ha, t1, h1, t2, h2, slides), _Problem(model, edge, lp1, lp2),
                max_node_offset_px, loss, max_nfev,
            )
        sel = int(np.argmin([lv.bic for lv in levels]))

    sel_model, sel_params = states[sel]
    e = _Problem(sel_model, edge, lp1, lp2)._evaluate(sel_params)
    c1, c2, own = e["c1"], e["c2"], e["own"]
    level = levels[sel]
    n_arcs1 = len(c1.heading)
    last1 = int(np.sum(own == n_arcs1 - 1))
    last2 = int(np.sum(own == n_arcs1 + len(c2.heading) - 1))
    sig1 = _angle_sigma_deg(float(c1.length[-1]), level.rmse_px, last1)
    sig2 = _angle_sigma_deg(float(c2.length[-1]), level.rmse_px, last2)
    if bias_correction == "off":
        corr1 = corr2 = 0.0
    elif bias_correction == "model" and layout is not None:
        b1, b2 = _model_bias_deg(
            layout, lp1, lp2, (level.n_p1, level.n_p2), flank, max_node_offset_px, max_nfev
        )
        corr1, corr2 = -b1, -b2
    else:
        corr1 = _richardson(levels, sel, "p1", sig1, richardson, richardson_z)
        corr2 = _richardson(levels, sel, "p2", sig2, richardson, richardson_z)
    apex_local = c1.start[0]
    physics = None
    if prior is not None:
        physics = {
            side: {
                "bond": sh.bond,
                "theta_deg": sh.theta_deg,
                "apex_radius_px": sh.apex_radius_px,
                "rms_px": sh.rms_px,
                "evaluations": sh.evaluations,
            }
            for side, sh in zip(("p1", "p2"), prior["shapes"])
        }
        physics["adequate"] = layout is not None
        physics["edge_noise_px"] = noise
    theta1, theta2 = level.theta_p1_deg + corr1, level.theta_p2_deg + corr2
    reasons = _quality_gate(
        (theta1, theta2), c1.segments(), c2.segments(), level.rmse_px, noise, max_rmse_px
    )
    return ArcSplineFit(
        p1_arcs=c1.segments(),
        p2_arcs=c2.segments(),
        theta_p1_deg=level.theta_p1_deg + corr1,
        theta_p2_deg=level.theta_p2_deg + corr2,
        theta_p1_raw_deg=level.theta_p1_deg,
        theta_p2_raw_deg=level.theta_p2_deg,
        correction_p1_deg=corr1,
        correction_p2_deg=corr2,
        sigma_p1_deg=sig1,
        sigma_p2_deg=sig2,
        apex_xy=tuple(float(v) for v in frame.to_image(apex_local)[0]),
        rmse_px=level.rmse_px,
        n_points=m,
        levels=levels,
        selected_level=sel,
        init="physics" if layout is not None else "blind",
        physics=physics,
        rejection_reasons=reasons,
        _frame=frame,
    )


def _quality_gate(
    thetas: tuple[float, float],
    p1_arcs: list[ArcSegment],
    p2_arcs: list[ArcSegment],
    rmse: float,
    noise: float,
    max_rmse_px: float,
) -> list[str]:
    """Reasons to distrust a fit: impossible angles, a model below the contact
    chord, or a residual far above the edge noise (wrong contacts, needle or
    reflection edges in the interface)."""
    reasons = []
    for side, theta in zip(("p1", "p2"), thetas):
        if not (np.isfinite(theta) and 0.0 < theta < 180.0):
            reasons.append(f"{side}_angle_out_of_range")
    model = _sample_chains(p1_arcs, p2_arcs, 2.0)
    # fitting frame: the contact chord is y = 0 and the drop lies at negative y
    if float(np.max(model[:, 1])) > max(2.0, 4.0 * noise):
        reasons.append("model_below_contact_line")
    if rmse > max(max_rmse_px, 4.0 * noise):
        reasons.append("residual_above_edge_noise")
    return reasons


def _apex_vertex(edge: _Edge, width: float) -> tuple[int, float, float]:
    """Apex: where the edge tangent is parallel to the contact chord.

    The single highest point of a noisy or pixel-staircase edge is biased upward
    and, on a flat top, wanders sideways by many pixels; both skew the per-side
    boxes. A cubic ``y(x)`` fitted to the raw points of a wide window around the
    top (30 % of the smaller half-width) averages the noise and the staircase
    runs, its cubic term absorbs an asymmetric top, and the root of its
    derivative is the horizontal-tangent point. The window is re-centred twice.

    Parameters
    ----------
    edge : _Edge
        The edge in the fitting frame.
    width : float
        Contact chord length, used to size the height window.

    Returns
    -------
    index : int
        Edge vertex nearest to the apex (for splitting the sides).
    x, y : float
        Apex position in the fitting frame (drop at negative y).
    """
    g, q = edge.guide, edge.q
    i_top = int(np.argmin(g[:, 1]))
    xa, ya = float(g[i_top, 0]), float(g[i_top, 1])
    top = q[:, 1] < 0.5 * ya  # the upper half of the drop, not the flanks
    for _ in range(3):
        half = max(8.0, 0.3 * min(xa, width - xa))
        near = top & (np.abs(q[:, 0] - xa) <= half)
        if near.sum() < 6:
            break
        u = q[near, 0] - xa
        c3, c2, c1, c0 = np.polyfit(u, q[near, 1], 3)
        roots = np.roots([3.0 * c3, 2.0 * c2, c1]) if abs(c3) > 1e-15 else np.array([-c1 / (2.0 * c2)])
        roots = roots[np.isreal(roots)].real
        roots = roots[np.abs(roots) <= half]
        if roots.size == 0:
            break
        r = float(roots[np.argmin(np.abs(roots))])
        xa, ya = xa + r, float(np.polyval([c3, c2, c1, c0], r))
    i_near = int(np.argmin(np.where(top, np.abs(q[:, 0] - xa), np.inf)))
    return i_near, xa, ya


def estimate_edge_noise(points: np.ndarray) -> float:
    """Robust standard deviation of the edge-point scatter normal to the edge.

    Second differences of consecutive points cancel the smooth shape (their
    curvature term is ``κ Δs²``, negligible at pixel spacing) and leave
    ``Var = 6 σ²`` for independent normal noise; the MAD makes the estimate
    insensitive to outliers. It does not depend on any shape model.

    Parameters
    ----------
    points : np.ndarray
        Ordered edge points, shape ``(M, 2)``.

    Returns
    -------
    float
        Estimated noise standard deviation in pixels.
    """
    q = np.asarray(points, float).reshape(-1, 2)
    if len(q) < 5:
        return 0.0
    t = q[2:] - q[:-2]
    t /= np.maximum(np.hypot(*t.T), 1e-12)[:, None]
    d2 = q[:-2] - 2.0 * q[1:-1] + q[2:]
    normal_d2 = d2[:, 0] * -t[:, 1] + d2[:, 1] * t[:, 0]
    mad = float(np.median(np.abs(normal_d2 - np.median(normal_d2))))
    return mad / 0.6745 / np.sqrt(6.0)


def _physics_layout(
    edge: _Edge, lp2: np.ndarray, max_arcs: int, tol_min: float, tol_fraction: float, noise: float
) -> dict[str, Any] | None:
    """Arc counts and join positions from box-matched Young-Laplace profiles.

    Returns ``None`` when the box is degenerate (apex outside the contacts, no
    matching profile), and the caller falls back to the blind search.
    """
    from menipy.math.sessile_box import fit_side_box

    i_apex, xa, ya = _apex_vertex(edge, float(lp2[0]))
    ta = float(edge.seg[i_apex])
    height = -float(ya)
    widths = (float(xa), float(lp2[0] - xa))
    if height <= 1.0 or min(widths) <= 1.0 or i_apex in (0, len(edge.q) - 1):
        return None
    sides = (edge.q[: i_apex + 1][::-1], edge.q[i_apex:])  # apex outwards
    signs = (-1.0, 1.0)
    try:
        shapes = [
            fit_side_box(np.column_stack([sg * (q[:, 0] - xa), q[:, 1] - ya]), w, height)
            for q, sg, w in zip(sides, signs, widths)
        ]
    except ValueError:
        return None
    tol = max(tol_min, tol_fraction * noise)
    cap = max(1, max_arcs // 2)
    full = tuple(min(cap, sh.arcs_needed(tol)) for sh in shapes)
    half = tuple(max(1, int(round(n / 2))) for n in full)
    return {
        "i_apex": i_apex, "t_apex": ta, "apex": (float(xa), float(ya)),
        "full": full, "half": half, "shapes": shapes, "tol": tol,
    }


def _join_positions(edge: _Edge, i_apex: int, layout: dict[str, Any], side: int, n_arcs: int) -> np.ndarray:
    """Edge positions of one side's joins, apex outwards, from the physics profile."""
    if n_arcs <= 1:
        return np.empty(0)
    xa, ya = layout["apex"]
    sign = -1.0 if side == 1 else 1.0
    sh = layout["shapes"][side - 1]
    s_join = sh.join_positions(n_arcs)
    target = np.column_stack(
        [xa + sign * np.interp(s_join, sh.s_px, sh.x_px), ya + np.interp(s_join, sh.s_px, sh.z_px)]
    )
    if side == 1:
        q, seg = edge.q[: i_apex + 1][::-1], edge.seg[: i_apex + 1][::-1]
    else:
        q, seg = edge.q[i_apex:], edge.seg[i_apex:]
    idx = np.argmin(((q[None, 1:-1, :] - target[:, None, :]) ** 2).sum(-1), axis=1) + 1
    t = seg[idx].astype(float)
    # keep joins strictly ordered away from the apex
    step = 1e-3 * edge.total
    for k in range(1, len(t)):
        if side == 1 and t[k] >= t[k - 1] - step:
            t[k] = t[k - 1] - step
        if side == 2 and t[k] <= t[k - 1] + step:
            t[k] = t[k - 1] + step
    return t


def _fit_physics_level(
    edge: _Edge, lp1, lp2, i_apex: int, layout: dict[str, Any], counts: tuple[int, int],
    max_offset: float, loss: str, max_nfev: int, slide_px: float = 0.0,
) -> tuple[_Model, np.ndarray]:
    """Fit one level whose joins start at the physics-placed positions."""
    n1, n2 = counts
    t1 = _join_positions(edge, i_apex, layout, 1, n1)
    t2 = _join_positions(edge, i_apex, layout, 2, n2)
    model = _Model(n1 - 1, n2 - 1, slide_px)
    params = model.pack(float(edge.seg[i_apex]), 0.0, t1, np.zeros(n1 - 1), t2, np.zeros(n2 - 1))
    return model, _refit(params, _Problem(model, edge, lp1, lp2), max_offset, loss, max_nfev)


def _model_bias_deg(
    layout: dict[str, Any], lp1, lp2, counts: tuple[int, int], flank, max_offset: float,
    max_nfev: int, step_px: float = 3.0,
) -> tuple[float, float]:
    """End-tangent bias of the arc layout, measured on the matched physics profile.

    The same arc counts and physics-placed joins are fitted to the noise-free
    box-matched Young-Laplace interface; the difference between the arcs' end
    angles and the profile's own contact angles is the discretization bias,
    free of the image noise that a Richardson extrapolation would amplify.
    """
    xa, ya = layout["apex"]
    sides = []
    for sh, sign in zip(layout["shapes"], (-1.0, 1.0)):
        s = np.append(np.arange(0.0, sh.s_px[-1], step_px), sh.s_px[-1])
        sides.append(np.column_stack([xa + sign * np.interp(s, sh.s_px, sh.x_px), ya + np.interp(s, sh.s_px, sh.z_px)]))
    q = np.vstack([sides[0][::-1], sides[1][1:]])
    medge = _Edge(q)
    i_apex = len(sides[0]) - 1
    model, params = _fit_physics_level(medge, lp1, lp2, i_apex, layout, counts, max_offset, "linear", max_nfev)
    e = _Problem(model, medge, lp1, lp2)._evaluate(params)
    exact = (np.pi - np.radians(layout["shapes"][0].theta_deg), np.radians(layout["shapes"][1].theta_deg))
    return (
        _end_angle(e["c1"].end_heading, flank[0]) - _end_angle(exact[0], flank[0]),
        _end_angle(e["c2"].end_heading, flank[1]) - _end_angle(exact[1], flank[1]),
    )


def fit_sessile_arc_spline(
    contour: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    substrate_tangents: tuple[np.ndarray, np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[ArcSplineFit, np.ndarray]:
    """Extract the interface from a closed silhouette and fit the arc spline.

    Parameters
    ----------
    contour : np.ndarray
        Ordered drop silhouette, shape ``(N, 2)``.
    p1, p2 : np.ndarray
        Contact points.
    substrate_tangents : tuple of np.ndarray, optional
        Substrate direction at each contact point (curved substrates).
    **kwargs
        Forwarded to :func:`fit_arc_spline`.

    Returns
    -------
    fit : ArcSplineFit
        The fitted model.
    interface : np.ndarray
        The interface points that were fitted.
    """
    interface = extract_interface(contour, p1, p2)
    return (
        fit_arc_spline(interface, p1, p2, substrate_tangents=substrate_tangents, **kwargs),
        interface,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        import cv2

        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img.astype(np.float32)


def sample_edge_along_normals(
    gray: np.ndarray,
    curve: np.ndarray,
    *,
    search_px: float = 4.0,
    sample_step_px: float = 0.25,
    blur_sigma: float = 1.0,
    interior_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the steepest intensity change along the normals of a curve.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale (or BGR) image.
    curve : np.ndarray
        Ordered curve points in image coordinates, shape ``(M, 2)``.
    search_px : float, optional
        Half-length of each normal profile.
    sample_step_px : float, optional
        Spacing of the profile samples.
    blur_sigma : float, optional
        Gaussian pre-smoothing; 0 disables it.
    interior_point : np.ndarray, optional
        A point inside the drop, used to orient the normals outward so that one
        edge polarity is enforced; without it the largest gradient magnitude wins.

    Returns
    -------
    points : np.ndarray
        Sub-pixel edge location on each normal, shape ``(M, 2)``.
    strength : np.ndarray
        Directional gradient at that location (polarity-signed), shape ``(M,)``.
    """
    import cv2

    img = _to_gray(gray)
    if blur_sigma > 0:
        k = int(2 * np.ceil(3 * blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    pts = np.asarray(curve, float).reshape(-1, 2)
    tangent = np.gradient(pts, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    if interior_point is not None:
        outward = np.einsum("ij,ij->i", normal, pts - np.asarray(interior_point, float))
        if np.median(outward) < 0:
            normal = -normal
    u = np.arange(-search_px, search_px + 0.5 * sample_step_px, sample_step_px)
    gx = (pts[:, 0, None] + normal[:, 0, None] * u[None, :]).astype(np.float32)
    gy = (pts[:, 1, None] + normal[:, 1, None] * u[None, :]).astype(np.float32)
    prof = cv2.remap(img, gx, gy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.gradient(prof.astype(float), sample_step_px, axis=1)
    if interior_point is not None:
        peak_abs = d[np.arange(len(d)), np.argmax(np.abs(d), axis=1)]
        d = d * (1.0 if np.median(peak_abs) >= 0 else -1.0)
    else:
        d = np.abs(d)
    k = np.argmax(d, axis=1)
    rows = np.arange(len(d))
    km = np.clip(k - 1, 0, len(u) - 1)
    kp = np.clip(k + 1, 0, len(u) - 1)
    y0, ym, yp = d[rows, k], d[rows, km], d[rows, kp]
    denom = ym - 2.0 * y0 + yp
    inner = (k > 0) & (k < len(u) - 1) & (denom < -1e-12)
    delta = np.where(inner, 0.5 * (ym - yp) / np.where(inner, denom, 1.0), 0.0)
    offset = u[k] + np.clip(delta, -1.0, 1.0) * sample_step_px
    # a peak on the window boundary is no edge found, just the steepest slope seen
    y0 = np.where((k == 0) | (k == len(u) - 1), 0.0, y0)
    return pts + offset[:, None] * normal, y0


def refine_on_image(
    image: np.ndarray,
    fit: ArcSplineFit,
    interface: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    iterations: int = 2,
    search_px: float = 4.0,
    step_px: float = 1.0,
    baseline_margin_px: float = 3.0,
    min_relative_contrast: float = 0.3,
    keep_margin_points: bool = True,
    refine_contacts: bool = True,
    min_contact_shift_px: float = 2.0,
    max_contact_shift_px: float = 8.0,
    fitter: Any = None,
    **fit_kwargs: Any,
) -> tuple[ArcSplineFit, np.ndarray]:
    """Move the edge points to the steepest contrast along the spline normals.

    Each iteration samples the current spline, finds the sub-pixel gradient
    peak of the drop's edge polarity on every normal, and refits the arc spline
    to those points. Near the contact chord the normals cross the substrate or
    its reflection, so points within ``baseline_margin_px`` of it keep their
    original positions; weak-contrast samples are dropped.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image the contour came from.
    fit : ArcSplineFit
        Initial fit, e.g. from :func:`fit_sessile_arc_spline`.
    interface : np.ndarray
        The interface points of the initial fit (P1 to P2).
    p1, p2 : np.ndarray
        Contact points.
    iterations : int, optional
        Sample-and-refit rounds.
    search_px : float, optional
        Half-length of the normal profiles.
    step_px : float, optional
        Spacing of the samples along the spline.
    baseline_margin_px : float, optional
        Band above the contact chord that keeps the original points.
    min_relative_contrast : float, optional
        Samples weaker than this fraction of the median peak are discarded.
    keep_margin_points : bool, optional
        Keep the original points inside the baseline band (otherwise only the
        contact points anchor the ends there).
    refine_contacts : bool, optional
        Slide each contact point along the contact line to where the refined
        edge, extrapolated from just above the band, meets it. A segmentation
        that sits inside the drop puts the contacts inside too, and the arcs,
        forced through them, would bend sharply at the ends.
    min_contact_shift_px : float, optional
        Smaller moves are treated as edge-localization noise and ignored: the
        extrapolated crossing is less precise than a contact point that is
        already on the edge.
    max_contact_shift_px : float, optional
        Largest accepted contact-point move.
    fitter : callable, optional
        ``fitter(points, p1, p2, **fit_kwargs)`` used for the refits, returning
        an object with ``sample`` and ``apex_xy`` (default
        :func:`fit_arc_spline`; e.g. ``clothoid_spline.fit_clothoid_spline``).
    **fit_kwargs
        Forwarded to :func:`fit_arc_spline`.

    Returns
    -------
    fit : ArcSplineFit
        The refined fit.
    points : np.ndarray
        The image-derived edge points it was fitted to.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    chord = p2 - p1
    length = float(np.hypot(*chord)) or 1.0
    normal = np.array([-chord[1], chord[0]]) / length
    apex = np.asarray(fit.apex_xy, float)
    up = 1.0 if (apex - p1) @ normal > 0 else -1.0
    height = abs(float((apex - p1) @ normal))
    interior = 0.5 * (p1 + p2) + up * normal * 0.4 * height

    def elevation(pts: np.ndarray) -> np.ndarray:
        return up * ((pts - p1) @ normal)

    base = np.asarray(interface, float)
    kept = base[elevation(base) < baseline_margin_px] if keep_margin_points else np.empty((0, 2))
    points = base
    for _ in range(max(1, iterations)):
        curve = fit.sample(step_px)
        found, strength = sample_edge_along_normals(
            image, curve, search_px=search_px, interior_point=interior
        )
        # on a dark substrate the drop and its reflection merge: an edge found
        # below the contact line belongs to the reflection, not the interface
        good = (
            (elevation(curve) >= baseline_margin_px)
            & (elevation(found) >= baseline_margin_px)
            & (strength > 0)
        )
        if good.any():
            good &= strength >= min_relative_contrast * float(np.median(strength[good]))
        if good.sum() < 10:
            break
        if refine_contacts:
            moved = [
                _contact_on_line(found[good], c, p1, p2, up, baseline_margin_px, max_contact_shift_px)
                for c in (p1, p2)
            ]
            shifted = [np.hypot(*(m - c)) >= min_contact_shift_px for m, c in zip(moved, (p1, p2))]
            if any(shifted):
                p1 = moved[0] if shifted[0] else p1
                p2 = moved[1] if shifted[1] else p2
                # the band points were measured against the old contacts
                kept = np.empty((0, 2))
        near1 = kept[np.hypot(*(kept - p1).T) < np.hypot(*(kept - p2).T)]
        near2 = kept[np.hypot(*(kept - p1).T) >= np.hypot(*(kept - p2).T)]
        points = np.vstack([p1, near1, found[good], near2, p2])
        points = _order_along(points, fit, p1)
        fit = (fitter or fit_arc_spline)(points, p1, p2, **fit_kwargs)
    return fit, points


def _contact_on_line(
    edge_points: np.ndarray,
    contact: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    up: float,
    margin: float,
    max_shift: float,
    band: float = 12.0,
) -> np.ndarray:
    """Slide a contact point along the P1-P2 line onto the extrapolated edge.

    The edge points of this contact's flank between ``margin`` and
    ``margin + band`` above the line are fitted with a quadratic
    ``along(elevation)`` and extrapolated to zero elevation.
    """
    unit = (p2 - p1) / (np.hypot(*(p2 - p1)) or 1.0)
    normal = np.array([-unit[1], unit[0]])
    rel = edge_points - p1
    along = rel @ unit
    elev = up * (rel @ normal)
    c_along = float((contact - p1) @ unit)
    mid = 0.5 * float((p2 - p1) @ unit)
    same_side = (along < mid) if c_along < mid else (along >= mid)
    sel = same_side & (elev >= margin) & (elev <= margin + band)
    if sel.sum() < 5:
        return contact
    deg = 2 if sel.sum() >= 8 else 1
    coef = np.polyfit(elev[sel], along[sel], deg)
    new_along = float(np.polyval(coef, 0.0))
    if not np.isfinite(new_along) or abs(new_along - c_along) > max_shift:
        return contact
    return p1 + new_along * unit


def _order_along(points: np.ndarray, fit: ArcSplineFit, p1: np.ndarray) -> np.ndarray:
    """Order points by their position along the fitted spline (P1 to P2)."""
    curve = fit.sample(1.0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
    idx = np.argmin(((points[:, None, :] - curve[None, ::4, :]) ** 2).sum(-1), axis=1)
    return points[np.argsort(s[::4][idx], kind="stable")]
