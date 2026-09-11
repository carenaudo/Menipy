"""Box-matched Young-Laplace sessile profiles with a golden-section Bond search.

An axisymmetric sessile interface is fixed, up to scale, by the Bond number
``Bo = Δρ g b² / γ`` (``b`` the apex radius): with the apex radius as unit
length, each ``Bo`` gives one dimensionless profile. The drop's *box* -- the
half-width ``a`` from the apex axis to the contact point and the apex height
``H`` -- then selects the point on that profile where ``z / x = H / a``: this is
the contact point, which fixes the scale ``b = a / x`` and the contact angle.
So for a measured box the only free shape parameter is ``Bo``, and the best one
is found by a one-dimensional golden-section search against the edge points.

Dimensionless profiles for a log-spaced ``Bo`` grid are integrated once per
process in a single vectorized RK4 pass; the search interpolates between
neighbouring grid rows, so an evaluation costs no ODE integration.

The matched profile is also a physics prior for piecewise models: its
curvature ``κ(s)`` and ``κ'(s)`` say how many circular arcs a side needs for a
given tolerance and where their joins belong.

References
----------
Rotenberg, Y., Boruvka, L. and Neumann, A. W. (1983). Determination of surface
tension and contact angle from the shapes of axisymmetric fluid interfaces.
J. Colloid Interface Sci., 93(1), 169-183.
Kiefer, J. (1953). Sequential minimax search for a maximum. Proc. Amer. Math.
Soc., 4(3), 502-506.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi

# Maximum deviation of an arc that leaves a curve tangentially and rejoins it
# after length L, where the curvature varies linearly: |kappa'| L^3 / 40.5.
ARC_DEVIATION_C = 40.5
# Maximum deviation of a G1 Hermite clothoid (linear curvature, matching both end
# points and tangents) from a curve whose curvature varies quadratically:
# |kappa''| L^4 / 384 (the heading error is kappa'' (L^2 s - 3 L s^2 + 2 s^3) / 12).
CLOTHOID_DEVIATION_C = 384.0


@dataclass(frozen=True)
class _Table:
    bond: np.ndarray  # (G,)
    s: np.ndarray  # (S,)
    x: np.ndarray  # (G, S)
    z: np.ndarray  # (G, S)
    phi: np.ndarray  # (G, S)
    kappa: np.ndarray  # (G, S) d(phi)/ds


@lru_cache(maxsize=4)
def profile_table(
    n_bond: int = 161, bond_max: float = 300.0, ds: float = 2e-3, s_max: float = 3.6
) -> _Table:
    """Integrate dimensionless sessile profiles for a ``Bo`` grid (cached).

    Parameters
    ----------
    n_bond : int, optional
        Grid size; the first entry is ``Bo = 0`` (spherical cap), the rest are
        log-spaced from ``1e-3`` to ``bond_max``.
    bond_max : float, optional
        Largest Bond number.
    ds : float, optional
        Arc-length step in apex radii.
    s_max : float, optional
        Integration length; a hemisphere is ``pi`` long.

    Returns
    -------
    _Table
        Profiles ``x(s), z(s), phi(s), kappa(s)`` per grid row, frozen once the
        tangent passes 180 degrees.
    """
    bond = np.concatenate([[0.0], np.logspace(-3.0, np.log10(bond_max), n_bond - 1)])
    s0 = 1e-3
    s = s0 + ds * np.arange(int(np.ceil((s_max - s0) / ds)) + 1)
    g = len(bond)
    x = np.empty((g, len(s)))
    z = np.empty((g, len(s)))
    phi = np.empty((g, len(s)))
    state = np.stack([np.full(g, s0), np.full(g, 0.5 * s0**2), np.full(g, s0)])
    alive = np.ones(g, dtype=bool)

    def rhs(y: np.ndarray) -> np.ndarray:
        xx, zz, pp = y
        return np.stack([np.cos(pp), np.sin(pp), 2.0 + bond * zz - np.sin(pp) / xx])

    for j in range(len(s)):
        x[:, j], z[:, j], phi[:, j] = state
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * ds * k1)
        k3 = rhs(state + 0.5 * ds * k2)
        k4 = rhs(state + ds * k3)
        step = ds / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        alive &= state[2] < np.pi + 0.05
        state = state + step * alive
    kappa = 2.0 + bond[:, None] * z - np.sin(phi) / x
    return _Table(bond, s, x, z, phi, kappa)


@dataclass(frozen=True)
class BoxShape:
    """Young-Laplace profile matched to one side's box.

    Attributes
    ----------
    bond : float
        Bond number of the matched profile.
    theta_deg : float
        Contact angle of the matched profile.
    apex_radius_px : float
        Apex radius ``b`` in pixels.
    rms_px : float
        Robust RMS distance from the edge points to the profile.
    s_px : np.ndarray
        Arc length from the apex, pixels.
    x_px, z_px : np.ndarray
        Profile in the side frame (apex at origin, x outward, z towards the
        substrate).
    kappa_px : np.ndarray
        Curvature ``d(phi)/ds`` in ``1/px`` along ``s_px``.
    evaluations : int
        Objective evaluations used by the search.
    """

    bond: float
    theta_deg: float
    apex_radius_px: float
    rms_px: float
    s_px: np.ndarray
    x_px: np.ndarray
    z_px: np.ndarray
    kappa_px: np.ndarray
    evaluations: int

    def curvature_slope(self) -> np.ndarray:
        """Return ``d(kappa)/ds`` in ``1/px^2`` along the profile."""
        return np.gradient(self.kappa_px, self.s_px)

    def curvature_derivative(self, order: int) -> np.ndarray:
        """Return ``d^order(kappa)/ds^order`` along the profile."""
        d = self.kappa_px
        for _ in range(order):
            d = np.gradient(d, self.s_px)
        return d

    def heading(self) -> np.ndarray:
        """Tangent angle along the profile, radians (0 at the apex)."""
        dx = np.gradient(self.x_px, self.s_px)
        dz = np.gradient(self.z_px, self.s_px)
        return np.unwrap(np.arctan2(dz, dx))

    def segments_needed(self, tol_px: float, order: int = 1) -> int:
        """Segments needed for a maximum deviation ``tol_px`` with optimal joins.

        ``order=1`` counts circular arcs (constant curvature): an arc that
        leaves the curve tangentially and rejoins it after length ``L`` deviates
        by ``|κ'| L³ / 40.5``. ``order=2`` counts G1 Hermite clothoids (linear
        curvature), which deviate by ``|κ''| L⁴ / 384``. With lengths
        ``L ∝ |κ^(order)|^(-1/(order+2))`` every segment reaches the same
        deviation, so ``N = ∫ (|κ^(order)| / (C tol))^(1/(order+2)) ds``.

        Parameters
        ----------
        tol_px : float
            Target maximum deviation in pixels.
        order : {1, 2}, optional
            1 for arcs, 2 for clothoids.

        Returns
        -------
        int
            At least one segment.
        """
        return segment_count(self.s_px, self.kappa_px, tol_px, order)

    def arcs_needed(self, tol_px: float) -> int:
        """Arcs needed for a maximum deviation ``tol_px`` (see ``segments_needed``)."""
        return self.segments_needed(tol_px, order=1)

    def join_positions(self, n_arcs: int, floor: float = 0.05, order: int = 1) -> np.ndarray:
        """Arc-length positions (px from the apex) of the ``n_arcs - 1`` joins.

        Joins sit at equal quantiles of ``∫ |κ^(order)|^(1/(order+2)) ds``, which
        equalizes the segments' deviation; ``floor`` (relative to the mean
        density) keeps nearly uniform stretches from collapsing into one segment.

        Parameters
        ----------
        n_arcs : int
            Segments on this side.
        floor : float, optional
            Relative density floor.
        order : {1, 2}, optional
            1 for arcs, 2 for clothoids.

        Returns
        -------
        np.ndarray
            Increasing join positions, shape ``(n_arcs - 1,)``.
        """
        return join_quantiles(self.s_px, self.kappa_px, n_arcs, floor, order)


def _curvature_derivative(s: np.ndarray, kappa: np.ndarray, order: int) -> np.ndarray:
    d = np.asarray(kappa, float)
    for _ in range(order):
        d = np.gradient(d, s)
    return d


def segment_count(s: np.ndarray, kappa: np.ndarray, tol_px: float, order: int = 1) -> int:
    """Segments needed to follow a curvature profile within ``tol_px``.

    ``N = ∫ (|κ^(order)| / (C tol))^(1/(order+2)) ds`` with ``C = 40.5`` for
    circular arcs (``order=1``) and ``C = 384`` for G1 clothoids (``order=2``).

    Parameters
    ----------
    s : np.ndarray
        Arc length, pixels.
    kappa : np.ndarray
        Curvature ``d(phi)/ds`` along ``s``, ``1/px``.
    tol_px : float
        Target maximum deviation in pixels.
    order : {1, 2}, optional
        1 for arcs, 2 for clothoids.

    Returns
    -------
    int
        At least one segment.
    """
    constant = {1: ARC_DEVIATION_C, 2: CLOTHOID_DEVIATION_C}[order]
    power = 1.0 / (order + 2)
    density = np.abs(_curvature_derivative(s, kappa, order)) ** power
    total = float(np.trapezoid(density, s)) / (constant * tol_px) ** power
    return max(1, int(np.ceil(total)))


def join_quantiles(
    s: np.ndarray, kappa: np.ndarray, n: int, floor: float = 0.05, order: int = 1
) -> np.ndarray:
    """Positions of the ``n - 1`` joins that equalize the segments' deviation.

    Joins sit at equal quantiles of ``∫ |κ^(order)|^(1/(order+2)) ds``; ``floor``
    (relative to the mean density) keeps nearly uniform stretches from
    collapsing into one segment.

    Parameters
    ----------
    s : np.ndarray
        Arc length, pixels.
    kappa : np.ndarray
        Curvature along ``s``.
    n : int
        Number of segments.
    floor : float, optional
        Relative density floor.
    order : {1, 2}, optional
        1 for arcs, 2 for clothoids.

    Returns
    -------
    np.ndarray
        Increasing join positions along ``s``, shape ``(n - 1,)``.
    """
    density = np.abs(_curvature_derivative(s, kappa, order)) ** (1.0 / (order + 2))
    density = density + floor * (float(np.mean(density)) + 1e-12)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s))])
    targets = cum[-1] * np.arange(1, n) / n
    return np.interp(targets, cum, s)


def _polyline_d2(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Squared distance from each point to a polyline (nearest-vertex segments)."""
    d2v = ((pts[:, None, :] - poly[None, :, :]) ** 2).sum(-1)
    k = np.argmin(d2v, axis=1)
    best = d2v[np.arange(len(pts)), k]
    for lo in (k - 1, k):  # the segments before and after the nearest vertex
        ok = (lo >= 0) & (lo < len(poly) - 1)
        a = poly[np.clip(lo, 0, len(poly) - 2)]
        b = poly[np.clip(lo + 1, 1, len(poly) - 1)]
        ab = b - a
        t = np.clip(((pts - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        d2 = ((pts - a - t[:, None] * ab) ** 2).sum(1)
        best = np.where(ok, np.minimum(best, d2), best)
    return best


def _match_row(x: np.ndarray, z: np.ndarray, ratio: float) -> tuple[int, float] | None:
    """Index and fraction of the first sample where ``z / x`` reaches ``ratio``."""
    f = z - ratio * x
    hit = np.flatnonzero(f[1:] >= 0.0)
    if hit.size == 0:
        return None
    j = int(hit[0])  # crossing between samples j and j + 1
    denom = f[j + 1] - f[j]
    frac = float(-f[j] / denom) if denom > 0 else 0.0
    return j, frac


def _blend(table: _Table, u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Profile at continuous grid index ``u`` by linear interpolation of rows."""
    i = int(np.clip(np.floor(u), 0, len(table.bond) - 2))
    w = float(np.clip(u - i, 0.0, 1.0))

    def mix(a):
        return (1.0 - w) * a[i] + w * a[i + 1]

    return mix(table.x), mix(table.z), mix(table.phi), mix(table.kappa), float(mix(table.bond))


def golden_section_minimize(
    func: Callable[[float], float], lo: float, hi: float, tol: float = 1e-3, max_iter: int = 60
) -> tuple[float, float, int]:
    """Minimize a unimodal function on ``[lo, hi]`` by golden-section search.

    Parameters
    ----------
    func : callable
        Scalar objective.
    lo, hi : float
        Bracket.
    tol : float, optional
        Absolute bracket width at which to stop.
    max_iter : int, optional
        Iteration cap.

    Returns
    -------
    x : float
        Location of the minimum found.
    fx : float
        Objective there.
    evaluations : int
        Number of objective calls.
    """
    a, b = float(lo), float(hi)
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = func(c), func(d)
    n = 2
    for _ in range(max_iter):
        if b - a <= tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = func(d)
        n += 1
    return (c, fc, n) if fc <= fd else (d, fd, n)


def fit_side_box(
    points: np.ndarray,
    half_width_px: float,
    height_px: float,
    *,
    max_points: int = 160,
    clip_px: float = 3.0,
    coarse_stride: int = 12,
) -> BoxShape:
    """Find the Bond number whose box-matched profile best follows one side.

    Parameters
    ----------
    points : np.ndarray
        Edge points of the side in the side frame: apex at the origin, ``x``
        outward from the axis, ``z`` towards the substrate; shape ``(M, 2)``.
    half_width_px : float
        Horizontal distance from the apex axis to the contact point.
    height_px : float
        Apex height above the contact chord.
    max_points : int, optional
        Edge points used by the search (evenly subsampled).
    clip_px : float, optional
        Distances are clipped here so stray points cannot dominate.
    coarse_stride : int, optional
        Grid stride of the bracketing scan before the golden-section search.

    Returns
    -------
    BoxShape
        The matched profile and its Bond number, angle and curvature.

    Raises
    ------
    ValueError
        If the box is degenerate or no profile reaches its aspect ratio.
    """
    if half_width_px <= 0 or height_px <= 0:
        raise ValueError("degenerate box")
    table = profile_table()
    ratio = height_px / half_width_px
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).round().astype(int)]
    clip2 = clip_px**2

    def matched(u: float):
        x, z, phi, kappa, bond = _blend(table, u)
        hit = _match_row(x, z, ratio)
        if hit is None:
            return None
        j, frac = hit
        scale = half_width_px / (x[j] + frac * (x[j + 1] - x[j]))
        return j, frac, scale, x, z, phi, kappa, bond

    def objective(u: float) -> float:
        m = matched(u)
        if m is None:
            return np.inf
        j, _, scale, x, z, *_ = m
        # coarse profile vertices (~4 px apart), then exact point-to-segment
        # distance on the two segments around the nearest vertex
        stride = max(1, int(4.0 / ((table.s[1] - table.s[0]) * scale)), int(np.ceil((j + 2) / 600)))
        idx = np.append(np.arange(0, j + 1, stride), j + 1)
        prof = np.column_stack([x[idx], z[idx]]) * scale
        return float(np.mean(np.minimum(_polyline_d2(pts, prof), clip2)))

    g = len(table.bond)
    coarse = np.arange(0, g, coarse_stride)
    scores = [objective(float(u)) for u in coarse]
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise ValueError("no Young-Laplace profile reaches the box aspect ratio")
    lo = float(coarse[max(best - 1, 0)])
    hi = float(coarse[min(best + 1, len(coarse) - 1)])
    u, fu, n_eval = golden_section_minimize(objective, lo, hi, tol=0.05)
    j, frac, scale, x, z, phi, kappa, bond = matched(u)
    s = table.s[: j + 2] * scale
    s = s - s[0]
    xs = x[: j + 2] * scale
    zs = z[: j + 2] * scale
    ks = kappa[: j + 2] / scale
    # end the profile exactly on the contact point
    end = j + 1
    s[end] = s[j] + frac * (s[end] - s[j])
    xs[end], zs[end] = half_width_px, height_px
    theta = float(np.degrees(phi[j] + frac * (phi[end] - phi[j])))
    ks[end] = ks[j] + frac * (ks[end] - ks[j])
    return BoxShape(
        bond=bond,
        theta_deg=theta,
        apex_radius_px=float(scale),
        rms_px=float(np.sqrt(fu)),
        s_px=s,
        x_px=xs,
        z_px=zs,
        kappa_px=ks,
        evaluations=len(coarse) + n_eval,
    )
