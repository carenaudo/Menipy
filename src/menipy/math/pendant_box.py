"""Anchored Young-Laplace pendant profiles with a golden-section Bond search.

A pendant interface, apex at the bottom and ``z`` upward, obeys
``dφ/ds = 2 - Bo z - sin φ / x`` in apex radii, so each Bond number
``Bo = Δρ g b² / γ`` (``b`` the apex radius) gives one dimensionless profile.
The profile has two zones with different roles:

* zone 1, apex -> equator (``φ = 90°``, the maximum diameter): convex; its size
  fixes the scale;
* zone 2, equator -> needle: the neck, where gravity has had the most height to
  act; it carries most of the information on ``Bo``.

Only well-conditioned measurements anchor a profile to the edge: the apex height
(an extremum *value* in ``z``) and the equatorial radius ``R_e`` (an extremum
value in ``x``). The *heights* of the equator and the apex's sideways position
sit on very flat extrema and are left to the model. With those anchors the only
free shape parameter is ``Bo``, found by a one-dimensional golden-section search
against the edge points of both zones.

Profiles for a Bond grid are integrated once per process in one vectorized RK4
pass; the search interpolates between neighbouring rows.

References
----------
Rotenberg, Y., Boruvka, L. and Neumann, A. W. (1983). Determination of surface
tension and contact angle from the shapes of axisymmetric fluid interfaces.
J. Colloid Interface Sci., 93(1), 169-183.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from menipy.math.sessile_box import (
    _polyline_d2,
    golden_section_minimize,
    join_quantiles,
    segment_count,
)


@dataclass(frozen=True)
class _PendantTable:
    bond: np.ndarray  # (G,)
    s: np.ndarray  # (S,)
    x: np.ndarray  # (G, S)
    z: np.ndarray  # (G, S)
    phi: np.ndarray  # (G, S)
    kappa: np.ndarray  # (G, S) d(phi)/ds
    length: np.ndarray  # (G,) valid samples per row


@lru_cache(maxsize=2)
def pendant_table(
    n_bond: int = 201, bond_max: float = 1.0, ds: float = 4e-3, s_max: float = 7.0
) -> _PendantTable:
    """Integrate dimensionless pendant profiles for a linear ``Bo`` grid (cached).

    Parameters
    ----------
    n_bond : int, optional
        Grid size, from ``Bo = 0`` (sphere) to ``bond_max``.
    bond_max : float, optional
        Largest Bond number.
    ds : float, optional
        Arc-length step in apex radii.
    s_max : float, optional
        Integration length in apex radii.

    Returns
    -------
    _PendantTable
        Profiles ``x(s), z(s), phi(s), kappa(s)`` per row; a row stops (and its
        ``length`` ends) when it closes onto the axis or turns back past 225°.
    """
    bond = np.linspace(0.0, bond_max, n_bond)
    s0 = 1e-3
    s = s0 + ds * np.arange(int(np.ceil((s_max - s0) / ds)) + 1)
    g = len(bond)
    x = np.empty((g, len(s)))
    z = np.empty((g, len(s)))
    phi = np.empty((g, len(s)))
    state = np.stack([np.full(g, s0), np.full(g, 0.5 * s0**2), np.full(g, s0)])
    alive = np.ones(g, dtype=bool)
    length = np.full(g, len(s))

    def rhs(y: np.ndarray) -> np.ndarray:
        xx, zz, pp = y
        return np.stack([np.cos(pp), np.sin(pp), 2.0 - bond * zz - np.sin(pp) / xx])

    for j in range(len(s)):
        x[:, j], z[:, j], phi[:, j] = state
        closed = (state[0] < 0.02) & (state[2] > 0.5 * np.pi)
        dead = alive & (closed | (state[2] > 1.25 * np.pi))
        length[dead] = j
        alive &= ~dead
        if not alive.any():
            length[length > j] = j
            break
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * ds * k1)
        k3 = rhs(state + 0.5 * ds * k2)
        k4 = rhs(state + ds * k3)
        state = state + ds / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4) * alive
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = 2.0 - bond[:, None] * z - np.sin(phi) / x
    return _PendantTable(bond, s, x, z, phi, kappa, length)


def _crossing(f: np.ndarray, start: int, stop: int) -> tuple[int, float] | None:
    """First ``j`` in ``[start, stop - 1)`` with ``f[j] < 0 <= f[j + 1]``, and the fraction."""
    seg = f[start:stop]
    hit = np.flatnonzero((seg[:-1] < 0.0) & (seg[1:] >= 0.0))
    if hit.size == 0:
        return None
    j = start + int(hit[0])
    denom = f[j + 1] - f[j]
    return j, float(-f[j] / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class PendantShape:
    """Young-Laplace pendant profile anchored to a drop's apex and equator.

    Attributes
    ----------
    bond : float
        Bond number ``Δρ g b² / γ`` of the matched profile.
    apex_radius_px : float
        Apex radius ``b`` in pixels.
    equator_radius_px : float
        Equatorial (maximum) radius, pixels -- the anchor.
    rms_px : float
        Robust RMS distance from the edge points to the profile.
    s_px : np.ndarray
        Arc length from the apex, pixels.
    x_px, z_px : np.ndarray
        Radius and height above the apex along the profile, pixels.
    phi : np.ndarray
        Tangent angle from the horizontal, radians (0 at the apex, 90° at the
        equator, above 90° in the neck).
    kappa_px : np.ndarray
        Meridional curvature ``d(phi)/ds`` in ``1/px``.
    i_equator : int
        Index of the equator sample (``phi = 90°``).
    needle_angle_deg : float
        ``phi`` where the profile reaches the needle height.
    evaluations : int
        Objective evaluations used by the search.
    """

    bond: float
    apex_radius_px: float
    equator_radius_px: float
    rms_px: float
    s_px: np.ndarray
    x_px: np.ndarray
    z_px: np.ndarray
    phi: np.ndarray
    kappa_px: np.ndarray
    i_equator: int
    needle_angle_deg: float
    evaluations: int

    def _zone(self, zone: int) -> slice:
        return slice(0, self.i_equator + 1) if zone == 1 else slice(self.i_equator, None)

    def segments_needed(self, tol_px: float, zone: int, order: int = 2) -> int:
        """Segments one zone needs for a maximum deviation ``tol_px``.

        Parameters
        ----------
        tol_px : float
            Target maximum deviation in pixels.
        zone : {1, 2}
            1 for apex -> equator, 2 for equator -> needle.
        order : {1, 2}, optional
            1 for circular arcs, 2 for clothoids.

        Returns
        -------
        int
            At least one segment.
        """
        sl = self._zone(zone)
        return segment_count(self.s_px[sl], self.kappa_px[sl], tol_px, order)

    def join_positions(self, n: int, zone: int, order: int = 2) -> np.ndarray:
        """Arc-length positions (px from the apex) of the ``n - 1`` joins in one zone."""
        sl = self._zone(zone)
        return join_quantiles(self.s_px[sl], self.kappa_px[sl], n, order=order)

    @property
    def equator_s_px(self) -> float:
        """Arc length of the equator from the apex, pixels."""
        return float(self.s_px[self.i_equator])

    def surface_tension_mN_m(self, px_per_mm: float, delta_rho: float, g: float) -> float:
        """Surface tension ``Δρ g b² / Bo`` in mN/m.

        Parameters
        ----------
        px_per_mm : float
            Image scale.
        delta_rho : float
            Density difference, kg/m³.
        g : float
            Gravitational acceleration, m/s².

        Returns
        -------
        float
            Surface tension, or NaN when ``Bo`` is too small to resolve it.
        """
        if px_per_mm <= 0 or self.bond < 1e-3:
            return float("nan")
        b_m = self.apex_radius_px / px_per_mm / 1000.0
        return float(delta_rho * g * b_m**2 / self.bond * 1000.0)


def fit_pendant_box(
    points: np.ndarray,
    equator_radius_px: float,
    top_height_px: float,
    *,
    max_points: int = 300,
    clip_px: float = 3.0,
    coarse_stride: int = 5,
    tol: float = 0.02,
) -> PendantShape:
    """Find the Bond number whose anchored pendant profile best follows the edge.

    Parameters
    ----------
    points : np.ndarray
        Edge points of both sides as ``(r, z)``: distance from the axis and
        height above the apex, pixels; shape ``(M, 2)``.
    equator_radius_px : float
        Maximum radius of the drop (the scale anchor).
    top_height_px : float
        Height of the needle contact above the apex; the profile ends there.
    max_points : int, optional
        Edge points used by the search (evenly subsampled).
    clip_px : float, optional
        Distances are clipped here so stray points cannot dominate.
    coarse_stride : int, optional
        Grid stride of the bracketing scan.
    tol : float, optional
        Golden-section tolerance in grid rows.

    Returns
    -------
    PendantShape
        The matched profile, its Bond number and curvature.

    Raises
    ------
    ValueError
        If the anchors are degenerate or no profile reaches the needle height.
    """
    if equator_radius_px <= 0 or top_height_px <= 0:
        raise ValueError("degenerate pendant anchors")
    table = pendant_table()
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).round().astype(int)]
    clip2 = clip_px**2
    ds = table.s[1] - table.s[0]

    def matched(u: float):
        i = int(np.clip(np.floor(u), 0, len(table.bond) - 2))
        w = float(np.clip(u - i, 0.0, 1.0))
        valid = int(min(table.length[i], table.length[i + 1]))
        if valid < 3:
            return None

        def mix(a):
            return (1.0 - w) * a[i, :valid] + w * a[i + 1, :valid]

        x, z, phi, kappa = mix(table.x), mix(table.z), mix(table.phi), mix(table.kappa)
        eq = _crossing(phi - 0.5 * np.pi, 0, valid)
        if eq is None:
            return None
        je, fe = eq
        x_eq = x[je] + fe * (x[je + 1] - x[je])
        scale = equator_radius_px / x_eq
        cut = _crossing(z - top_height_px / scale, je, valid)
        if cut is None:
            return None
        bond = (1.0 - w) * table.bond[i] + w * table.bond[i + 1]
        return je, fe, cut[0], cut[1], scale, x, z, phi, kappa, float(bond)

    def objective(u: float) -> float:
        m = matched(u)
        if m is None:
            return np.inf
        _, _, jc, _, scale, x, z, *_ = m
        stride = max(1, int(4.0 / (ds * scale)))
        idx = np.append(np.arange(0, jc + 1, stride), jc + 1)
        prof = np.column_stack([x[idx], z[idx]]) * scale
        return float(np.mean(np.minimum(_polyline_d2(pts, prof), clip2)))

    g = len(table.bond)
    coarse = np.arange(0, g - 1, coarse_stride)
    scores = [objective(float(u)) for u in coarse]
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        raise ValueError("no pendant profile reaches the needle height")
    lo = float(coarse[max(best - 1, 0)])
    hi = float(coarse[min(best + 1, len(coarse) - 1)])
    u, fu, n_eval = golden_section_minimize(objective, lo, hi, tol=tol)
    je, fe, jc, fc, scale, x, z, phi, kappa, bond = matched(u)

    # samples up to the cut, with the exact equator and needle points inserted
    def at(a, j, f):
        return a[j] + f * (a[j + 1] - a[j])

    s = table.s[: len(x)] - table.s[0]
    keep = np.arange(jc + 1)
    ins_eq = je + 1
    cols = []
    for a in (s, x, z, phi, kappa):
        col = np.concatenate([a[keep[:ins_eq]], [at(a, je, fe)], a[keep[ins_eq:]], [at(a, jc, fc)]])
        cols.append(col)
    s_d, x_d, z_d, phi_d, kappa_d = cols
    phi_d[ins_eq] = 0.5 * np.pi
    return PendantShape(
        bond=bond,
        apex_radius_px=float(scale),
        equator_radius_px=float(equator_radius_px),
        rms_px=float(np.sqrt(fu)),
        s_px=s_d * scale,
        x_px=x_d * scale,
        z_px=z_d * scale,
        phi=phi_d,
        kappa_px=kappa_d / scale,
        i_equator=int(ins_eq),
        needle_angle_deg=float(np.degrees(phi_d[-1])),
        evaluations=len(coarse) + n_eval,
    )
