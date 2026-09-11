"""Feasibility study: zone-wise arc/clothoid splines for pendant drops.

A pendant profile has two regions with different geometry:

* zone 1, apex -> equator (maximum diameter): convex, curvature falling
  smoothly from the apex; the equator tangent is vertical by definition, just as
  the apex tangent is horizontal;
* zone 2, equator -> needle contact (P1, P2): the neck, where the meridional
  curvature changes sign (inflection), then the attachment to the needle.

Questions answered on exact Young-Laplace profiles (independent ``solve_ivp``):

1. How many circular arcs versus G1 clothoids each zone needs for a given
   tolerance (from the curvature derivatives, as in ``sessile_box``).
2. Whether zone 1 alone -- its box (equatorial radius R_e, apex-to-equator
   height H_e) plus a golden-section search over the Bond number -- recovers the
   Bond number and hence the surface tension, and how edge noise propagates.
3. Whether a clothoid chain with the equator as a node of fixed vertical tangent
   reproduces a whole side (apex -> equator -> needle) with few segments.

This script measures; it changes no application behavior.

Usage
-----
``uv run python scripts/pendant_zones_study.py [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from menipy.common.clothoid_spline import ClothoidSegment, solve_hermite  # noqa: E402
from menipy.math.sessile_box import (  # noqa: E402
    ARC_DEVIATION_C,
    CLOTHOID_DEVIATION_C,
    golden_section_minimize,
)

EQUATOR_RADIUS_PX = 150.0


@lru_cache(maxsize=32)
def pendant_profile(bond: float, needle_fraction: float = 0.5, ds: float = 2e-4):
    """Dimensionless pendant half-profile from the apex (bottom) up to the needle.

    Apex radius 1, ``z`` upward from the apex, ``dφ/ds = 2 - Bo z - sin φ / x``.
    The profile ends where, past the equator, ``x`` falls to
    ``needle_fraction * R_e`` (the needle radius).

    Returns
    -------
    dict
        ``s, x, z, phi, kappa`` arrays and the equator and needle indices.
    """

    def rhs(_s, y):
        x, z, phi = y
        return [np.cos(phi), np.sin(phi), 2.0 - bond * z - np.sin(phi) / x]

    s0 = 1e-4
    sol = solve_ivp(rhs, (s0, 30.0), [s0, s0**2 / 2, s0], max_step=ds * 10,
                    rtol=1e-10, atol=1e-12, dense_output=True)
    s = np.arange(s0, sol.t[-1], ds)
    x, z, phi = sol.sol(s)
    i_eq = int(np.argmax(x[: np.argmax(phi > np.pi / 2) + 2]))  # first maximum of x
    r_eq = x[i_eq]
    # a real needle is wider than the neck: cut before or at the neck
    neck = float(np.min(x[i_eq:]))
    fraction = max(needle_fraction, neck / r_eq + 0.08)
    past = np.flatnonzero((np.arange(len(x)) > i_eq) & (x <= fraction * r_eq))
    i_n = int(past[0])
    kappa = 2.0 - bond * z - np.sin(phi) / x
    sl = slice(0, i_n + 1)
    return {"s": s[sl] - s0, "x": x[sl], "z": z[sl], "phi": phi[sl], "kappa": kappa[sl],
            "i_eq": i_eq, "i_n": i_n}


def scaled(bond: float, needle_fraction: float = 0.5):
    """Profile scaled so the equatorial radius is ``EQUATOR_RADIUS_PX``."""
    p = pendant_profile(bond, needle_fraction)
    k = EQUATOR_RADIUS_PX / p["x"][p["i_eq"]]
    return {"s": p["s"] * k, "x": p["x"] * k, "z": p["z"] * k, "phi": p["phi"],
            "kappa": p["kappa"] / k, "i_eq": p["i_eq"], "i_n": p["i_n"], "b_px": k}


def segments_needed(s, kappa, tol, order):
    constant = {1: ARC_DEVIATION_C, 2: CLOTHOID_DEVIATION_C}[order]
    d = kappa
    for _ in range(order):
        d = np.gradient(d, s)
    power = 1.0 / (order + 2)
    return float(np.trapezoid(np.abs(d) ** power, s)) / (constant * tol) ** power


def zone_counts(tol: float = 0.25) -> dict:
    out = {}
    for bond in (0.1, 0.2, 0.3, 0.4):
        p = scaled(bond)
        e = p["i_eq"]
        z1 = slice(0, e + 1)
        z2 = slice(e, None)
        inflection = bool(np.any(np.diff(np.sign(p["kappa"][z2])) != 0))
        out[f"bo{bond}"] = {
            "zone1_arcs": segments_needed(p["s"][z1], p["kappa"][z1], tol, 1),
            "zone1_clothoids": segments_needed(p["s"][z1], p["kappa"][z1], tol, 2),
            "zone2_arcs": segments_needed(p["s"][z2], p["kappa"][z2], tol, 1),
            "zone2_clothoids": segments_needed(p["s"][z2], p["kappa"][z2], tol, 2),
            "zone2_has_inflection": inflection,
            "needle_angle_deg": float(np.degrees(p["phi"][-1])),
            "height_px": float(p["z"][-1]),
        }
    return out


# ----------------------------------------------------------------------------
# Zone-1 box: Bond number from the apex-to-equator shape
# ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def zone1_table(n_bond: int = 121, bond_max: float = 0.6):
    """Dimensionless apex-to-equator profiles for a Bond grid (x, z up to phi = 90 deg)."""
    bonds = np.linspace(0.0, bond_max, n_bond)
    rows = []
    for bond in bonds:
        def rhs(_s, y, b=bond):
            x, z, phi = y
            return [np.cos(phi), np.sin(phi), 2.0 - b * z - np.sin(phi) / x]

        def equator(_s, y):
            return y[2] - np.pi / 2

        equator.terminal = True
        s0 = 1e-4
        sol = solve_ivp(rhs, (s0, 10.0), [s0, s0**2 / 2, s0], events=equator,
                        max_step=2e-3, rtol=1e-10, atol=1e-12)
        rows.append((sol.y[0], sol.y[1]))
    return bonds, rows


def _zone1_row(u: float, n: int = 400):
    """Zone-1 shape at continuous grid index ``u``, rows resampled in phi and blended."""
    bonds, rows = zone1_table()
    i = int(np.clip(np.floor(u), 0, len(bonds) - 2))
    w = float(np.clip(u - i, 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, n)
    shapes = []
    for j in (i, i + 1):
        x, z = rows[j]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
        shapes.append(np.column_stack([np.interp(grid * s[-1], s, x), np.interp(grid * s[-1], s, z)]))
    return (1 - w) * shapes[0] + w * shapes[1], float(bonds[i] + w * (bonds[i + 1] - bonds[i]))


def fit_zone1_bond(points: np.ndarray, r_eq: float, apex_z: float):
    """Golden-section search for the Bond number of zone 1.

    Only well-conditioned measurements anchor the shape: the apex height (an
    extremum value in z) and the equatorial radius (an extremum value in x).
    The equator *height* -- the location of a very flat maximum of x(z) -- is
    left to the model: half a pixel of noise in x moves it by ~sqrt(R_e) px.
    """
    bonds, _ = zone1_table()

    def objective(u: float) -> float:
        shape, _ = _zone1_row(u)
        prof = shape * (r_eq / shape[-1, 0])  # the equator x matches R_e
        prof[:, 1] += apex_z
        d = cKDTree(prof).query(points)[0]
        return float(np.mean(np.minimum(d, 3.0) ** 2))

    coarse = np.arange(0, len(bonds), 8)
    scores = [objective(float(u)) for u in coarse]
    k = int(np.argmin(scores))
    lo, hi = float(coarse[max(k - 1, 0)]), float(coarse[min(k + 1, len(coarse) - 1)])
    u, _, _ = golden_section_minimize(objective, lo, hi, tol=0.01)
    return _zone1_row(u)[1]


def zone1_bond_recovery(seeds: int = 5) -> dict:
    """Bond / surface-tension error from zone-1 points and the equator box."""
    bonds, rows = zone1_table()
    ratio = np.array([z[-1] / x[-1] for x, z in rows])
    out = {"box_ratio_is_monotonic": bool(np.all(np.diff(ratio) > 0))}
    rng = np.random.default_rng(0)
    for bond in (0.1, 0.2, 0.3, 0.4):
        p = scaled(bond)
        e = p["i_eq"]
        pts = np.column_stack([p["x"][: e + 1], p["z"][: e + 1]])
        step = max(1, int(1.0 / (p["s"][1] - p["s"][0])))
        pts = pts[::step]
        for noise in (0.0, 0.5, 1.0):
            est = []
            for _ in range(seeds if noise else 1):
                q = pts + rng.normal(0.0, noise, pts.shape) if noise else pts
                # extremum values from the noisy edge: R_e from a parabola x(z)
                # through the widest points, apex height from a parabola z(x)
                # through the lowest points
                wide = np.argsort(q[:, 0])[-40:]
                c = np.polyfit(q[wide, 1], q[wide, 0], 2)
                r_eq = float(c[2] - c[1] ** 2 / (4 * c[0])) if c[0] < 0 else float(q[wide, 0].max())
                low = q[q[:, 0] < 0.25 * r_eq]
                c = np.polyfit(low[:, 0], low[:, 1], 2)  # symmetric about the axis x = 0
                apex_z = float(c[2])
                est.append(fit_zone1_bond(q, r_eq, apex_z))
            est = np.array(est)
            out[f"bo{bond}_noise{noise}"] = {
                "bond_mean": float(est.mean()), "bond_sd": float(est.std()),
                "gamma_rel_err_mean": float(np.mean([_gamma_rel(bond, b) for b in est])),
            }
    return out


def _gamma_rel(bond_true: float, bond_est: float) -> float:
    """Relative surface-tension error: gamma = drho g b^2 / Bo, b = R_e / x_eq(Bo)."""
    bonds, rows = zone1_table()
    profile = pendant_profile(bond_true)
    x_true = profile["x"][profile["i_eq"]]
    x_est = rows[int(np.argmin(np.abs(bonds - bond_est)))][0][-1]
    return (x_true / x_est) ** 2 * (bond_true / max(bond_est, 1e-9)) - 1.0


@lru_cache(maxsize=1)
def full_table(n_bond: int = 121, bond_max: float = 0.6):
    """Dimensionless profiles from the apex through the neck, resampled in arc length."""
    bonds = np.linspace(0.0, bond_max, n_bond)
    rows = []
    for bond in bonds:
        def rhs(_s, y, b=bond):
            x, z, phi = y
            return [np.cos(phi), np.sin(phi), 2.0 - b * z - np.sin(phi) / x]

        def closing(_s, y):
            return y[0] - 0.15  # stop when the profile narrows close to the axis

        closing.terminal = True
        closing.direction = -1
        s0 = 1e-4
        sol = solve_ivp(rhs, (s0, 8.0), [s0, s0**2 / 2, s0], events=closing,
                        max_step=2e-3, rtol=1e-10, atol=1e-12)
        x, z, phi = sol.y
        i_eq = int(np.argmax(x[: np.argmax(phi > np.pi / 2) + 2]))
        rows.append((x, z, x[i_eq]))
    return bonds, rows


def fit_full_bond(points: np.ndarray, r_eq: float, apex_z: float):
    """Golden-section Bond search on both zones, same anchors as zone 1."""
    bonds, rows = full_table()

    def shape(u):
        i = int(np.clip(np.floor(u), 0, len(bonds) - 2))
        w = float(np.clip(u - i, 0.0, 1.0))
        out = []
        for j in (i, i + 1):
            x, z, x_eq = rows[j]
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
            grid = np.linspace(0.0, s[-1], 800)
            out.append(np.column_stack([np.interp(grid, s, x), np.interp(grid, s, z)]) / x_eq)
        n = min(len(out[0]), len(out[1]))
        return (1 - w) * out[0][:n] + w * out[1][:n], float(bonds[i] + w * (bonds[i + 1] - bonds[i]))

    def objective(u: float) -> float:
        prof, _ = shape(u)
        prof = prof * r_eq
        prof[:, 1] += apex_z
        d = cKDTree(prof).query(points)[0]
        return float(np.mean(np.minimum(d, 3.0) ** 2))

    coarse = np.arange(0, len(bonds), 8)
    scores = [objective(float(u)) for u in coarse]
    k = int(np.argmin(scores))
    lo, hi = float(coarse[max(k - 1, 0)]), float(coarse[min(k + 1, len(coarse) - 1)])
    u, _, _ = golden_section_minimize(objective, lo, hi, tol=0.01)
    return shape(u)[1]


def zone_information(seeds: int = 5) -> dict:
    """Bond from zone 1 alone versus zones 1+2, same anchors and noise."""
    bonds, rows = full_table()
    rng = np.random.default_rng(1)
    out = {}
    for bond in (0.1, 0.2, 0.3, 0.4):
        p = scaled(bond)
        step = max(1, int(1.0 / (p["s"][1] - p["s"][0])))
        pts_all = np.column_stack([p["x"], p["z"]])[::step]
        for noise in (0.5, 1.0):
            est = {"zone1": [], "zones12": []}
            for _ in range(seeds):
                q = pts_all + rng.normal(0.0, noise, pts_all.shape)
                wide = np.argsort(q[:, 0])[-40:]
                c = np.polyfit(q[wide, 1], q[wide, 0], 2)
                r_eq = float(c[2] - c[1] ** 2 / (4 * c[0])) if c[0] < 0 else float(q[wide, 0].max())
                low = q[q[:, 0] < 0.25 * r_eq]
                apex_z = float(np.polyfit(low[:, 0], low[:, 1], 2)[2])
                i_eq_obs = int(np.argmax(q[:, 0]))
                est["zone1"].append(fit_zone1_bond(q[: i_eq_obs + 1], r_eq, apex_z))
                est["zones12"].append(fit_full_bond(q, r_eq, apex_z))
            out[f"bo{bond}_noise{noise}"] = {
                k: {"bond_mean": float(np.mean(v)), "bond_sd": float(np.std(v))} for k, v in est.items()
            }
    return out


# ----------------------------------------------------------------------------
# Zone-structured clothoid chain on one side (research fitter, FD jacobian)
# ----------------------------------------------------------------------------


def fit_side_chain(pts: np.ndarray, apex: np.ndarray, needle: np.ndarray, eq_guess: np.ndarray,
                   n1: int, n2: int, true_needle_heading: float):
    """Fit apex -> equator -> needle with n1 / n2 clothoids per zone.

    Nodes: apex (fixed position here, heading 0), zone-1 joins (x, z, heading),
    equator (x, z free, heading fixed at 90 deg), zone-2 joins, needle contact
    (fixed position, heading free = the angle at the needle).
    """
    tree = cKDTree(pts)

    def nodes(p):
        k = 0
        pos, head = [apex], [0.0]
        for _ in range(n1 - 1):
            pos.append(p[k : k + 2])
            head.append(p[k + 2])
            k += 3
        pos.append(p[k : k + 2])  # equator: vertical tangent
        head.append(np.pi / 2)
        k += 2
        for _ in range(n2 - 1):
            pos.append(p[k : k + 2])
            head.append(p[k + 2])
            k += 3
        pos.append(needle)
        head.append(p[k])
        return np.array(pos), np.array(head)

    def dense(p):
        pos, head = nodes(p)
        h = solve_hermite(pos[:-1], pos[1:], head[:-1], head[1:])
        out = []
        for j in range(len(h.L)):
            seg = ClothoidSegment(pos[j], head[j], h.k0[j], (h.k1[j] - h.k0[j]) / h.L[j], h.L[j])
            out.append(seg.points(np.linspace(0.0, h.L[j], 200)))
        return np.vstack(out), h

    def residuals(p):
        d, h = dense(p)
        return np.concatenate([cKDTree(d).query(pts)[0], 0.5 * tree.query(d[::5])[0],
                               [1e3 * np.count_nonzero(~h.valid)]])

    # initial guess: joins spread along the data by arc length within each zone
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    i_eq = int(np.argmin(np.hypot(*(pts - eq_guess).T)))
    d = np.gradient(pts, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    p0 = []
    for zone, (a, b, n) in enumerate(((0, i_eq, n1), (i_eq, len(pts) - 1, n2))):
        for f in np.arange(1, n) / n:
            i = int(np.argmin(np.abs(s - (s[a] + f * (s[b] - s[a])))))
            p0 += [pts[i, 0], pts[i, 1], heading[i]]
        if zone == 0:
            p0 += [pts[i_eq, 0], pts[i_eq, 1]]
    p0.append(heading[-1])
    res = least_squares(residuals, np.array(p0, float), x_scale="jac", max_nfev=400)
    d_final, _ = dense(res.x)
    rms = float(np.sqrt(np.mean(cKDTree(d_final).query(pts)[0] ** 2)))
    return rms, float(np.degrees(res.x[-1] - true_needle_heading))


def side_chain_study() -> dict:
    out = {}
    for bond in (0.2, 0.4):
        p = scaled(bond)
        step = max(1, int(1.0 / (p["s"][1] - p["s"][0])))
        pts = np.column_stack([p["x"], p["z"]])[::step]
        pts = np.vstack([pts, [p["x"][-1], p["z"][-1]]])
        apex, needle = pts[0], pts[-1]
        eq = np.array([p["x"][p["i_eq"]], p["z"][p["i_eq"]]])
        row = {}
        for n1, n2 in ((1, 1), (1, 2), (2, 2), (2, 3)):
            rms, err = fit_side_chain(pts, apex, needle, eq, n1, n2, float(p["phi"][-1]))
            row[f"{n1}+{n2}"] = {"rms_px": rms, "needle_heading_err_deg": err}
        out[f"bo{bond}"] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = {"zone_counts_tol_0.25px": zone_counts(), "zone1_bond": zone1_bond_recovery(),
              "zone_information": zone_information(), "side_chain": side_chain_study()}
    print(f"Segments per zone and side, tolerance 0.25 px, equatorial radius {EQUATOR_RADIUS_PX:.0f} px:")
    for k, v in report["zone_counts_tol_0.25px"].items():
        print(f"  {k}: zone1 arcs {v['zone1_arcs']:.1f} / clothoids {v['zone1_clothoids']:.1f} | "
              f"zone2 arcs {v['zone2_arcs']:.1f} / clothoids {v['zone2_clothoids']:.1f} | "
              f"inflection {v['zone2_has_inflection']} | needle angle {v['needle_angle_deg']:.1f} deg | "
              f"height {v['height_px']:.0f} px")
    zb = report["zone1_bond"]
    print(f"\nZone-1 box ratio H_e/R_e monotonic in Bond: {zb['box_ratio_is_monotonic']}")
    for k, v in zb.items():
        if isinstance(v, dict):
            print(f"  {k}: Bond {v['bond_mean']:.4f} +- {v['bond_sd']:.4f}, "
                  f"surface tension rel. error {100 * v['gamma_rel_err_mean']:+.2f} %")
    print("\nBond number from zone 1 alone vs zones 1+2 (mean +- sd over seeds):")
    for k, v in report["zone_information"].items():
        print(f"  {k}: zone1 {v['zone1']['bond_mean']:.3f} +- {v['zone1']['bond_sd']:.3f} | "
              f"zones1+2 {v['zones12']['bond_mean']:.3f} +- {v['zones12']['bond_sd']:.3f}")
    print("\nClothoid chain per side (apex -> equator [vertical tangent] -> needle):")
    for k, row in report["side_chain"].items():
        print(f"  {k}: " + ", ".join(
            f"{n}: rms {v['rms_px']:.3f} px, needle angle err {v['needle_heading_err_deg']:+.2f} deg"
            for n, v in row.items()))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
