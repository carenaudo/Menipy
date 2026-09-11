"""Benchmark the arc-spline contact angle against Menipy's estimators.

Reproducible evidence for ``docs/research/arc_spline_contour.md``. Three parts:

1. Accuracy sweep on exact Young-Laplace profiles (Bond 0-8, 30-150 deg,
   clean / binary staircase / Gaussian edge noise, random tilt): the existing
   ``tangent`` and ``circle_fit`` estimators, the blind arc spline
   (coarse-to-fine + Richardson), the physics arc spline (box-matched
   Young-Laplace prior + model bias correction) and the box-matched
   Young-Laplace angle itself.
2. Drops that are *not* one Young-Laplace profile: asymmetric sides and a
   pinning bump on one flank. The physics angle alone is then wrong; the arcs
   should not be.
3. Arcs versus clothoids (linear curvature per segment): raw end-tangent error
   against segments per side on clean profiles, without any bias correction.

This script measures; it changes no application behavior.

Usage
-----
``uv run python scripts/benchmark_arc_spline.py [--seeds 3] [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from menipy.common import arc_spline as A  # noqa: E402
from menipy.common.clothoid_spline import fit_clothoid_spline  # noqa: E402
from menipy.common.geometry import (  # noqa: E402
    circle_fit_angle_at_point,
    tangent_angle_at_point,
)
from menipy.math.sessile_box import profile_table  # noqa: E402
from tests.synthetic_sessile import (  # noqa: E402
    closed_silhouette,
    degrade,
    half_profile,
    place,
    sessile_case,
)

BONDS = (0.0, 0.5, 2.0, 8.0)
THETAS = (30.0, 90.0, 150.0)
NOISES = ("clean", "quantized", "gauss0.5", "gauss1.0")


def menipy_angles(points, p1, p2):
    contour = closed_silhouette(points, p1, p2)
    line = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
    window = 30 if len(contour) > 200 else 15
    power = 2.0 if len(contour) > 200 else 4.0
    tangent = [tangent_angle_at_point(contour, p, line, window, power)[0] for p in (p1, p2)]
    circle = [circle_fit_angle_at_point(contour, p, line)[0] for p in (p1, p2)]
    return tangent, circle


def run_methods(points, p1, p2) -> dict[str, tuple[list[float], float]]:
    out = {}
    t0 = time.perf_counter()
    tangent, circle = menipy_angles(points, p1, p2)
    t_menipy = (time.perf_counter() - t0) / 2
    out["menipy_tangent"] = (tangent, t_menipy)
    out["menipy_circle_fit"] = (circle, t_menipy)
    t0 = time.perf_counter()
    blind = A.fit_arc_spline(points, p1, p2, init="blind", bias_correction="richardson")
    out["arcs_blind"] = ([blind.theta_p1_deg, blind.theta_p2_deg], time.perf_counter() - t0)
    t0 = time.perf_counter()
    phys = A.fit_arc_spline(points, p1, p2)
    dt = time.perf_counter() - t0
    out["arcs_physics"] = ([phys.theta_p1_deg, phys.theta_p2_deg], dt)
    if phys.physics is not None:
        out["young_laplace_box"] = ([phys.physics["p1"]["theta_deg"], phys.physics["p2"]["theta_deg"]], dt)
    t0 = time.perf_counter()
    clo = fit_clothoid_spline(points, p1, p2)
    out["clothoids_physics"] = ([clo.theta_p1_deg, clo.theta_p2_deg], time.perf_counter() - t0)
    out["_arcs"] = ([phys.n_arcs, blind.n_arcs, clo.n_segments], 0.0)
    return out


def accuracy_sweep(seeds: int) -> dict:
    rng = np.random.default_rng(2026)
    errors = defaultdict(lambda: defaultdict(list))
    times = defaultdict(list)
    arcs = defaultdict(list)
    clothoid_counts = defaultdict(list)
    for bond in BONDS:
        for theta in THETAS:
            for noise in NOISES:
                for seed in range(1 if noise == "clean" else seeds):
                    tilt = float(rng.uniform(-8.0, 8.0))
                    pts, p1, p2 = sessile_case(bond, theta, noise, seed=seed, tilt_deg=tilt)
                    for method, (angles, dt) in run_methods(pts, p1, p2).items():
                        if method == "_arcs":
                            arcs[(bond, theta)].append(angles[0])
                            clothoid_counts[(bond, theta)].append(angles[2])
                            continue
                        errors[method][noise].extend(a - theta for a in angles)
                        errors[method][f"theta{int(theta)}"].extend(a - theta for a in angles)
                        times[method].append(dt)
    rms = {m: {g: float(np.sqrt(np.mean(np.square(v)))) for g, v in groups.items()} for m, groups in errors.items()}
    return {
        "rmse_deg": rms,
        "median_ms": {m: 1e3 * float(np.median(v)) for m, v in times.items()},
        "physics_arcs": {f"bo{b}_t{int(t)}": float(np.mean(v)) for (b, t), v in arcs.items()},
        "physics_clothoids": {f"bo{b}_t{int(t)}": float(np.mean(v)) for (b, t), v in clothoid_counts.items()},
    }


def bumped_case(bond, theta, amplitude, seed):
    """Young-Laplace profile with a smooth pinning bump on the P2 flank."""
    pts, p1, p2 = sessile_case(bond, theta, "clean", seed=seed)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    total = s[-1]
    center, width = 0.8 * total, 0.04 * total  # bump well clear of the contact
    d = np.gradient(pts, axis=0)
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(*d.T)[:, None]
    bump = amplitude * np.exp(-(((s - center) / width) ** 2))
    bumped = pts + bump[:, None] * normal
    noisy = degrade(bumped, "gauss0.5", np.random.default_rng(seed))
    return noisy, p1, p2


def asymmetric_case(theta_left, theta_right, seed, bond=1.0, height=150.0):
    halves = []
    for theta in (theta_left, theta_right):
        h = half_profile(bond, theta, 100.0)
        h = h * (height / h[-1, 1])
        steps = np.hypot(*np.diff(h, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        grid = np.append(np.arange(0.0, s[-1], 1.0), s[-1])
        halves.append(np.column_stack([np.interp(grid, s, h[:, 0]), np.interp(grid, s, h[:, 1])]))
    exact = np.vstack([halves[0][::-1] * [-1.0, 1.0], halves[1][1:]])
    pts = place(degrade(exact, "gauss0.5", np.random.default_rng(seed)), height)
    p1, p2 = place(exact[[0, -1]], height)
    return pts, p1, p2


def non_young_laplace(seeds: int) -> dict:
    out = {}
    for label, maker, truth in (
        ("asymmetric_70_115", lambda s: asymmetric_case(70.0, 115.0, s), (70.0, 115.0)),
        ("bump_4px_bo2_t100", lambda s: bumped_case(2.0, 100.0, 4.0, s), (100.0, 100.0)),
    ):
        errs = defaultdict(list)
        for seed in range(seeds):
            pts, p1, p2 = maker(seed)
            for method, (angles, _) in run_methods(pts, p1, p2).items():
                if method != "_arcs":
                    errs[method].extend(a - t for a, t in zip(angles, truth))
        out[label] = {m: float(np.sqrt(np.mean(np.square(v)))) for m, v in errs.items()}
    return out


# ----------------------------------------------------------------------------
# Clothoid (linear-curvature) chains versus arcs, raw end-tangent error
# ----------------------------------------------------------------------------


def _clothoid_side(apex, heading0, kappa_knots, length, samples=800):
    """Chain with curvature linear between equally spaced knots; dense points.

    A fixed sample count keeps the residual vector length constant while the
    optimizer changes ``length``.
    """
    n_seg = len(kappa_knots) - 1
    s = np.linspace(0.0, max(length, 1.0), samples)
    kappa = np.interp(s, np.linspace(0.0, length, n_seg + 1), kappa_knots)
    heading = heading0 + np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(s))])
    ds = np.diff(s)
    mid = 0.5 * (heading[1:] + heading[:-1])
    xy = np.vstack([[0.0, 0.0], np.cumsum(np.column_stack([np.cos(mid) * ds, np.sin(mid) * ds]), axis=0)])
    return apex + xy, heading[-1]


def fit_clothoids(q, lp1, lp2, n_seg, apex_guess):
    """Fit apex (x, y) plus per-side curvature knots and lengths; end on the contacts."""
    tree = cKDTree(q)

    def unpack(p):
        apex = p[:2]
        k1 = p[2 : 3 + n_seg]
        k2 = np.concatenate([[p[2]], p[3 + n_seg : 3 + 2 * n_seg]])  # shared apex curvature
        return apex, k1, k2, p[-2], p[-1]

    def residuals(p):
        apex, k1, k2, len1, len2 = unpack(p)
        pts1, h1 = _clothoid_side(apex, np.pi, -k1, len1)
        pts2, h2 = _clothoid_side(apex, 0.0, k2, len2)
        dense = np.vstack([pts1[::-1], pts2[1:]])
        d_data = cKDTree(dense).query(q)[0]
        d_back = tree.query(dense[::4])[0]
        return np.concatenate([d_data, 0.5 * d_back, 50.0 * (pts1[-1] - lp1), 50.0 * (pts2[-1] - lp2)])

    height = -apex_guess[1]
    k0 = 1.0 / max(height, 1.0)
    side_len = [np.hypot(*(apex_guess - lp1)) * 1.3, np.hypot(*(apex_guess - lp2)) * 1.3]
    p0 = np.concatenate([apex_guess, np.full(n_seg + 1, k0), np.full(n_seg, k0), side_len])
    res = least_squares(residuals, p0, x_scale="jac", max_nfev=400)
    apex, k1, k2, len1, len2 = unpack(res.x)
    _, h1 = _clothoid_side(apex, np.pi, -k1, len1)
    _, h2 = _clothoid_side(apex, 0.0, k2, len2)
    return np.degrees(np.pi - h1), np.degrees(h2)


def arcs_versus_clothoids() -> dict:
    table = {}
    for bond, theta in ((2.0, 90.0), (8.0, 150.0), (0.5, 120.0)):
        pts, p1, p2 = sessile_case(bond, theta)
        frame = A._Frame.from_contacts(p1, p2, pts)
        q = frame.to_local(pts)
        lp1, lp2 = frame.to_local(p1)[0], frame.to_local(p2)[0]
        apex = q[np.argmin(q[:, 1])]
        row = {}
        for n in (1, 2, 3, 4, 6):
            arc = A.fit_arc_spline(pts, p1, p2, init="blind", bias_correction="off", max_arcs=2 * n, patience=99)
            # the blind run ends at n arcs per side; its last level is the n-arc fit
            lv = arc.levels[-1]
            clo = fit_clothoids(q[::2], lp1, lp2, n, apex.copy())
            row[n] = {
                "arcs_raw_err_deg": float(np.mean([lv.theta_p1_deg - theta, lv.theta_p2_deg - theta])),
                "clothoid_raw_err_deg": float(np.mean([clo[0] - theta, clo[1] - theta])),
            }
        table[f"bo{bond}_t{int(theta)}"] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-clothoids", action="store_true")
    args = parser.parse_args()
    profile_table()

    report = {"accuracy": accuracy_sweep(args.seeds), "non_young_laplace": non_young_laplace(args.seeds)}
    acc = report["accuracy"]
    groups = list(NOISES) + [f"theta{int(t)}" for t in THETAS]
    print("RMSE (deg) by noise model and by contact angle; median time per fit")
    print(f"{'method':20s} " + " ".join(f"{g:>10s}" for g in groups) + f" {'ms':>8s}")
    for method, row in acc["rmse_deg"].items():
        print(f"{method:20s} " + " ".join(f"{row[g]:10.2f}" for g in groups) + f" {acc['median_ms'][method]:8.1f}")
    print("\nPhysics-chosen arcs / clothoids (both sides), mean over noise models:")
    print("  " + ", ".join(
        f"{k}: {v:.1f}/{acc['physics_clothoids'][k]:.1f}" for k, v in acc["physics_arcs"].items()
    ))
    print("\nDrops that are not a single Young-Laplace profile (RMSE deg, 0.5 px noise):")
    for label, row in report["non_young_laplace"].items():
        print(f"  {label:20s} " + ", ".join(f"{m}={v:.2f}" for m, v in row.items()))

    if not args.skip_clothoids:
        report["arcs_vs_clothoids"] = arcs_versus_clothoids()
        print("\nRaw end-tangent error (deg, no bias correction) vs segments per side:")
        for label, row in report["arcs_vs_clothoids"].items():
            cells = ", ".join(
                f"n={n}: arcs {v['arcs_raw_err_deg']:+.2f} / clothoids {v['clothoid_raw_err_deg']:+.2f}"
                for n, v in row.items()
            )
            print(f"  {label}: {cells}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
