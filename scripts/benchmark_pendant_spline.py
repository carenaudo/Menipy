"""Benchmark: two-zone clothoid spline versus the strict Young-Laplace fit (pendant).

Surface tension, needle angle and time on

* exact synthetic edges (independent ``solve_ivp`` profiles) with Gaussian or
  pixel-staircase noise, several seeds;
* rendered images (anti-aliased drop on a needle, blur, sensor noise) through
  the pipeline's binary-mask contour, with and without the spline's sub-pixel
  image refinement;
* the repository's real pendant samples, when detection finds drop and needle.

Methods

* ``zone_box``: anchored Young-Laplace profile + golden-section Bond search;
* ``zone_laplace``: slope of the spline's mean curvature against height;
* ``strict``: the pipeline's strict Young-Laplace fit on the raw contour;
* ``strict_seeded``: the same fit on the spline's samples (3 px apart), seeded
  with the box's Bond number, apex radius and symmetry axis.

Usage
-----
``uv run python scripts/benchmark_pendant_spline.py [--seeds 5] [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from menipy.common.pendant_spline import (  # noqa: E402
    fit_pendant_contour,
    fit_pendant_spline,
    refine_pendant_on_image,
)
from menipy.models.context import Context  # noqa: E402
from menipy.models.geometry import Contour  # noqa: E402
from menipy.pipelines.pendant.stages import (  # noqa: E402
    PendantPipeline,
    _clip_contour_at_pendant_contacts,
)
from menipy.pipelines.pendant.strict_young_laplace import (  # noqa: E402
    PendantStrictFitInput,
    fit_pendant_young_laplace_strict,
)
from tests.synthetic_pendant import (  # noqa: E402
    pendant_case,
    pendant_truth,
    render_pendant,
)
from tests.synthetic_sessile import mask_contour  # noqa: E402

PX_PER_MM = 100.0
PHYSICS = {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
DRHO = PHYSICS["rho1"] - PHYSICS["rho2"]
BONDS = (0.1, 0.2, 0.3, 0.4)


def true_gamma(bond: float) -> float:
    b_m = pendant_truth(bond)["apex_radius_px"] / PX_PER_MM / 1000.0
    return DRHO * PHYSICS["g"] * b_m**2 / bond * 1000.0


def strict_pipeline(xy: np.ndarray, contacts=None, px_per_mm: float = PX_PER_MM, physics=PHYSICS):
    ctx = Context(contour=Contour(xy=xy), scale={"px_per_mm": px_per_mm}, physics=physics,
                  pendant_approximation_methods=[])
    if contacts is not None:
        ctx.contact_points = tuple(tuple(int(round(v)) for v in c) for c in contacts)
    pipe = PendantPipeline()
    t = time.perf_counter()
    pipe.do_geometric_features(ctx)
    pipe.do_profile_fitting(ctx)
    return ctx.fit, time.perf_counter() - t


def strict_seeded(fit, px_per_mm: float = PX_PER_MM, physics=PHYSICS, step_px: float = 3.0):
    point, up = fit.axis_image
    t = time.perf_counter()
    out = fit_pendant_young_laplace_strict(PendantStrictFitInput(
        contour_px=fit.sample(step_px), axis_x_px=float(point[0]), apex_y_px=float(point[1]),
        px_per_mm=px_per_mm, r0_seed_mm=fit.shape.apex_radius_px / px_per_mm,
        beta_seed=fit.shape.bond, physics=physics,
        axis_origin_px=(float(point[0]), float(point[1])), axis_direction_xy=(float(up[0]), float(up[1]))))
    return out, time.perf_counter() - t


def strict_needle_angle(fit_dict) -> float:
    """Tangent angle at the top of the strict model profile, degrees."""
    prof = np.asarray(fit_dict.get("model_radial_profile_mm") or [], float)
    if len(prof) < 3:
        return float("nan")
    d = prof[-1] - prof[-3]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def rel(value: float, truth: float) -> float:
    return 100.0 * (float(value) / truth - 1.0)


def summarize(rows: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    if v.size == 0:
        return {"rmse": float("nan"), "bias": float("nan"), "n": 0}
    return {"rmse": float(np.sqrt(np.mean(v**2))), "bias": float(np.mean(v)), "n": int(v.size)}


def synthetic(seeds: int) -> list[dict]:
    rows = []
    for bond in BONDS:
        gt, tr = true_gamma(bond), pendant_truth(bond)
        for noise in ("clean", "quantized", "gauss0.5", "gauss1.0"):
            for seed in range(1 if noise == "clean" else seeds):
                pts, p1, p2 = pendant_case(bond, noise, seed=seed + 1)
                t = time.perf_counter()
                f = fit_pendant_spline(pts, p1, p2, px_per_mm=PX_PER_MM, delta_rho=DRHO, g=PHYSICS["g"])
                t_zone = time.perf_counter() - t
                sf, t_strict = strict_pipeline(pts)
                ss, t_seeded = strict_seeded(f)
                rows.append({
                    "bond": bond, "noise": noise, "seed": seed,
                    "zone_box": rel(f.surface_tension_mN_m, gt),
                    "zone_laplace": rel(f.laplace["surface_tension_mN_m"], gt),
                    "zone_laplace_raw": rel(f.laplace["surface_tension_raw_mN_m"], gt),
                    "strict": rel(sf["strict_surface_tension_mN_m"], gt),
                    "strict_seeded": rel(ss["strict_surface_tension_mN_m"], gt),
                    "needle_zone": 0.5 * (f.needle_angle_p1_deg + f.needle_angle_p2_deg) - tr["needle_angle_deg"],
                    "needle_zone_raw": 0.5 * (f.needle_angle_p1_raw_deg + f.needle_angle_p2_raw_deg) - tr["needle_angle_deg"],
                    "needle_strict": strict_needle_angle(sf) - tr["needle_angle_deg"],
                    "sigma_needle": 0.5 * (f.sigma_p1_deg + f.sigma_p2_deg),
                    "zones": [list(f.zones_p1), list(f.zones_p2)],
                    "ms_zone": 1e3 * t_zone, "ms_strict": 1e3 * t_strict, "ms_seeded": 1e3 * t_seeded,
                    "nfev_strict": sf["solver"]["iterations"], "nfev_seeded": ss["solver"]["iterations"],
                    "n_raw": len(pts), "n_seeded": len(f.sample(3.0)),
                    "accepted": f.accepted,
                })
    return rows


def rendered(seeds: int) -> list[dict]:
    rows = []
    kw = {"px_per_mm": PX_PER_MM, "delta_rho": DRHO, "g": PHYSICS["g"]}
    for bond in BONDS:
        gt, tr = true_gamma(bond), pendant_truth(bond)
        for tilt in (0.0, 3.0):
            for seed in range(seeds):
                img, p1, p2 = render_pendant(bond, tilt_deg=tilt, seed=seed + 1)
                contour = _clip_contour_at_pendant_contacts(mask_contour(img), np.array([p1, p2]))
                sf, t_strict = strict_pipeline(contour, contacts=(p1, p2))
                t = time.perf_counter()
                f, _ = fit_pendant_contour(contour, p1, p2, **kw)
                t_zone = time.perf_counter() - t
                t = time.perf_counter()
                fr, _ = refine_pendant_on_image(img, f, p1, p2, **kw)
                t_refine = time.perf_counter() - t
                ss, t_seeded = strict_seeded(fr)
                rows.append({
                    "bond": bond, "tilt": tilt, "seed": seed,
                    "strict_mask": rel(sf["strict_surface_tension_mN_m"], gt),
                    "zone_box_mask": rel(f.surface_tension_mN_m, gt),
                    "zone_box_refined": rel(fr.surface_tension_mN_m, gt),
                    "zone_laplace_refined": rel(fr.laplace["surface_tension_mN_m"], gt),
                    "strict_seeded_refined": rel(ss["strict_surface_tension_mN_m"], gt),
                    "needle_zone_mask": f.needle_angle_deg - tr["needle_angle_deg"],
                    "needle_zone_refined": fr.needle_angle_deg - tr["needle_angle_deg"],
                    "needle_strict_mask": strict_needle_angle(sf) - tr["needle_angle_deg"],
                    "axis_tilt_deg": fr.axis_tilt_deg,
                    "ms_strict": 1e3 * t_strict, "ms_zone": 1e3 * t_zone, "ms_refine": 1e3 * t_refine,
                    "ms_seeded": 1e3 * t_seeded, "accepted": fr.accepted,
                })
    return rows


def contact_errors(seeds: int) -> list[dict]:
    """Needle angle and surface tension when the contact points are off the edge.

    ``Context.contact_points`` holds integers, and detected contacts are off
    by a pixel or so; the spline either passes through them or lets them slide
    sideways at fixed height.
    """
    rows = []
    rng = np.random.default_rng(7)
    for bond in BONDS:
        gt, tr = true_gamma(bond), pendant_truth(bond)
        for seed in range(seeds):
            pts, p1, p2 = pendant_case(bond, "gauss0.5", seed=seed + 1)
            for kind in ("rounded", "sideways_1px"):
                q1, q2 = np.round(p1), np.round(p2)
                if kind == "sideways_1px":
                    q1 = q1 + [rng.choice([-1.0, 1.0]), 0.0]
                    q2 = q2 + [rng.choice([-1.0, 1.0]), 0.0]
                for slide in (0.0, 2.0):
                    f = fit_pendant_spline(pts, q1, q2, px_per_mm=PX_PER_MM, delta_rho=DRHO, g=PHYSICS["g"],
                                           max_contact_slide_px=slide)
                    rows.append({"bond": bond, "seed": seed, "kind": f"{kind}_slide{slide:g}",
                                 "needle": f.needle_angle_deg - tr["needle_angle_deg"],
                                 "gamma": rel(f.surface_tension_mN_m, gt)})
    return rows


REAL_SAMPLES = {
    "data/samples/gota pendiente 1.png": 29.24,
    "data/samples/prueba pend 1.png": 70.94,
}


def real_samples(needle_mm: float = 1.83) -> list[dict]:
    import cv2

    from menipy.common.auto_calibrator import AutoCalibrator

    rows = []
    kw_phys = {"delta_rho": DRHO, "g": PHYSICS["g"]}
    for path, reference in REAL_SAMPLES.items():
        img = cv2.imread(str(ROOT / path))
        row = {"sample": path, "reference_mN_m": reference}
        if img is None:
            rows.append({**row, "error": "image not found"})
            continue
        cal = AutoCalibrator(img, "pendant").detect_all()
        if cal.needle_rect is None or cal.drop_contour is None or cal.contact_points is None:
            rows.append({**row, "error": "detection failed"})
            continue
        px_per_mm = float(cal.needle_rect[2]) / needle_mm
        contour = np.asarray(cal.drop_contour, float).reshape(-1, 2)
        contacts = np.asarray(cal.contact_points, float).reshape(-1, 2)[:2]
        clipped = _clip_contour_at_pendant_contacts(contour, contacts)
        sf, t_strict = strict_pipeline(clipped, contacts=contacts, px_per_mm=px_per_mm)
        row.update({"px_per_mm": px_per_mm, "strict_mN_m": sf.get("strict_surface_tension_mN_m"),
                    "strict_ok": sf.get("strict_fit_success"), "ms_strict": 1e3 * t_strict})
        try:
            t = time.perf_counter()
            f, _ = fit_pendant_contour(clipped, contacts[0], contacts[1], px_per_mm=px_per_mm, **kw_phys)
            fr, _ = refine_pendant_on_image(img, f, contacts[0], contacts[1], px_per_mm=px_per_mm, **kw_phys)
            row["ms_zone_refined"] = 1e3 * (time.perf_counter() - t)
            ss, t_seeded = strict_seeded(fr, px_per_mm=px_per_mm)
            row.update({
                "zone_box_mask_mN_m": f.surface_tension_mN_m,
                "zone_box_refined_mN_m": fr.surface_tension_mN_m,
                "zone_laplace_refined_mN_m": fr.laplace["surface_tension_mN_m"],
                "strict_seeded_mN_m": ss["strict_surface_tension_mN_m"], "ms_seeded": 1e3 * t_seeded,
                "needle_angle_deg": fr.needle_angle_deg, "sigma_needle_deg": 0.5 * (fr.sigma_p1_deg + fr.sigma_p2_deg),
                "bond": fr.shape.bond, "zones": [list(fr.zones_p1), list(fr.zones_p2)],
                "edge_noise_px": fr.edge_noise_px, "rmse_px": fr.rmse_px, "axis_tilt_deg": fr.axis_tilt_deg,
                "rejection_reasons": fr.rejection_reasons,
            })
        except ValueError as exc:
            row["zone_error"] = str(exc)
        rows.append(row)
    return rows


def _print_table(title: str, rows: list[dict], keys: list[str], group: str) -> None:
    print(f"\n{title}")
    groups = sorted({r[group] for r in rows}, key=str)
    print(f"{group:>10s} " + " ".join(f"{k:>22s}" for k in keys))
    for gv in groups:
        sub = [r for r in rows if r[group] == gv]
        cells = []
        for k in keys:
            s = summarize(sub, k)
            cells.append(f"{s['rmse']:9.2f} ({s['bias']:+6.2f})")
        print(f"{str(gv):>10s} " + " ".join(f"{c:>22s}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    syn = synthetic(args.seeds)
    gamma_keys = ["zone_box", "zone_laplace", "strict", "strict_seeded"]
    _print_table("Synthetic edges: surface-tension error %, RMSE (bias), by noise", syn, gamma_keys, "noise")
    _print_table("Synthetic edges: surface-tension error %, RMSE (bias), by Bond", syn, gamma_keys, "bond")
    _print_table("Synthetic edges: needle-angle error deg, RMSE (bias)", syn,
                 ["needle_zone", "needle_zone_raw", "needle_strict", "sigma_needle"], "noise")
    for k in ("ms_zone", "ms_strict", "ms_seeded", "nfev_strict", "nfev_seeded", "n_raw", "n_seeded"):
        print(f"  median {k}: {np.median([r[k] for r in syn]):.1f}")
    print(f"  accepted: {sum(r['accepted'] for r in syn)}/{len(syn)}")

    con = contact_errors(args.seeds)
    _print_table("Contact points off the edge (0.5 px noise): error, RMSE (bias)", con, ["needle", "gamma"], "kind")

    ren = rendered(max(1, args.seeds // 2 + 1))
    _print_table("Rendered images: surface-tension error %, RMSE (bias), by Bond", ren,
                 ["strict_mask", "zone_box_mask", "zone_box_refined", "zone_laplace_refined", "strict_seeded_refined"], "bond")
    _print_table("Rendered images: needle-angle error deg, RMSE (bias), by tilt", ren,
                 ["needle_strict_mask", "needle_zone_mask", "needle_zone_refined"], "tilt")
    for k in ("ms_strict", "ms_zone", "ms_refine", "ms_seeded"):
        print(f"  median {k}: {np.median([r[k] for r in ren]):.1f}")
    print(f"  accepted: {sum(r['accepted'] for r in ren)}/{len(ren)}")

    real = [] if args.skip_real else real_samples()
    if real:
        print("\nReal samples (reference from a commercial instrument):")
        for r in real:
            print("  " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    if args.json:
        args.json.write_text(json.dumps({"synthetic": syn, "contacts": con, "rendered": ren, "real": real},
                                        indent=2, default=float))


if __name__ == "__main__":
    main()
