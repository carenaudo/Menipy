"""Profile the two-zone pendant clothoid spline and the strict fit it seeds.

Per case: wall time of the spline fit and where it goes (anchored box search,
spline least squares, foot-point quadrature, model bias fit, symmetry axis),
the image refinement on a rendered drop, and the strict Young-Laplace fit on
the raw contour versus on the spline's samples with the spline's seeds.

Usage
-----
``uv run python tools/profile_pendant_spline.py [--repeats 3] [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from menipy.common import clothoid_spline, pendant_spline  # noqa: E402
from menipy.math import pendant_box  # noqa: E402
from menipy.pipelines.pendant import strict_young_laplace as strict  # noqa: E402
from menipy.pipelines.pendant.stages import (  # noqa: E402
    _clip_contour_at_pendant_contacts,
)
from tests.synthetic_pendant import pendant_case, render_pendant  # noqa: E402
from tests.synthetic_sessile import mask_contour  # noqa: E402

KW = {"px_per_mm": 100.0, "delta_rho": 998.8, "g": 9.80665}
PHYSICS = {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
CASES = [
    ("bo0.1_clean", 0.1, "clean"),
    ("bo0.3_clean", 0.3, "clean"),
    ("bo0.2_gauss0.5", 0.2, "gauss0.5"),
    ("bo0.3_gauss1", 0.3, "gauss1.0"),
    ("bo0.4_quantized", 0.4, "quantized"),
]
STAGES = [
    (pendant_spline, "_symmetry_slope", "symmetry axis"),
    (pendant_spline, "fit_pendant_box", "box golden search"),
    (pendant_spline, "_refit_zones", "least squares"),
    (clothoid_spline._Problem, "_evaluate", "residual eval"),
    (clothoid_spline._Problem, "jacobian", "jacobian"),
    (clothoid_spline, "_curve", "foot-point quadrature"),
    (pendant_spline, "_model_bias", "model bias fit"),
    (strict, "integrate_young_laplace_profile_mm", "strict ODE integration"),
]


class StageTimer:
    """Accumulate call counts and inclusive wall time of wrapped callables."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def wrap(self, owner, attr, label):
        original = getattr(owner, attr)

        def inner(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[label] += time.perf_counter() - start
                self.calls[label] += 1

        setattr(owner, attr, inner)
        try:
            yield
        finally:
            setattr(owner, attr, original)


def timed(fn, stages=STAGES):
    timer = StageTimer()
    with ExitStack() as stack:
        for owner, attr, label in stages:
            stack.enter_context(timer.wrap(owner, attr, label))
        start = time.perf_counter()
        out = fn()
        total = time.perf_counter() - start
    parts = {k: {"calls": timer.calls[k], "seconds": timer.seconds[k], "share": timer.seconds[k] / total}
             for _, _, k in stages if timer.calls[k]}
    return out, total, parts


def strict_fit(xy, fit=None):
    if fit is None:
        i = int(np.argmax(xy[:, 1]))
        kwargs = {"axis_x_px": float(np.median(xy[:, 0])), "apex_y_px": float(xy[i, 1]),
                  "r0_seed_mm": 1.2, "beta_seed": 0.3}
    else:
        point, up = fit.axis_image
        kwargs = {"axis_x_px": float(point[0]), "apex_y_px": float(point[1]),
                  "r0_seed_mm": fit.shape.apex_radius_px / KW["px_per_mm"], "beta_seed": fit.shape.bond,
                  "axis_origin_px": tuple(point), "axis_direction_xy": tuple(up)}
        xy = fit.sample(3.0)
    return strict.fit_pendant_young_laplace_strict(strict.PendantStrictFitInput(
        contour_px=xy, px_per_mm=KW["px_per_mm"], physics=PHYSICS, **kwargs))


def _line(parts) -> str:
    return ", ".join(f"{k}={v['share'] * 100:4.1f}% ({v['calls']}x)" for k, v in parts.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    pendant_box.pendant_table()
    report = {"table_build_s": time.perf_counter() - start, "cases": {}}
    print(f"one-off pendant table build: {report['table_build_s'] * 1e3:.0f} ms (cached per process)\n")
    for name, bond, noise in CASES:
        pts, p1, p2 = pendant_case(bond, noise, seed=7, tilt_deg=4.0)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fit = pendant_spline.fit_pendant_spline(pts, p1, p2, **KW)
            times.append(time.perf_counter() - t0)
        _, _, parts = timed(lambda q=pts, a=p1, b=p2: pendant_spline.fit_pendant_spline(q, a, b, **KW))
        raw, t_raw, raw_parts = timed(lambda q=pts: strict_fit(q))
        seeded, t_seeded, seeded_parts = timed(lambda q=pts, f=fit: strict_fit(q, f))
        report["cases"][name] = {
            "n_points": len(pts), "spline_median_s": median(times), "zones": [fit.zones_p1, fit.zones_p2],
            "spline_stages": parts, "strict_raw_s": t_raw, "strict_raw_nfev": raw["solver"]["iterations"],
            "strict_seeded_s": t_seeded, "strict_seeded_nfev": seeded["solver"]["iterations"],
            "strict_raw_stages": raw_parts, "strict_seeded_stages": seeded_parts,
        }
        print(f"{name:16s} pts={len(pts):4d} zones={fit.zones_p1}/{fit.zones_p2} spline={median(times) * 1e3:6.1f} ms"
              f" | strict raw {t_raw * 1e3:6.1f} ms ({raw['solver']['iterations']} nfev)"
              f" -> seeded {t_seeded * 1e3:6.1f} ms ({seeded['solver']['iterations']} nfev)")
        print(f"  spline: {_line(parts)}")
        print(f"  strict raw: {_line(raw_parts)}")

    print("\nRendered image: mask contour fit, then contrast refinement (1 round):")
    image, p1, p2 = render_pendant(0.3, seed=0, tilt_deg=2.0)
    contour = _clip_contour_at_pendant_contacts(mask_contour(image), np.array([p1, p2]))
    (fit, _), t_mask, mask_parts = timed(lambda: pendant_spline.fit_pendant_contour(contour, p1, p2, **KW))
    (refined, _), t_ref, ref_parts = timed(lambda: pendant_spline.refine_pendant_on_image(image, fit, p1, p2, **KW))
    report["image_case"] = {"mask_fit_s": t_mask, "refine_s": t_ref, "mask_stages": mask_parts,
                            "refine_stages": ref_parts}
    print(f"  mask fit {t_mask * 1e3:.1f} ms ({_line(mask_parts)})")
    print(f"  refinement {t_ref * 1e3:.1f} ms ({_line(ref_parts)})")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
