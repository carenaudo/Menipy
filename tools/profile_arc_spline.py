"""Profile the arc-spline contact-angle fit and show where the time goes.

Fits exact Young-Laplace sessile profiles (clean, Gaussian edge noise and a
binary-mask staircase, tilted 7 degrees) with :mod:`menipy.common.arc_spline`
in both initialization modes and reports:

* wall time per case and mode (median of repeats), arcs used and angle error;
* a stage breakdown from timing wrappers: optimizer refits, residual and
  Jacobian evaluations, chain construction, the physics prior (box-matched
  Young-Laplace golden-section search) and the model bias fit;
* a rendered-image case: mask-contour fit plus contrast refinement;
* the cProfile top functions for the slowest case.

This tool measures; it changes no application behavior.

Usage
-----
``uv run python tools/profile_arc_spline.py [--repeats 3] [--json report.json]``
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from menipy.common import arc_spline, clothoid_spline  # noqa: E402
from menipy.math import sessile_box  # noqa: E402
from tests.synthetic_sessile import (  # noqa: E402
    mask_contour,
    render_image,
    sessile_case,
)

CASES = [
    ("sphere_clean", 0.0, 90.0, "clean"),
    ("bo2_t90_clean", 2.0, 90.0, "clean"),
    ("bo8_t150_clean", 8.0, 150.0, "clean"),
    ("bo2_t90_gauss1", 2.0, 90.0, "gauss1.0"),
    ("bo0.5_t30_quantized", 0.5, 30.0, "quantized"),
    ("bo8_t150_gauss0.5", 8.0, 150.0, "gauss0.5"),
]
MODES = ("blind", "physics")

# (owner, attribute, label): module functions and class methods to time
STAGES = [
    (arc_spline, "_refit", "refit (least_squares)"),
    (arc_spline._Problem, "_evaluate", "residual eval"),
    (arc_spline._Problem, "jacobian", "jacobian"),
    (arc_spline, "_build_chain", "chain build"),
    (arc_spline, "_physics_layout", "physics prior"),
    (sessile_box, "fit_side_box", "box golden search"),
    (arc_spline, "_model_bias_deg", "model bias fit"),
    (arc_spline, "sample_edge_along_normals", "normal profiles"),
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


def breakdown(pts, p1, p2, mode) -> dict:
    timer = StageTimer()
    with ExitStack() as stack:
        for owner, attr, label in STAGES:
            stack.enter_context(timer.wrap(owner, attr, label))
        start = time.perf_counter()
        arc_spline.fit_arc_spline(pts, p1, p2, init=mode)
        total = time.perf_counter() - start
    return {
        "total_s": total,
        "stages": {
            label: {
                "calls": timer.calls[label],
                "seconds": timer.seconds[label],
                "share": timer.seconds[label] / total if total else 0.0,
                "us_per_call": 1e6 * timer.seconds[label] / max(timer.calls[label], 1),
            }
            for _, _, label in STAGES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    start = time.perf_counter()
    sessile_box.profile_table()
    table_s = time.perf_counter() - start
    report = {"environment": platform.platform(), "python": sys.version,
              "profile_table_build_s": table_s, "cases": {}}
    print(f"one-off Young-Laplace table build: {table_s * 1e3:.0f} ms (cached per process)\n")

    slowest = (None, None, -1.0)
    for name, bond, theta, noise in CASES:
        pts, p1, p2 = sessile_case(bond, theta, noise, seed=7, tilt_deg=7.0)
        report["cases"][name] = {"n_points": len(pts)}
        for mode in MODES:
            times, fit = [], None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fit = arc_spline.fit_arc_spline(pts, p1, p2, init=mode)
                times.append(time.perf_counter() - t0)
            wall = median(times)
            err = [fit.theta_p1_deg - theta, fit.theta_p2_deg - theta]
            report["cases"][name][mode] = {
                "median_s": wall, "runs_s": times, "n_arcs": fit.n_arcs,
                "levels": len(fit.levels), "init_used": fit.init, "error_deg": err,
                "breakdown": breakdown(pts, p1, p2, mode),
            }
            if wall > slowest[2]:
                slowest = (name, mode, wall)
            print(f"{name:20s} {mode:7s} pts={len(pts):4d} arcs={fit.n_arcs:2d} "
                  f"levels={len(fit.levels)} median={wall * 1e3:7.1f} ms "
                  f"err={err[0]:+.3f}/{err[1]:+.3f} deg")

    print("\nStage breakdown (inclusive share of wall time; calls x cost per call):")
    for name, case in report["cases"].items():
        for mode in MODES:
            stages = case[mode]["breakdown"]["stages"]
            parts = ", ".join(
                f"{k}={v['share'] * 100:4.1f}% ({v['calls']}x{v['us_per_call']:.0f}us)"
                for k, v in stages.items() if v["calls"]
            )
            print(f"  {name:20s} {mode:7s} {parts}")

    print("\nClothoid spline (physics prior), same cases:")
    clothoid_stages = [
        (clothoid_spline, "_refit", "refit (least_squares)"),
        (clothoid_spline._Problem, "_evaluate", "residual eval"),
        (clothoid_spline._Problem, "jacobian", "jacobian"),
        (clothoid_spline, "_curve", "foot-point quadrature"),
        (clothoid_spline, "solve_hermite", "hermite solve"),
        (clothoid_spline, "_model_bias", "model bias fit"),
        (sessile_box, "fit_side_box", "box golden search"),
    ]
    report["clothoid"] = {}
    for name, bond, theta, noise in CASES:
        pts, p1, p2 = sessile_case(bond, theta, noise, seed=7, tilt_deg=7.0)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fit = clothoid_spline.fit_clothoid_spline(pts, p1, p2)
            times.append(time.perf_counter() - t0)
        timer = StageTimer()
        with ExitStack() as stack:
            for owner, attr, label in clothoid_stages:
                stack.enter_context(timer.wrap(owner, attr, label))
            t0 = time.perf_counter()
            clothoid_spline.fit_clothoid_spline(pts, p1, p2)
            total = time.perf_counter() - t0
        err = [fit.theta_p1_deg - theta, fit.theta_p2_deg - theta]
        report["clothoid"][name] = {
            "median_s": median(times), "n_segments": fit.n_segments, "error_deg": err,
            "stages": {k: {"calls": timer.calls[k], "seconds": timer.seconds[k]} for _, _, k in clothoid_stages},
        }
        parts = ", ".join(
            f"{k}={timer.seconds[k] / total * 100:4.1f}% ({timer.calls[k]}x)"
            for _, _, k in clothoid_stages if timer.calls[k]
        )
        print(f"  {name:20s} segments={fit.n_segments:2d} median={median(times) * 1e3:7.1f} ms "
              f"err={err[0]:+.3f}/{err[1]:+.3f} deg | {parts}")

    print("\nRendered image: mask contour fit, then contrast refinement (2 rounds):")
    exact, p1, p2 = sessile_case(1.0, 110.0, contact_radius_px=120.0, tilt_deg=5.0)
    image = render_image(exact, p1, p2, seed=0)
    contour = mask_contour(image)
    timer = StageTimer()
    with ExitStack() as stack:
        for owner, attr, label in STAGES:
            stack.enter_context(timer.wrap(owner, attr, label))
        t0 = time.perf_counter()
        fit, interface = arc_spline.fit_sessile_arc_spline(contour, p1, p2)
        t1 = time.perf_counter()
        refined, _ = arc_spline.refine_on_image(image, fit, interface, p1, p2)
        t2 = time.perf_counter()
    report["image_case"] = {
        "mask_fit_s": t1 - t0, "refine_s": t2 - t1,
        "error_deg": [refined.theta_p1_deg - 110.0, refined.theta_p2_deg - 110.0],
        "stages": {k: {"calls": timer.calls[k], "seconds": timer.seconds[k]} for _, _, k in STAGES},
    }
    print(f"  mask fit {1e3 * (t1 - t0):.1f} ms, refinement {1e3 * (t2 - t1):.1f} ms, "
          f"error {refined.theta_p1_deg - 110.0:+.2f}/{refined.theta_p2_deg - 110.0:+.2f} deg")
    total = t2 - t0
    print("  " + ", ".join(
        f"{k}={timer.seconds[k] / total * 100:4.1f}% ({timer.calls[k]}x)"
        for _, _, k in STAGES if timer.calls[k]
    ))

    name, mode, _ = slowest
    bond, theta, noise = next((b, t, n) for c, b, t, n in CASES if c == name)
    pts, p1, p2 = sessile_case(bond, theta, noise, seed=7, tilt_deg=7.0)
    profiler = cProfile.Profile()
    profiler.enable()
    arc_spline.fit_arc_spline(pts, p1, p2, init=mode)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(args.top)
    print(f"\ncProfile of slowest case '{name}' ({mode}), sorted by own time:")
    print(stream.getvalue())
    report["cprofile_case"] = f"{name}/{mode}"
    report["cprofile_top"] = stream.getvalue()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
