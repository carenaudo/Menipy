"""Compare contact-angle methods on the bundled sessile sample images.

Runs the application chain -- ``AutoCalibrator`` detection, then the sessile
pipeline's metric stage -- once per contact-angle method on each sample in
``data/samples`` and prints left/right angles, left-right asymmetry and timing.
The samples have no ground truth, so the report shows agreement between
methods, side symmetry and the arc spline's own diagnostics (arc count, whether
the physics prior held, the box-matched Young-Laplace angle, residual).

Optionally writes an overlay PNG per sample with the fitted arc spline.

This tool measures; it changes no application behavior.

Usage
-----
``uv run python tools/compare_arc_spline_samples.py [--overlays DIR] [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLES = (
    "gota depositada 1.png",
    "prueba sesil 2.png",
    "sessile_3.jpeg",
    "sessile_clean_reference.png",
    "sessile_needle_reference.png",
)
METHODS = ("tangent", "circle_fit", "auto_residual", "lbadsa", "arc_spline")


def load_plugins() -> None:
    """Register the preprocessors the application loads at startup."""
    import menipy.common.detection_helpers  # noqa: F401  (adds plugins/ to sys.path)

    for name in ("preproc_detect_substrate", "preproc_detect_needle", "preproc_detect_drop",
                 "preproc_detect_roi", "preproc_auto_detect"):
        try:
            import_module(name)
        except ImportError:
            pass


def measure(path: Path, calibration, method: str) -> dict:
    from menipy.pipelines.sessile.stages import SessilePipeline

    start = time.perf_counter()
    ctx = SessilePipeline().run_with_plan(
        only=["compute_metrics"],
        image=str(path),
        drop_contour=calibration.drop_contour,
        contact_points=calibration.contact_points,
        apex_point=calibration.apex_point,
        needle_rect=calibration.needle_rect,
        roi_rect=calibration.roi_rect,
        substrate_line=calibration.substrate_line,
        contact_angle_method=method,
        calibration_params={"needle_diameter_mm": 1.0, "drop_density_kg_m3": 1000.0,
                            "fluid_density_kg_m3": 1.2},
    )
    elapsed = time.perf_counter() - start
    res = ctx.results or {}
    return {
        "left": float(res.get("theta_left_deg", float("nan"))),
        "right": float(res.get("theta_right_deg", float("nan"))),
        "seconds": elapsed,
        "arc_spline": res.get("arc_spline"),
        "model": res.get("arc_spline_model_contour_xy"),
    }


def draw_overlay(image: np.ndarray, calibration, model, out: Path) -> None:
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contour = np.asarray(calibration.drop_contour, float).reshape(-1, 2)
    cv2.polylines(canvas, [np.round(contour).astype(np.int32)], True, (0, 200, 255), 1)
    if model:
        cv2.polylines(canvas, [np.round(np.asarray(model)).astype(np.int32)], False, (255, 255, 0), 1)
    for p in calibration.contact_points or ():
        cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 3, (0, 0, 255), -1)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overlays", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    from menipy.common.auto_calibrator import AutoCalibrator

    load_plugins()
    report = {}
    print(f"{'sample':30s} " + " ".join(f"{m:>17s}" for m in METHODS))
    for name in SAMPLES:
        path = ROOT / "data" / "samples" / name
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        calibration = AutoCalibrator(image, "sessile").detect_all()
        if calibration.drop_contour is None or not calibration.contact_points:
            print(f"{name:30s} no drop contour / contact points detected")
            continue
        row = {}
        for method in METHODS:
            try:
                row[method] = measure(path, calibration, method)
            except Exception as exc:  # report and continue with the next method
                row[method] = {"left": float("nan"), "right": float("nan"), "seconds": 0.0, "error": repr(exc)}
        report[name] = {m: {k: v for k, v in r.items() if k != "model"} for m, r in row.items()}
        print(f"{name:30s} " + " ".join(
            f"{r['left']:7.1f}/{r['right']:5.1f}{'' if 'error' not in r else '!'}" for r in row.values()
        ))
        diag = row["arc_spline"].get("arc_spline") or {}
        if diag.get("accepted"):
            phys = diag.get("physics") or {}
            yl = [phys.get(s, {}).get("theta_deg", float("nan")) for s in ("p1", "p2")]
            print(f"{'':30s} arcs={diag['n_arcs']} ({diag['n_arcs_p1']}+{diag['n_arcs_p2']}) "
                  f"init={diag['init']} physics_adequate={phys.get('adequate')} "
                  f"YL box={yl[0]:.1f}/{yl[1]:.1f} rmse={diag['rmse_px']:.2f}px "
                  f"noise={phys.get('edge_noise_px', float('nan')):.2f}px "
                  f"refined={diag['image_refined']} "
                  f"time={row['arc_spline']['seconds'] * 1e3:.0f} ms (tangent "
                  f"{row['tangent']['seconds'] * 1e3:.0f} ms)")
        elif diag:
            print(f"{'':30s} arc spline rejected: {', '.join(diag.get('rejection_reasons', []))} "
                  f"(arcs={diag.get('n_arcs')}, rmse={diag.get('rmse_px', float('nan')):.2f}px)")
        if args.overlays:
            args.overlays.mkdir(parents=True, exist_ok=True)
            draw_overlay(image, calibration, row["arc_spline"].get("model"),
                         args.overlays / f"{Path(name).stem}_arc_spline.png")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
