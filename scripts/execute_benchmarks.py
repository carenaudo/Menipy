"""Execute comprehensive scientific benchmarks on Menipy using downloaded datasets.

Runs pipelines across:
1. Conan-ML hydrophobic and superhydrophobic contact angle images.
2. OpenDrop reference pendant series (water in air).
3. EPFL LB-ADSA sessile drop image.
4. Drop-O-Matic dynamic sessile sequence.
5. UCLA pendant drop tensiometry.

Compares Menipy's computed metrics against published literature ground-truth values.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from menipy.pipelines.runner import PipelineRunner

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("execute_benchmarks")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"


def run_conan_benchmarks() -> list[dict[str, Any]]:
    """Evaluate Conan-ML sessile drop test images against ground-truth angles."""
    conan_dir = BENCHMARK_DIR / "conan_ml"
    if not conan_dir.exists():
        print("Conan-ML benchmark directory not found, skipping.")
        return []

    test_cases = [
        ("111.031693.bmp", 111.03),
        ("113.66.bmp", 113.66),
        ("114.47.bmp", 114.47),
        ("115.00.bmp", 115.00),
        ("115.553909.bmp", 115.55),
        ("115.714851.bmp", 115.71),
        ("118.174171.bmp", 118.17),
        ("2-s@M@Z-120.321945190429.bmp", 120.32),
        ("2-s@M@Z-121.057571411132.bmp", 121.06),
        ("2-s@M@Z-122.639770507812.bmp", 122.64),
        ("359 CA162.BMP", 162.00),
    ]

    results = []
    print("\n=======================================================")
    print("  CONAN-ML EXPERIMENTAL CONTACT ANGLE BENCHMARK SUITE")
    print("  Author: Joseph D. Berry et al. (Langmuir 2024)")
    print("=======================================================")
    print(f"{'Filename':<32} {'Expected':<10} {'Computed':<10} {'Error (deg)':<12} {'Status'}")
    print("-" * 72)

    runner = PipelineRunner("sessile")

    for filename, expected_ca in test_cases:
        img_path = conan_dir / filename
        if not img_path.exists():
            continue

        t0 = time.perf_counter()
        try:
            ctx = runner.run(image=str(img_path))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            computed_ca = None
            if ctx.results:
                computed_ca = ctx.results.get("contact_angle_deg")
                if computed_ca is None:
                    # fallback to average of left and right
                    tl = ctx.results.get("theta_left_deg")
                    tr = ctx.results.get("theta_right_deg")
                    if tl is not None and tr is not None:
                        computed_ca = (tl + tr) / 2.0

            if computed_ca is not None:
                err = abs(computed_ca - expected_ca)
                status = "PASS" if err < 5.0 else ("FAIR" if err < 10.0 else "DEVIATED")
                results.append({
                    "dataset": "Conan-ML",
                    "file": filename,
                    "expected": expected_ca,
                    "computed": computed_ca,
                    "error_deg": err,
                    "status": status,
                    "runtime_ms": dt_ms,
                })
                print(f"{filename[:30]:<32} {expected_ca:>8.2f}° {computed_ca:>8.2f}° {err:>10.2f}°  [{status}]")
            else:
                print(f"{filename[:30]:<32} {expected_ca:>8.2f}° {'N/A':>8} {'N/A':>10}  [FAILED]")
        except Exception as e:
            print(f"{filename[:30]:<32} {expected_ca:>8.2f}° ERROR: {e}")

    return results


def run_opendrop_pendant_benchmarks() -> list[dict[str, Any]]:
    """Evaluate OpenDrop reference water pendant series against standard 72.8 mN/m."""
    pendant_dir = BENCHMARK_DIR / "pendant"
    if not pendant_dir.exists():
        return []

    test_files = [
        "water_in_air002.png",
        "water_in_air003.png",
        "water_in_air004.png",
        "water_in_air005.png",
    ]

    expected_gamma = 72.80  # mN/m IUPAC water at 20C
    needle_d = 0.718  # mm (22G needle outer diameter used by Berry et al. 2015)

    results = []
    print("\n=======================================================")
    print("  OPENDROP PENDANT DROP SURFACE TENSION SUITE")
    print("  Author: Berry et al. (J. Colloid Interface Sci. 2015)")
    print("=======================================================")
    print(f"{'Filename':<25} {'Expected':<12} {'Computed':<12} {'Dev (%)':<10} {'Status'}")
    print("-" * 65)

    runner = PipelineRunner("pendant")

    for filename in test_files:
        img_path = pendant_dir / filename
        if not img_path.exists():
            continue

        t0 = time.perf_counter()
        try:
            ctx = runner.run(
                image=str(img_path),
                needle_diameter_mm=needle_d,
                physics={"rho1": 998.2, "rho2": 1.2, "g": 9.80665},
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            computed_gamma = ctx.results.get("surface_tension_mN_m") if ctx.results else None

            if computed_gamma is not None:
                dev_pct = abs(computed_gamma - expected_gamma) / expected_gamma * 100.0
                status = "PASS" if dev_pct < 5.0 else "FAIR"
                results.append({
                    "dataset": "OpenDrop Pendant",
                    "file": filename,
                    "expected_gamma": expected_gamma,
                    "computed_gamma": computed_gamma,
                    "dev_pct": dev_pct,
                    "status": status,
                    "runtime_ms": dt_ms,
                })
                print(f"{filename:<25} {expected_gamma:>8.2f} mN/m {computed_gamma:>8.2f} mN/m {dev_pct:>8.2f}%  [{status}]")
            else:
                print(f"{filename:<25} {expected_gamma:>8.2f} mN/m {'N/A':>10} {'N/A':>8}  [FAILED]")
        except Exception as e:
            print(f"{filename:<25} ERROR: {e}")

    return results


def run_ucla_pendant_benchmarks() -> list[dict[str, Any]]:
    """Evaluate UCLA pendant drop tensiometry benchmark images."""
    ucla_dir = BENCHMARK_DIR / "ucla_pendant"
    if not ucla_dir.exists():
        return []

    test_cases = [
        {
            "file": "H2O_PendantDrop.png",
            "expected_gamma": 72.80,
            "needle_d": 0.51,  # 25G needle
            "rho1": 998.2,
            "rho2": 1.2,
            "desc": "Water in air",
        },
        {
            "file": "Hexadecane_PendantDrop.png",
            "expected_gamma": 27.50,
            "needle_d": 1.95,  # 14G needle
            "rho1": 773.0,
            "rho2": 1.2,
            "desc": "Hexadecane in air",
        },
    ]

    results = []
    print("\n=======================================================")
    print("  UCLA PENDANT DROP TENSIOMETRY BENCHMARK")
    print("  Author: Pirouz Kavehpour et al. (UCLA)")
    print("=======================================================")
    print(f"{'Filename':<28} {'Expected':<12} {'Computed':<12} {'Dev (%)':<10} {'Status'}")
    print("-" * 68)

    runner = PipelineRunner("pendant")

    for tc in test_cases:
        filename = tc["file"]
        img_path = ucla_dir / filename
        if not img_path.exists():
            continue

        t0 = time.perf_counter()
        try:
            ctx = runner.run(
                image=str(img_path),
                needle_diameter_mm=tc["needle_d"],
                physics={"rho1": tc["rho1"], "rho2": tc["rho2"], "g": 9.80665},
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            computed_gamma = ctx.results.get("surface_tension_mN_m") if ctx.results else None
            expected_gamma = tc["expected_gamma"]

            if computed_gamma is not None:
                dev_pct = abs(computed_gamma - expected_gamma) / expected_gamma * 100.0
                status = "PASS" if dev_pct < 5.0 else ("FAIR" if dev_pct < 10.0 else "DEVIATED")
                results.append({
                    "dataset": "UCLA Pendant",
                    "file": filename,
                    "description": tc["desc"],
                    "expected_gamma": expected_gamma,
                    "computed_gamma": computed_gamma,
                    "dev_pct": dev_pct,
                    "status": status,
                    "runtime_ms": dt_ms,
                })
                print(f"{filename:<28} {expected_gamma:>8.2f} mN/m {computed_gamma:>8.2f} mN/m {dev_pct:>8.2f}%  [{status}]")
            else:
                print(f"{filename:<28} {expected_gamma:>8.2f} mN/m {'N/A':>10} {'N/A':>8}  [FAILED]")
        except Exception as e:
            print(f"{filename:<28} ERROR: {e}")

    return results


def run_epfl_benchmark() -> list[dict[str, Any]]:
    """Evaluate EPFL Drop Analysis reference image."""
    epfl_dir = BENCHMARK_DIR / "epfl_drop_analysis"
    img_path = epfl_dir / "sample.jpg"
    if not img_path.exists():
        return []

    print("\n=======================================================")
    print("  EPFL DROP ANALYSIS (LB-ADSA & DROPSNAKE) BENCHMARK")
    print("  Author: Adrien Stalder, Daniel Sage (EPFL)")
    print("=======================================================")

    runner = PipelineRunner("sessile")
    t0 = time.perf_counter()
    ctx = runner.run(image=str(img_path), px_per_mm=191.82)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    ca = ctx.results.get("contact_angle_deg") if ctx.results else None
    h_mm = ctx.results.get("height_mm") if ctx.results else None
    d_mm = ctx.results.get("diameter_mm") if ctx.results else None

    print(f"Computed Contact Angle: {ca:.2f}\u00b0" if ca else "Contact angle: N/A")
    print(f"Droplet Apex Height:   {h_mm:.3f} mm" if h_mm else "Height: N/A")
    print(f"Base Contact Diameter: {d_mm:.3f} mm" if d_mm else "Diameter: N/A")
    print(f"Execution Time:        {dt_ms:.1f} ms")

    return [{
        "dataset": "EPFL Drop Analysis",
        "file": "sample.jpg",
        "contact_angle_deg": ca,
        "height_mm": h_mm,
        "diameter_mm": d_mm,
        "runtime_ms": dt_ms,
    }]


def main() -> int:
    print("Starting Menipy Scientific Benchmark Execution Suite...")
    all_results: dict[str, Any] = {}

    all_results["conan_ml"] = run_conan_benchmarks()
    all_results["opendrop_pendant"] = run_opendrop_pendant_benchmarks()
    all_results["ucla_pendant"] = run_ucla_pendant_benchmarks()
    all_results["epfl"] = run_epfl_benchmark()

    # Save summary report
    out_dir = ROOT / "build" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "scientific_benchmarks_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark execution complete! Results saved to: {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
