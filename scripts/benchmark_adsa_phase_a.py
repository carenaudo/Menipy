"""Run the non-blocking Phase-A ADSA timing/memory report."""

from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from adsa_conformance import evaluate_detection, generate_case  # noqa: E402


def main() -> int:
    manifest_path = ROOT / "tests" / "data" / "adsa_detector_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for spec in manifest["cases"]:
        generated = generate_case(spec, manifest["seed"])
        tracemalloc.start()
        started = time.perf_counter()
        _, metrics, reasons = evaluate_detection(generated, spec["pipeline"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "id": spec["id"],
                "pipeline": spec["pipeline"],
                "expected_valid": spec["expected_valid"],
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "runtime_ms": elapsed_ms,
                "peak_python_bytes": peak,
                "metrics": metrics,
            }
        )
    runtimes = [row["runtime_ms"] for row in rows]
    report = {
        "schema_version": "1.0",
        "seed": manifest["seed"],
        "cases": rows,
        "performance_non_blocking": {
            "median_runtime_ms": statistics.median(runtimes),
            "p95_runtime_ms": sorted(runtimes)[int(0.95 * (len(runtimes) - 1))],
            "max_peak_python_bytes": max(row["peak_python_bytes"] for row in rows),
        },
    }
    output = ROOT / "build" / "research" / "adsa_phase_a_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
