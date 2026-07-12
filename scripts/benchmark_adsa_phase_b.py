"""Write non-blocking Phase-B geometry evidence to build/research."""

from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from adsa_geometry_conformance import (  # noqa: E402
    evaluate_geometry_case,
    generate_geometry_case,
)


def main() -> int:
    manifest = json.loads((ROOT / "tests" / "data" / "adsa_geometry_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for spec in manifest["cases"]:
        case = generate_geometry_case(spec, manifest["seed"])
        tracemalloc.start()
        started = time.perf_counter()
        outcome = evaluate_geometry_case(case, spec)
        elapsed = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"id": spec["id"], "kind": spec["kind"], "accepted": outcome["accepted"], "runtime_ms": elapsed, "peak_python_bytes": peak, "outcome": outcome})
    runtimes = [row["runtime_ms"] for row in rows]
    report = {"schema_version": "1.0", "seed": manifest["seed"], "cases": rows, "performance_non_blocking": {"median_runtime_ms": statistics.median(runtimes), "p95_runtime_ms": sorted(runtimes)[int(0.95 * (len(runtimes) - 1))], "max_peak_python_bytes": max(row["peak_python_bytes"] for row in rows)}}
    output = ROOT / "build" / "research" / "adsa_phase_b_geometry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    except PermissionError:
        # Some developer workspaces keep build/ artifacts read-only.  Preserve
        # the same report in the ignored temporary tree so the benchmark still
        # completes; CI normally writes the canonical build/research path.
        output = ROOT / ".tmp" / "adsa_phase_b_geometry.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
