"""Paired first-ever lookup generation, bypassing disk and in-memory tables."""

import json
import platform
import sys
import time
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from check_gui_execution import isolate

    root = repo / ".cache/cold-lookup-build" / uuid4().hex
    isolate(root / "state")
    from menipy.pipelines.pendant import approximations as approx
    from tests.test_lookup_derivative_reuse import (
        assert_tables_equal,
        uncached_integrate,
    )

    functions = {"original": uncached_integrate, "optimized": approx.integrate_young_laplace_profile_mm}
    times = {name: [] for name in functions}
    for repeat in range(4):
        outputs = {}
        for name in (list(functions) if repeat % 2 == 0 else list(reversed(functions))):
            with patch.object(approx, "integrate_young_laplace_profile_mm", functions[name]):
                start = time.perf_counter()
                outputs[name] = approx._build_selected_plane_lookup()
                times[name].append(time.perf_counter() - start)
        assert_tables_equal(outputs["original"], outputs["optimized"])
    report = {"platform": platform.platform(), "python": sys.version,
              "grid": [24, 24], "runs_seconds": times,
              "median_seconds": {name: median(values) for name, values in times.items()},
              "exact_table_outputs": True}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
