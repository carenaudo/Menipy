"""Paired full temporal classification timings for exact regression convergence."""

import json
import sys
import time
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from check_gui_execution import isolate

    root = (
        Path(__file__).resolve().parents[1]
        / ".cache"
        / "slope-convergence"
        / uuid4().hex
    )
    isolate(root / "state")
    from menipy.common import temporal_sessile as temporal
    from tests.test_numerical_repeated_work import frames
    from tests.test_robust_slope_convergence import legacy_slope

    functions = {"original": legacy_slope, "converged": temporal._robust_slope}
    report = {}
    for kind in ("flat", "varying"):
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                records = frames(3000)
                if kind == "flat":
                    for record in records:
                        record.half_width_mm = 2.0
                with patch.object(temporal, "_robust_slope", functions[name]):
                    start = time.perf_counter()
                    deadband = temporal._assign_states(records)
                    times[name].append(time.perf_counter() - start)
                outputs[name] = (deadband, [f.model_dump() for f in records])
            assert outputs["original"] == outputs["converged"]
        report[kind] = {
            "frames": 3000,
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
