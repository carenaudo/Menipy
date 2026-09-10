"""Paired warmed multi-plane estimation timings, including real lookup queries."""

import json
import sys
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from uuid import uuid4

import numpy as np


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from check_gui_execution import isolate

    root = Path(__file__).resolve().parents[1] / ".cache" / "multi-plane" / uuid4().hex
    isolate(root / "state")
    from menipy.pipelines.pendant import approximations as approx
    from tests.test_multi_plane_preparation import legacy_multi

    approx._selected_plane_lookup_all()
    ctx = SimpleNamespace(results={"r0_mm": 1.2})
    report = {}
    for count in (300, 10000):
        z = np.linspace(0, 3, count)
        profile = np.column_stack([np.sin(z), z])
        functions = {"original": legacy_multi, "prepared": approx.multi_selected_plane}
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                start = time.perf_counter()
                for _ in range(100):
                    outputs[name] = functions[name](ctx, profile, {})
                times[name].append((time.perf_counter() - start) / 100)
            assert outputs["original"] == outputs["prepared"]
        report[count] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds_per_call": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
