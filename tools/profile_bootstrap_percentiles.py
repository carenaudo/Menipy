"""Paired full bootstrap timing for shared confidence-interval percentiles."""

import json
import sys
import time
from pathlib import Path
from statistics import median
from uuid import uuid4

import numpy as np


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from check_gui_execution import isolate

    root = (
        Path(__file__).resolve().parents[1]
        / ".cache"
        / "bootstrap-percentiles"
        / uuid4().hex
    )
    isolate(root / "state")
    from menipy.common.temporal_sessile import _bootstrap_stats
    from tests.test_bootstrap_percentiles import separate_percentiles

    report = {}
    for count, repeats in ((5, 200), (1000, 10)):
        values = np.random.default_rng(42).normal(90, 10, count)
        functions = {"separate": separate_percentiles, "combined": _bootstrap_stats}
        times = {name: [] for name in functions}
        for run in range(6):
            outputs = {}
            for name in list(functions) if run % 2 == 0 else list(reversed(functions)):
                start = time.perf_counter()
                for _ in range(repeats):
                    outputs[name] = functions[name](values)
                times[name].append((time.perf_counter() - start) / repeats)
            assert outputs["separate"] == outputs["combined"]
        report[count] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds_per_call": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
