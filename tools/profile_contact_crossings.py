"""Paired straight-baseline contact detection timings with exact outputs."""

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
        / "contact-crossings"
        / uuid4().hex
    )
    isolate(root / "state")
    from menipy.common.geometry import find_contact_points_from_contour
    from tests.test_contact_crossing_selection import legacy_contacts

    report = {}
    for count in (1000, 10000):
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        contour = np.column_stack([100 * np.cos(theta), 100 * np.sin(theta)])
        line = ((-150, 50), (150, 50))
        functions = {
            "original": legacy_contacts,
            "selected": find_contact_points_from_contour,
        }
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                start = time.perf_counter()
                for _ in range(100):
                    outputs[name] = functions[name](contour, line)
                times[name].append((time.perf_counter() - start) / 100)
            np.testing.assert_array_equal(outputs["original"], outputs["selected"])
        report[count] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds_per_call": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
