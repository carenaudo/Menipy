"""Paired envelope timings with exact numerical comparisons.

Run with uv run --extra test python tools/profile_pendant_envelope.py.
"""

import json
import platform
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
        / "envelope-profile"
        / uuid4().hex
    )
    isolate(root / "state")
    from menipy.pipelines.pendant.strict_young_laplace import (
        build_pendant_profile_envelope_mm,
    )
    from tests.test_pendant_envelope_grouping import legacy_envelope

    records = []
    for count in (1000, 5000, 20000):
        # Two contour sides, approximately one point per pixel row per side.
        z = np.arange(count // 2, dtype=float)
        radius = 100 * np.sin(np.linspace(0, np.pi, len(z)))
        contour = np.column_stack([np.r_[radius, -radius], -np.r_[z, z]])
        kwargs = {"axis_x_px": 0, "apex_y_px": 0, "px_per_mm": 100}
        functions = {
            "original": legacy_envelope,
            "grouped": build_pendant_profile_envelope_mm,
        }
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                start = time.perf_counter()
                outputs[name] = functions[name](contour, **kwargs)
                times[name].append(time.perf_counter() - start)
            np.testing.assert_array_equal(outputs["original"], outputs["grouped"])
        records.append(
            {
                "points": count,
                "height_bins": len(z),
                "median_seconds": {
                    name: median(values) for name, values in times.items()
                },
                "runs_seconds": times,
            }
        )
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "records": records,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
