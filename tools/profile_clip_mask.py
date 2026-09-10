"""Paired contour clipping timings with exact geometry comparisons."""

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

    root = Path(__file__).resolve().parents[1] / ".cache" / "clip-mask" / uuid4().hex
    isolate(root / "state")
    from menipy.pipelines.sessile.geometry import clip_contour_to_substrate
    from tests.test_clip_mask_reuse import legacy_clip

    report = {}
    for count in (1000, 10000):
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        points = np.column_stack([100 * np.cos(theta), 100 * np.sin(theta)])
        functions = {"original": legacy_clip, "mask": clip_contour_to_substrate}
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                start = time.perf_counter()
                for _ in range(10):
                    outputs[name] = functions[name](
                        points, ((-150, 50), (150, 50)), (0, -100)
                    )
                times[name].append((time.perf_counter() - start) / 10)
            np.testing.assert_array_equal(outputs["original"][0], outputs["mask"][0])
            assert outputs["original"][1] == outputs["mask"][1]
        report[count] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds_per_call": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
