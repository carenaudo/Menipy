"""Paired complete snake evolution timing with warm matrix reuse."""

import json
import sys
import time
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4

import numpy as np


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from check_gui_execution import isolate

    root = Path(__file__).resolve().parents[1] / ".cache" / "snake-matrix" / uuid4().hex
    isolate(root / "state")
    from menipy.math import active_contour as snake
    from tests.test_active_contour import make_synthetic_circle_image
    from tests.test_snake_matrix_cache import uncached_inverse

    image = make_synthetic_circle_image()
    cfg = snake.ActiveContourConfig(max_iterations=15)
    cached = snake._shape_inverse
    report = {}
    for n in (80, 200, 300):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
        cached(n, cfg, snake.SnakeBoundaryCondition.PERIODIC)
        functions = {"original": uncached_inverse, "cached": cached}
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                with patch.object(snake, "_shape_inverse", functions[name]):
                    start = time.perf_counter()
                    outputs[name] = snake.evolve_active_contour(
                        image, points, config=cfg
                    )
                    times[name].append(time.perf_counter() - start)
            for field, value in vars(outputs["original"]).items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(
                        value, getattr(outputs["cached"], field)
                    )
                else:
                    assert value == getattr(outputs["cached"], field)
        report[n] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
