"""Paired full evolution timings for shared gradient sampling."""

import json
import platform
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

    root = Path(__file__).resolve().parents[1] / ".cache" / "snake-forces" / uuid4().hex
    isolate(root / "state")
    from menipy.math import active_contour as snake
    from tests.test_active_contour import make_synthetic_circle_image
    from tests.test_snake_force_reuse import original_forces

    image = make_synthetic_circle_image()
    optimized = snake.compute_external_forces
    report = {"environment": platform.platform(), "python": sys.version,
              "image_shape": list(image.shape), "cases": {}}
    for label, line, flux in (("default", 0, 0), ("combined", 0.2, 0.3)):
        for n in (80, 300):
            cfg = snake.ActiveContourConfig(max_iterations=15, w_line=line, w_flux=flux)
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([60 + 33 * np.cos(theta), 60 + 33 * np.sin(theta)])
            functions = {"original": original_forces, "shared": optimized}
            times = {name: [] for name in functions}
            for func in functions.values():
                with patch.object(snake, "compute_external_forces", func):
                    snake.evolve_active_contour(image, points, config=cfg)
            for repeat in range(6):
                outputs = {}
                for name in (list(functions) if repeat % 2 == 0 else list(reversed(functions))):
                    with patch.object(snake, "compute_external_forces", functions[name]):
                        start = time.perf_counter()
                        for _ in range(30):
                            outputs[name] = snake.evolve_active_contour(image, points, config=cfg)
                        times[name].append((time.perf_counter() - start) / 30)
                for field, value in vars(outputs["original"]).items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, getattr(outputs["shared"], field))
                    else:
                        assert value == getattr(outputs["shared"], field)
            report["cases"][f"{label}_{n}"] = {
                "median_seconds": {name: median(values) for name, values in times.items()},
                "runs_seconds": times, "iterations": outputs["shared"].iterations,
                "weights": {"line": line, "flux": flux},
            }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
