"""Paired shared ODE integration and fit timings with exact comparisons."""

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
        Path(__file__).resolve().parents[1] / ".cache" / "ode-invariants" / uuid4().hex
    )
    isolate(root / "state")
    from menipy.math import young_laplace as yl
    from tests.test_ode_invariants import fitted_output, legacy_pendant, legacy_sessile

    report = {}
    for mode, current, reference in (
        ("pendant", yl.young_laplace_ode, legacy_pendant),
        ("sessile", yl.sessile_young_laplace_ode, legacy_sessile),
    ):
        for operation in ("50_integrations", "fit"):
            functions = {"original": reference, "optimized": current}
            times = {name: [] for name in functions}
            for repeat in range(6):
                outputs = {}
                for name in (
                    list(functions) if repeat % 2 == 0 else list(reversed(functions))
                ):
                    start = time.perf_counter()
                    if operation == "fit":
                        outputs[name] = fitted_output(functions[name])
                    else:
                        for _ in range(50):
                            outputs[name] = functions[name](
                                np.array([2.0, 0.25]), {}, {"height_mm": 1.8}
                            )
                    times[name].append(time.perf_counter() - start)
                if operation == "fit":
                    assert outputs["original"] == outputs["optimized"]
                else:
                    np.testing.assert_array_equal(
                        outputs["original"], outputs["optimized"]
                    )
            report[f"{mode}_{operation}"] = {
                "median_seconds": {
                    name: median(values) for name, values in times.items()
                },
                "runs_seconds": times,
            }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
