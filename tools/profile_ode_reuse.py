"""Paired strict ODE and full-fit timings; run through uv with the test extra."""

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

    root = Path(__file__).resolve().parents[1] / ".cache" / "ode-reuse" / uuid4().hex
    isolate(root / "state")
    from menipy.pipelines.pendant import strict_young_laplace as strict
    from tests.test_ode_callback_reuse import legacy_integrate
    from tests.test_pendant_fit_cache import fit_input

    functions = {
        "original": legacy_integrate,
        "reuse": strict.integrate_young_laplace_profile_mm,
    }
    inputs = fit_input(0.2)
    records = {}
    for mode in ("50_integrations", "full_fit"):
        times = {name: [] for name in functions}
        for repeat in range(8):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                with patch.object(
                    strict, "integrate_young_laplace_profile_mm", functions[name]
                ):
                    start = time.perf_counter()
                    if mode == "full_fit":
                        outputs[name] = json.dumps(
                            strict.fit_pendant_young_laplace_strict(inputs),
                            sort_keys=True,
                        )
                    else:
                        for _ in range(50):
                            outputs[name] = functions[name](
                                1.2, 0.6, target_height_mm=2.0
                            )
                    times[name].append(time.perf_counter() - start)
            if mode == "full_fit":
                assert outputs["original"] == outputs["reuse"]
            else:
                np.testing.assert_array_equal(outputs["original"], outputs["reuse"])
        records[mode] = {
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds": times,
        }
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
