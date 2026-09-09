"""Paired residual/classification benchmarks with fixed scientific inputs."""

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

    root = (
        Path(__file__).resolve().parents[1] / ".cache" / "repeated-work" / uuid4().hex
    )
    isolate(root / "state")
    from menipy.common import solver
    from menipy.common import temporal_sessile as temporal
    from tests.test_numerical_repeated_work import (
        frames,
        legacy_pointwise,
        legacy_windows,
    )

    report = {}
    rng = np.random.default_rng(82)
    obs, model = rng.normal(size=(5000, 2)), rng.normal(size=(300, 2))
    prepared = solver._prepare_pointwise_residual(obs)
    times = {"original": [], "prepared": []}
    for repeat in range(6):
        for name in (
            ("original", "prepared") if repeat % 2 == 0 else ("prepared", "original")
        ):
            start = time.perf_counter()
            for _ in range(500):
                result = (
                    legacy_pointwise(obs, model)
                    if name == "original"
                    else prepared(model)
                )
            times[name].append(time.perf_counter() - start)
            np.testing.assert_array_equal(result, legacy_pointwise(obs, model))
    report["residual_500_evaluations"] = {
        name: median(values) for name, values in times.items()
    }
    report["temporal"] = []
    optimized_windows = temporal._velocity_windows
    for count in (1000, 3000):
        times = {"original": [], "windowed": []}
        outputs = {}
        for repeat in range(3):
            for name in (
                ("original", "windowed")
                if repeat % 2 == 0
                else ("windowed", "original")
            ):
                records = frames(count)
                with patch.object(
                    temporal,
                    "_velocity_windows",
                    legacy_windows if name == "original" else optimized_windows,
                ):
                    start = time.perf_counter()
                    deadband = temporal._assign_states(records)
                    times[name].append(time.perf_counter() - start)
                outputs[name] = (deadband, [f.model_dump() for f in records])
            assert outputs["original"] == outputs["windowed"]
        report["temporal"].append(
            {
                "frames": count,
                **{name: median(values) for name, values in times.items()},
            }
        )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
