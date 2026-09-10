"""Paired temporal classification timing for scalar adjacent velocities."""

import json
import sys
import time
from pathlib import Path
from statistics import median
from uuid import uuid4


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from check_gui_execution import isolate

    root = (
        Path(__file__).resolve().parents[1]
        / ".cache"
        / "adjacent-velocity"
        / uuid4().hex
    )
    isolate(root / "state")
    from menipy.common.temporal_sessile import _assign_states
    from tests.test_adjacent_velocity import legacy_states
    from tests.test_numerical_repeated_work import frames

    report = {}
    for kind in ("flat", "varying"):
        functions = {"original": legacy_states, "scalar": _assign_states}
        times = {name: [] for name in functions}
        for repeat in range(6):
            outputs = {}
            for name in (
                list(functions) if repeat % 2 == 0 else list(reversed(functions))
            ):
                records = frames(3000)
                if kind == "flat":
                    for record in records:
                        record.half_width_mm = 2.0
                start = time.perf_counter()
                deadband = functions[name](records)
                times[name].append(time.perf_counter() - start)
                outputs[name] = (deadband, [f.model_dump() for f in records])
            assert outputs["original"] == outputs["scalar"]
        report[kind] = {
            "frames": 3000,
            "median_seconds": {name: median(values) for name, values in times.items()},
            "runs_seconds": times,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
