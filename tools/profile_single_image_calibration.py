"""Paired single-image calibration timing with exact output comparisons."""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from check_gui_execution import isolate

    root = repo / ".cache" / "single-image-calibration" / uuid4().hex
    isolate(root / "state")
    import cv2

    from menipy.common import sessile_detection as detection
    from menipy.common.auto_calibrator import run_auto_calibration
    from tests.test_needle_single_run import (
        assert_calibration_equal,
        original_center_run,
    )

    report = {"platform": platform.platform(), "python": sys.version, "cases": {}}
    functions = {"original": original_center_run, "optimized": detection._center_run}
    for mode, filename in [("sessile", "sessile_needle_reference.png"),
                           ("pendant", "pendant_water_reference.png")]:
        source = repo / "data/samples" / filename
        image = cv2.imread(str(source))
        times = {name: [] for name in functions}
        run_auto_calibration(image, mode)
        for repeat in range(6):
            outputs = {}
            for name in (list(functions) if repeat % 2 == 0 else list(reversed(functions))):
                with patch.object(detection, "_center_run", functions[name]):
                    start = time.perf_counter()
                    for _ in range(10):
                        outputs[name] = run_auto_calibration(image, mode)
                    times[name].append((time.perf_counter() - start) / 10)
            assert_calibration_equal(outputs["original"], outputs["optimized"])
        report["cases"][mode] = {
            "shape": list(image.shape), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "runs_seconds": times, "median_seconds": {k: median(v) for k, v in times.items()},
            "exact_calibration_output": True,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
