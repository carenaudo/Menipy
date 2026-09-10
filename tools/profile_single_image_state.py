"""Paired single-image pipeline timing with exact persisted result checks."""

import hashlib
import json
import logging
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from unittest.mock import patch
from uuid import uuid4

import numpy as np


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from check_gui_execution import isolate

    root = repo / ".cache" / "single-image-state" / uuid4().hex
    isolate(root / "state")
    import cv2

    from menipy.common import solver
    from menipy.common.auto_calibrator import run_auto_calibration
    from menipy.models.results import build_persisted_analysis
    from menipy.pipelines.discover import PIPELINE_MAP
    from menipy.pipelines.pendant import strict_young_laplace as strict
    from tests import ode_state_reference as reference

    logging.disable(logging.CRITICAL)
    original_run = solver.run

    def reference_run(ctx, *, integrator, **kwargs):
        assert integrator.__name__ in ("young_laplace_ode", "sessile_young_laplace_ode")
        return original_run(ctx, integrator=getattr(reference, integrator.__name__), **kwargs)

    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in ("duration_ms", "timings_ms", "elapsed_ms", "time_ms")}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    report = {"platform": platform.platform(), "python": sys.version, "cases": {}}
    for mode in ("sessile", "pendant"):
        filename = "sessile_needle_reference.png" if mode == "sessile" else "pendant_water_reference.png"
        source = repo / "data" / "samples" / filename
        image = cv2.imread(str(source))
        calibration = run_auto_calibration(image, mode)
        parameters = {
            "image": str(source), "scale": {"px_per_mm": 100.0},
            "needle_diameter_mm": 1.0, "roi": calibration.roi_rect,
            "needle_rect": calibration.needle_rect, "substrate_line": calibration.substrate_line,
            "physics": {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665},
        }
        times = {"original": [], "optimized": []}
        for repeat in range(7):
            outputs = {}
            for name in (list(times) if repeat % 2 == 0 else list(reversed(times))):
                with ExitStack() as stack:
                    if name == "original":
                        stack.enter_context(patch.object(solver, "run", reference_run))
                        stack.enter_context(patch.object(strict, "integrate_young_laplace_profile_mm",
                                                         reference.integrate_young_laplace_profile_mm))
                    start = time.perf_counter()
                    ctx = PIPELINE_MAP[mode]().run(**parameters)
                    elapsed = time.perf_counter() - start
                    outputs[name] = clean(build_persisted_analysis(ctx))
                    if repeat:
                        times[name].append(elapsed)
            if outputs["original"] != outputs["optimized"]:
                (root / "mismatch.json").write_text(json.dumps(outputs, indent=2, default=str))
                raise AssertionError(f"{mode}: {root / 'mismatch.json'}")
        report["cases"][mode] = {
            "dimensions": list(image.shape), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "parameters": parameters, "runs_seconds": times,
            "median_seconds": {k: median(v) for k, v in times.items()},
            "exact_persisted_output": True,
        }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
