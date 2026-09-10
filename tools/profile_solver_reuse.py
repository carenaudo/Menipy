"""Audit exact integration reuse opportunities without changing numerical work."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np


def main():
    from check_gui_execution import isolate

    root = Path(__file__).resolve().parents[1] / ".cache" / "solver-reuse" / uuid4().hex
    isolate(root / "state")
    from profile_analysis import worker

    from menipy.common import solver

    original_run = solver.run
    fits = []

    def traced_run(ctx, *, integrator, **kwargs):
        evaluations = []
        seen = set()

        def traced_integrator(params, physics, geometry):
            key = np.asarray(params).tobytes()
            repeated = key in seen
            seen.add(key)
            start = time.perf_counter()
            result = integrator(params, physics, geometry)
            evaluations.append(
                {
                    "params": params.tolist(),
                    "repeated": repeated,
                    "seconds": time.perf_counter() - start,
                    "profile_points": len(result),
                }
            )
            return result

        result = original_run(ctx, integrator=traced_integrator, **kwargs)
        fits.append(
            {
                "evaluations": evaluations,
                "unique_parameters": len(seen),
                "total_evaluations": len(evaluations),
                "repeated_integration_seconds": sum(
                    e["seconds"] for e in evaluations if e["repeated"]
                ),
            }
        )
        return result

    with patch.object(solver, "run", traced_run):
        worker(
            SimpleNamespace(output=root, case="sessile", repeats=1, profile_first=False)
        )
    (root / "reuse.json").write_text(json.dumps(fits, indent=2))
    print(root)
    print(
        json.dumps(
            [{k: v for k, v in fit.items() if k != "evaluations"} for fit in fits],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
